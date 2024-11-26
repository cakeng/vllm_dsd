import pickle as pk
import matplotlib.pyplot as plt
from typing import Dict
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

_BATCH_SIZE_ALIGNMENT = 32


def _get_graph_batch_size(batch_size: int) -> int:
    """Returns the padded batch size given actual batch size.

    Batch sizes are 1, 2, 4, _BATCH_SIZE_ALIGNMENT,
    2*_BATCH_SIZE_ALIGNMENT, 3*_BATCH_SIZE_ALIGNMENT...
    """
    if batch_size <= 32:
        return 1 << (batch_size - 1).bit_length()
    else:
        return ((batch_size + _BATCH_SIZE_ALIGNMENT - 1) //
                _BATCH_SIZE_ALIGNMENT * _BATCH_SIZE_ALIGNMENT)



def _fit_2d_latency_models(
        seq_data_dict: Dict[int, Dict[int, float]]) -> LinearRegression:
    seq_lens = []
    batch_sizes = []
    query_lens = []
    latencies = []
    for seq_len in seq_data_dict:
        data_dict = seq_data_dict[seq_len]
        for batch_size in data_dict:
            for query_len in data_dict[batch_size]:
                seq_lens.append(seq_len)
                batch_sizes.append(batch_size)
                query_lens.append(query_len)
                latencies.append(data_dict[batch_size][query_len])

    X = np.column_stack((seq_lens, batch_sizes, query_lens))
    y = np.array(latencies)
    
    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Try different models and transformations
    models = {
        'Linear': LinearRegression(),
        'Polynomial': Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ]),
        'Ridge': Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=1.0))
        ]),
        'Log-Linear': LinearRegression()  # Will use log-transformed data
    }
    
    # Try log transformation for latency
    y_train_log = np.log(y_train)
    y_test_log = np.log(y_test)
    
    best_score = 0
    best_model = None
    best_model_name = None

    for name, model in models.items():
        if name == 'Log-Linear':
            # Fit on log-transformed data
            model.fit(X_train, y_train_log)
            score = model.score(X_test, y_test_log)
        else:
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
        
        print(f"{name} R² score: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model = model
            best_model_name = name
    # Feature importance for the best model
    if best_model_name == 'Linear':
        coefficients = best_model.coef_
        feature_names = ['seq_len', 'batch_size', 'query_len']
        print("\nFeature importance:")
        for name, coef in zip(feature_names, coefficients):
            print(f"{name}: {coef:.4f}")
    return best_model, best_model_name
    
    
def print_summary(data):
    # delete 0 entry
    for seq_len in list(data['draft_times_map'].keys()):
        for batch_size in list(data['draft_times_map'][seq_len].keys()):
            for k in list(data['draft_times_map'][seq_len][batch_size].keys()):
                if data['draft_times_map'][seq_len][batch_size][k] == 0 and data['target_times_map'][seq_len][batch_size][k] == 0:
                    del data['draft_times_map'][seq_len][batch_size][k]
                    del data['target_times_map'][seq_len][batch_size][k]
                    del data['target_overhead_map'][seq_len][batch_size][k]
    
    # count valid entries, get max batch_size, max seq_len
    count = 0
    max_batch_size = 0
    max_seq_len = 0
    for seq_len in data['draft_times_map']:
        max_seq_len = max(max_seq_len, seq_len)
        for batch_size in data['draft_times_map'][seq_len]:
            max_batch_size = max(max_batch_size, batch_size)
            for k in data['draft_times_map'][seq_len][batch_size]:
                count += 1
    
    # print summary
    print(f"Number of valid entries: {count}")
    print(f"Max batch size: {max_batch_size}")
    print(f"Max seq len: {max_seq_len}")
                    
# Additional helper function for predictions
def predict_latency(model, model_name, X):
    if model_name == 'Log-Linear':
        # Convert back from log space
        return np.exp(model.predict(X))
    return model.predict(X)

def get_goodput(batch_size, k, acc_rate, data):
    draft_time = data['draft_times_map'][seq_len][batch_size][k]
    target_time = data['target_times_map'][seq_len][batch_size][k]
    overhead = data['target_overhead_map'][seq_len][batch_size][k]
    batch_latency = draft_time + target_time + overhead
    acc_len = (1 - acc_rate**(k+1)) / (1 - acc_rate)
    print(k, batch_size, acc_len, batch_latency)
    return acc_len * batch_size / batch_latency

def plot_goodput_vs_k(seq_len, batch_size, ks, data, acc_rate=0.5):
    goodput = []
    for k in ks:
        goodput.append(get_goodput(batch_size, k, acc_rate, data))
    plt.plot(ks, goodput, label='goodput', marker='o')
    plt.xlabel('k')
    plt.ylabel('Goodput')
    plt.title(f"seq_len={seq_len}_batch_size={batch_size}")
    plt.legend()
    plt.savefig(f"seq_len={seq_len}_batch_size={batch_size}_goodput_vs_k.png")
    plt.close()

def plot_latency_vs_bsz(k, seq_len, batch_size, data):
    draft_latency = []
    target_latency = []
    overhead = []
    for batch_size in batch_sizes:
        draft_latency.append(data['draft_times_map'][seq_len][batch_size][k])
        target_latency.append(data['target_times_map'][seq_len][batch_size][k])
        overhead.append(data['target_overhead_map'][seq_len][batch_size][k])
        
    plt.plot(batch_sizes, draft_latency, label='draft', marker='o')
    plt.plot(batch_sizes, target_latency, label='target', marker='o')
    plt.plot(batch_sizes, overhead, label='overhead', marker='o')
    plt.xlabel('Number of requests')
    plt.ylabel('Latency')
    plt.title(f"seq_len={seq_len}_k={k}")
    plt.legend()
    plt.savefig(f"seq_len={seq_len}_k={k}_latency_vs_batch_size.png")
    plt.close()

def plot_latency_vs_k(seq_len, batch_size, ks, data):
    draft_latency = []
    target_latency = []
    overhead = []
    for k in ks:
        draft_latency.append(data['draft_times_map'][seq_len][batch_size][k])
        target_latency.append(data['target_times_map'][seq_len][batch_size][k])
        overhead.append(data['target_overhead_map'][seq_len][batch_size][k])
    plt.plot(ks, draft_latency, label='draft', marker='o')
    plt.plot(ks, target_latency, label='target', marker='o')
    plt.plot(ks, overhead, label='overhead', marker='o')
    plt.xlabel('k')
    plt.ylabel('Latency')
    plt.title(f"seq_len={seq_len}_batch_size={batch_size}")
    plt.legend()
    plt.savefig(f"seq_len={seq_len}_batch_size={batch_size}_latency_vs_k.png")
    plt.close()

def plot_latency_vs_seq_lens(k, batch_size, seq_lens, data):
    draft_latency = []
    target_latency = []
    overhead = []
    for seq_len in seq_lens:
        draft_latency.append(data['draft_times_map'][seq_len][batch_size][k])
        target_latency.append(data['target_times_map'][seq_len][batch_size][k])
        overhead.append(data['target_overhead_map'][seq_len][batch_size][k])
    plt.plot(seq_lens, draft_latency, label='draft', marker='o')
    plt.plot(seq_lens, target_latency, label='target', marker='o')
    plt.plot(seq_lens, overhead, label='overhead', marker='o')
    plt.xlabel('Sequence length')
    plt.ylabel('Latency')
    plt.title(f"seq_len={seq_len}_batch_size={batch_size}")
    plt.legend()
    plt.savefig(f"seq_len={seq_len}_batch_size={batch_size}_latency_vs_seq_lens.png")
    plt.close() 
    
if __name__ == '__main__':

    filename = 'profile_data.pkl'

    with open(filename, 'rb') as f:
        data = pk.load(f)

    print_summary(data)

    seq_lens = [256]
    k = 1
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 80, 96, 112, 128]

    for seq_len in seq_lens:
        plot_latency_vs_bsz(k, seq_len, batch_sizes, data)

    seq_lens = [1]
    ks = [1, 2, 3, 4, 5, 6]
    batch_size = 64
    for seq_len in seq_lens:
        plot_latency_vs_k(seq_len, batch_size, ks, data)

    ks = [1]
    batch_size = 16
    seq_lens = [1, 256, 512, 1024, 1280, 1536, 1792, 2048]
    for k in ks:
        plot_latency_vs_seq_lens(k, batch_size, seq_lens, data)
    
    
    plot_goodput_vs_k(1, 64, [1, 2, 3, 4, 5, 6], data)
    
    
    model, model_name = _fit_2d_latency_models(data['target_times_map'])

    print(data['target_times_map'][256][32][1])
    print(model.predict(np.array([[256, 32, 1]])))
    print(model.predict(np.array([[256, 64, 1]])))