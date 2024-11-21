import json
import matplotlib.pyplot as plt

tracedir = "benchmarks/dsd/trace/"

def load(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data['traces']

def load_all(acc, all_batch_sizes):
    data = {}
    for batch_size in all_batch_sizes:
        data[batch_size] = load(tracedir + f"{batch_size}_{acc}_True.json")
    return data

def get_avg_proposed_len(data):
    proposed_lens = []
    for trace in data:
        if trace['type'] == 'Request':
            continue
        if 'proposed_len' not in trace:
            continue
        proposed_lens.append(trace['proposed_len'])
    avg_proposed_len = sum(proposed_lens) / (len(proposed_lens) + 1e-5)
    return avg_proposed_len

if __name__ == "__main__":
    all_batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    acc_rates = [0.5, 0.7, 0.9]

    plt.figure(figsize=(3, 3))
    for acc in acc_rates:
        data = load_all(acc, all_batch_sizes)
        proposed_lens = []
        for batch_size in all_batch_sizes:
            proposed_lens.append(get_avg_proposed_len(data[batch_size]))
        plt.plot(all_batch_sizes, proposed_lens, marker='o', label=f"acc={acc}")    
    
    plt.xscale("log", base=2)
    plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
    plt.gca().xaxis.set_major_locator(plt.FixedLocator(all_batch_sizes))
    plt.xlabel("Batch size")
    plt.ylabel("Avg proposed length")
    plt.tight_layout()
    plt.savefig("benchmarks/dsd/figures/proposed_len.png")