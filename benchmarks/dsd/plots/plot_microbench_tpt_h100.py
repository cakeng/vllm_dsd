import matplotlib.pyplot as plt
import numpy as np

# Prepare data
acc_rates = [0.7, 0.8, 0.9]
tokens = [1, 3, 5, 7, 9, 11]
data = {
    0.7: [1.037810384, 1.180586907, 1.212189616, 1.093115124, 0.9864559819, 1.21],
    0.8: [1.06772009, 1.06772009, 1.340857788, 1.28, 1.22, 1.34],
    0.9: [1.100451467, 1.286117381, 1.471218962, 1.33, 1.32, 1.46]
}

# Create plot
fig, ax = plt.subplots(figsize=(5, 2))

# Plot each group
for i, acc in enumerate(acc_rates):
    # Create subplot position
    left = (i + 0.1) * (len(tokens) + 1)
    x = np.arange(len(tokens))
    
    # Plot bars
    bars = ax.bar(x + left, data[acc], 
                  color=['dodgerblue' if j != len(tokens) - 1 else 'crimson' for j in range(len(tokens))])

# Customize plot
# plt.title('Llama2-7b Throughput Speedup', pad=20)
# plt.xlabel('num_spec_token')
plt.xticks([])  # Hide x ticks

# Add vertical lines and acc labels
for i in range(len(acc_rates)):
    left = (i + 0.1) * (len(tokens) + 1)
    if i > 0:
        plt.axvline(x=left - 1, color='gray', linestyle='--')
    plt.text(left + 1, 1.62, f'acc={acc_rates[i]}', size=12)
    
    # Add x-axis labels
    for j, token in enumerate(tokens):
        if token == 11:
            label = 'DSD'
        else:
            label = str(token)
        plt.text(left + j, -0.2, label, ha='center', size=10)

# Add right boundary line
# plt.axvline(x=len(acc_rates) * (len(tokens) + 1) - 1.5, color='gray', linestyle='--')

plt.ylim(0, 1.6)
plt.grid(axis='y')
plt.tight_layout()
plt.ylabel('Speedup (x)', size=12)
# plt.xlabel('num_spec_token', loc)
plt.tight_layout()
plt.savefig(f"/data/lily/vllm-dsd-osdi/benchmarks/dsd/figures/7b_tpt.pdf")