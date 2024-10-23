import json
import os
from matplotlib import pyplot as plt

ttft_results = {}
tpot_results = {}
model = "llama2-7b"
result_dir = f"benchmarks/dsd/results/{model}/"
for file in os.listdir(result_dir):
    if file.endswith(".json"):
        with open(result_dir + file, "r") as f:
            data = json.load(f)
            file = file.split(".")[0]
            batch_size = int(file.split("_")[0].split("=")[1])
            input_len = int(file.split("_")[1].split("=")[1])
            if batch_size not in ttft_results:
                ttft_results[batch_size] = {}
                tpot_results[batch_size] = {}

            ttft_results[batch_size][input_len] = sum(data["ttfts"]) / len(
                data["ttfts"])
            tpot_results[batch_size][input_len] = sum(data["tpots"]) / len(
                data["tpots"])

all_batch_sizes = sorted(list(ttft_results.keys()))
print(all_batch_sizes)
all_context_lengths = sorted(list(ttft_results[all_batch_sizes[0]].keys()))
all_context_lengths = [x for x in all_context_lengths if x <= 512]
print(all_context_lengths)

# Plot batch latency VS bacth size for different context lengths
for context_length in all_context_lengths:
    ttft_latencies = [
        ttft_results[batch_size][context_length]
        for batch_size in all_batch_sizes
        if context_length in ttft_results[batch_size]
    ]
    tpot_latencies = [
        tpot_results[batch_size][context_length]
        for batch_size in all_batch_sizes
        if context_length in tpot_results[batch_size]
    ]
    # plt.plot(all_batch_sizes[:len(ttft_latencies)], ttft_latencies, label=f"TTFT, context length={context_length}", marker="o", markersize=5)
    plt.plot(all_batch_sizes[:len(tpot_latencies)],
             tpot_latencies,
             label=f"TPOT, context length={context_length}",
             marker="o",
             markersize=5)
plt.ylim(bottom=0)
plt.xlim(left=0)
plt.xlabel("Batch size")
plt.ylabel("Batch Latency (ms)")
plt.legend()
plt.savefig(f"benchmarks/dsd/figures/{model}_batch_size.png")
plt.close()

# Plot batch latency VS context length for different batch sizes
for batch_size in all_batch_sizes:
    ttft_latencies = [
        ttft_results[batch_size][context_length]
        for context_length in all_context_lengths
        if context_length in ttft_results[batch_size]
    ]
    tpot_latencies = [
        tpot_results[batch_size][context_length]
        for context_length in all_context_lengths
        if context_length in tpot_results[batch_size]
    ]
    # plt.plot(all_context_lengths, ttft_latencies, label=f"TTFT, batch size={batch_size}", marker="o", markersize=5)
    plt.plot(all_context_lengths[:len(tpot_latencies)],
             tpot_latencies,
             label=f"TPOT, batch size={batch_size}",
             marker="o",
             markersize=5)
plt.ylim(bottom=0)
plt.xlabel("Context length")
plt.ylabel("Batch Latency (ms)")
plt.legend()
plt.savefig(f"benchmarks/dsd/figures/{model}_context_length.png")
