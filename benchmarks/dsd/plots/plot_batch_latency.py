from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class LatencyResult:
    batch_size: int
    acceptance_rate: float
    input_len: int
    output_len: int
    num_spec_tokens: int

    avg_latency: float
    p90_latency: float
    p99_latency: float


def load_baseline_results(file_path):
    results = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.split(',')
            batch_size = int(parts[2])
            input_len = int(parts[3])
            output_len = int(parts[4])
            avg_latency = float(parts[5])
            p90_latency = float(parts[9])
            p99_latency = float(parts[10])
            results.append(
                LatencyResult(batch_size, -1, input_len, output_len, -1,
                              avg_latency, p90_latency, p99_latency))
    return results


def load_results(file_path):
    results = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.split(',')
            model = parts[0]
            spec_model = parts[1]
            eager = parts[2]
            num_spec_tokens = int(parts[3])
            batch_size = int(parts[4])
            acceptance_rate = float(parts[5])
            input_len = int(parts[6])
            output_len = int(parts[7])
            avg_latency = float(parts[13])
            p90_latency = float(parts[17])
            p99_latency = float(parts[18])
            results.append(
                LatencyResult(batch_size, acceptance_rate, input_len,
                              output_len, num_spec_tokens, avg_latency,
                              p90_latency, p99_latency))
    return results


def plot_by_acceptance_rate(acceptance_rate, sd_results, dsd_results,
                            baseline_results, fig_dir):
    sd_results = [
        result for result in sd_results
        if result.acceptance_rate == acceptance_rate
    ]
    dsd_results = [
        result for result in dsd_results
        if result.acceptance_rate == acceptance_rate
    ]
    # baseline_results = [result for result in baseline_results if result.acceptance_rate == acceptance_rate]

    sd_batch_sizes = set([result.batch_size for result in sd_results])
    dsd_batch_sizes = set([result.batch_size for result in dsd_results])
    batch_sizes = sorted(list(sd_batch_sizes & dsd_batch_sizes))
    num_spec_tokens = sorted(
        list(set([result.num_spec_tokens for result in sd_results])))

    for bsz in batch_sizes:
        threshold = [
            result for result in baseline_results if result.batch_size == bsz
        ][0].avg_latency
        print(f"Threshold: {threshold}")
        categories = [str(x) for x in num_spec_tokens] + ['DSD']
        values = [
            result.avg_latency
            for result in sd_results if result.batch_size == bsz
        ] + [
            result.avg_latency
            for result in dsd_results if result.batch_size == bsz
        ]
        plt.figure(figsize=(4, 4))
        plt.axhline(y=threshold,
                    color='red',
                    linestyle='--',
                    label='Threshold')
        plt.bar(categories, values)
        plt.title(f'Batch size: {bsz}, Acceptance rate: {acceptance_rate}',
                  size=12)
        plt.xlabel('Number of speculative tokens', size=12)
        plt.ylabel('Average latency (s)', size=12)
        plt.tight_layout()
        plt.savefig(f'{fig_dir}/latency_{acceptance_rate}_{bsz}.png')
        plt.close()


if __name__ == "__main__":
    # sd_result_loc = "/data/lily/vllm-dsd-osdi/benchmarks/dsd/results/7B_160M_Eager_False.csv"
    # org_result_loc = "/data/lily/vllm-dsd-osdi/benchmarks/dsd/results/baseline_7B_160M.csv"
    # dsd_result_loc = "/data/lily/vllm-dsd-osdi/benchmarks/dsd/results/dsd_7B_160M.csv"
    # sd_results = load_results(sd_result_loc)
    # dsd_results = load_results(dsd_result_loc)
    # org_results = load_baseline_results(org_result_loc)
    # fig_dir = "/data/lily/vllm-dsd-osdi/benchmarks/dsd/figures"
    # plot_by_acceptance_rate(0.6, sd_results, dsd_results, org_results, fig_dir)

    sd_result_loc = "/data/lily/vllm-dsd-osdi/benchmarks/dsd/results/70B_1B_Eager_False.csv"
    org_result_loc = "/data/lily/vllm-dsd-osdi/benchmarks/dsd/results/baseline_70B_1B.csv"
    dsd_result_loc = "/data/lily/vllm-dsd-osdi/benchmarks/dsd/results/dsd_70B_1B.csv"
    sd_results = load_results(sd_result_loc)
    dsd_results = load_results(dsd_result_loc)
    org_results = load_baseline_results(org_result_loc)
    fig_dir = "/data/lily/vllm-dsd-osdi/benchmarks/dsd/figures/70B"
    plot_by_acceptance_rate(0.8, sd_results, dsd_results, org_results, fig_dir)
