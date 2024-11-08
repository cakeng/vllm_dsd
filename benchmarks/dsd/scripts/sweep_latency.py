import os
import re
import time
from itertools import product

cuda_devices = [4, 5, 6, 7]

model_list = ['meta-llama/Llama-3.1-70B-Instruct']
tp_list = [4]
# model_list = ['meta-llama/Llama-2-7b-hf']
# tp_list = [1]

batch_size_list = [1]
eager_list = [False]
input_len_list = [1024, 2048, 4096, 8192, 16384]
output_len_list = [256]
acceptance_rate_list = [0.7, 0.8]
dsd_list = [True]
num_iters_warmup = 3
num_iters = 10

# Baseline
num_spec_tokens_list = [None]
batch_size_list = [1, 2]
spec_model_list = [None]
acceptance_rate_list = [None]
tp_list = [4]
dsd_list = [False]

assert len(model_list) == len(tp_list) and len(model_list) == len(spec_model_list),\
    "model_list, tp_list, and spec_model_list must have the same length"

start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
output_time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
csv_output = f'sweep_latency_{output_time_str}.csv'
log_output = f'sweep_latency_{output_time_str}.log'
tmp_output = f'sweep_latency_{output_time_str}.tmp'

with open(log_output, 'w') as f:
    out_str = f"{start_time_str} Running sweep with the following parameters:\n"
    out_str += f"\tmodel_list: {model_list}\n"
    out_str += f"\ttp_list: {tp_list}\n"
    out_str += f"\tspec_model_list: {spec_model_list}\n"
    out_str += f"\teager_list: {eager_list}\n"
    out_str += f"\tnum_spec_tokens_list: {num_spec_tokens_list}\n"
    out_str += f"\tbatch_size_list: {batch_size_list}\n"
    out_str += f"\tacceptance_rate_list: {acceptance_rate_list}\n"
    out_str += f"\tinput_len_list: {input_len_list}\n"
    out_str += f"\toutput_len_list: {output_len_list}\n"
    out_str += f"\tnum_iters_warmup: {num_iters_warmup}\n"
    out_str += f"\tnum_iters: {num_iters}\n"
    out_str += "\n"
    print(out_str, end='')
    f.write(out_str)

with open(csv_output, 'w') as f:
    f.write(
        "model,speculative_model,eager,num-speculative-tokens,batch_size,"
    )
    f.write("acceptance_rate,input_len, output_len,")
    f.write(
        "draft_acceptance_rate,system_efficiency,draft_tokens,emitted_tokens,accepted_tokens,"
    )
    f.write("avg latency, 10%, 25%, 50%, 90%, 99%,")
    f.write("Raw Dump\n")

for model, tp, spec_model in zip(model_list, tp_list, spec_model_list):
    for eager, num_spec_tokens, batch_size, acceptance_rate, input_len, output_len, dsd in \
        product(eager_list, num_spec_tokens_list, batch_size_list, acceptance_rate_list,
                input_len_list, output_len_list, dsd_list):
        cmd = f"CUDA_VISIBLE_DEVICES={','.join(map(str, cuda_devices))} "
        cmd += "python benchmarks/benchmark_latency.py"
        cmd += f" --model {model}"
        cmd += f" --tensor_parallel_size {tp}"
        cmd += f" --batch_size {batch_size}"
        cmd += f" --input_len {input_len}"
        cmd += f" --output_len {output_len}"
        cmd += f" --num_iters_warmup {num_iters_warmup}"
        cmd += f" --num_iters {num_iters}"
        if eager:
            cmd += " --enforce_eager"
        if spec_model:
            cmd += f" --speculative_model {spec_model}"
            cmd += f" --speculative-draft-tensor-parallel-size 1"
            cmd += f" --acceptance_rate {acceptance_rate}"
            if num_spec_tokens:
                cmd += f" --num-speculative-tokens {num_spec_tokens}"
            if dsd:
                cmd += " --dsd"
        
        cur_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(log_output, 'a') as f:
            print(f"{cur_time_str} Running: {cmd}")
            f.write(f"{cur_time_str} Running: {cmd}\n")
        with open(csv_output, 'a') as f:
            f.write(
                f"{model},{spec_model},{eager},{num_spec_tokens},{batch_size},"
            )
            f.write(f"{acceptance_rate},{input_len},{output_len},")

        # Run 
        os.system(f"{cmd} > {tmp_output}")
        
        with open(tmp_output, 'r') as f:
            output = f.read()
            with open(log_output, 'a') as f_log:
                f_log.write(output)
            # Find "Profiling iterations" in the output and slice the rest of the output
            # idx = output.find("Profiling iterations:")
            idx = 0
            lines = output[idx:].split('\n')
            draft_acceptance_rate_list = []
            system_efficiency_list = []
            draft_tokens_list = []
            emitted_tokens_list = []
            accepted_tokens_list = []
            avg_latency = -1
            percentile_10_latency = -1
            percentile_25_latency = -1
            percentile_50_latency = -1
            percentile_90_latency = -1
            percentile_99_latency = -1
            for line in lines:
                match = re.search(r'draft_acceptance_rate=([\d\.]+)', line)
                if match:
                    draft_acceptance_rate_list.append(float(match.group(1)))
                match = re.search(r'system_efficiency=([\d\.]+)', line)
                if match:
                    system_efficiency_list.append(float(match.group(1)))
                match = re.search(r'draft_tokens=([\d]+)', line)
                if match:
                    draft_tokens_list.append(int(match.group(1)))
                match = re.search(r'emitted_tokens=([\d]+)', line)
                if match:
                    emitted_tokens_list.append(int(match.group(1)))
                match = re.search(r'accepted_tokens=([\d]+)', line)
                if match:
                    accepted_tokens_list.append(int(match.group(1)))
                match = re.search(r'Avg latency: ([\d\.]+) seconds', line)
                if match:
                    avg_latency = float(match.group(1))
                match = re.search(r'10% percentile latency: ([\d\.]+) seconds',
                                  line)
                if match:
                    percentile_10_latency = float(match.group(1))
                match = re.search(r'25% percentile latency: ([\d\.]+) seconds',
                                  line)
                if match:
                    percentile_25_latency = float(match.group(1))
                match = re.search(r'50% percentile latency: ([\d\.]+) seconds',
                                  line)
                if match:
                    percentile_50_latency = float(match.group(1))
                match = re.search(r'90% percentile latency: ([\d\.]+) seconds',
                                  line)
                if match:
                    percentile_90_latency = float(match.group(1))
                match = re.search(r'99% percentile latency: ([\d\.]+) seconds',
                                  line)
                if match:
                    percentile_99_latency = float(match.group(1))
            with open(log_output, 'a') as f:
                f.write(
                    f"Draft acceptance rate: {draft_acceptance_rate_list}\n")
                f.write(f"System efficiency: {system_efficiency_list}\n")
                f.write(f"Draft tokens: {draft_tokens_list}\n")
                f.write(f"Emitted tokens: {emitted_tokens_list}\n")
                f.write(f"Accepted tokens: {accepted_tokens_list}\n")
                f.write(f"Avg latency: {avg_latency} seconds\n")
                f.write(
                    f"10% percentile latency: {percentile_10_latency} seconds\n"
                )
                f.write(
                    f"25% percentile latency: {percentile_25_latency} seconds\n"
                )
                f.write(
                    f"50% percentile latency: {percentile_50_latency} seconds\n"
                )
                f.write(
                    f"90% percentile latency: {percentile_90_latency} seconds\n"
                )
                f.write(
                    f"99% percentile latency: {percentile_99_latency} seconds\n"
                )
                f.write("\n")
            fin_draft_acceptance_rate = draft_acceptance_rate_list[-1] if \
                len(draft_acceptance_rate_list) > 0 else -1
            fin_system_efficiency = system_efficiency_list[-1] if \
                len(system_efficiency_list) > 0 else -1
            fin_draft_tokens =  draft_tokens_list[-1] if \
                len(draft_tokens_list) > 0 else -1
            fin_emitted_tokens = emitted_tokens_list[-1] if \
                len(emitted_tokens_list) > 0 else -1
            fin_accepted_tokens = accepted_tokens_list[-1] if \
                len(accepted_tokens_list) > 0 else -1
            with open(csv_output, 'a') as f:
                f.write(
                    f"{fin_draft_acceptance_rate},{fin_system_efficiency},"
                    f"{fin_draft_tokens},{fin_emitted_tokens},{fin_accepted_tokens},"
                    f"{avg_latency},{percentile_10_latency},{percentile_25_latency},"
                    f"{percentile_50_latency},{percentile_90_latency},{percentile_99_latency},"
                )
                f.write("\n")

os.remove(tmp_output)
end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
with open(log_output, 'a') as f:
    f.write(f"{end_time_str} Sweep completed.\n")
