from dataclasses import dataclass
import subprocess
from subprocess import CompletedProcess
from typing import List
import torch
import os
import time
import signal
import csv
import argparse
import json
import numpy as np
from pathlib import Path

# from analyse import analyse_results

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
PROFILING_DATA_DIR = PROJECT_ROOT_DIR / "data"
INFO_STR_PATH = "info_str.tmp"

def parse_tuple(string):
    """
    Parse a string into a tuple of values.
    Accepts formats like "1,2,3" or "(1,2,3)" or "1, 2, 3"
    """
    # Remove parentheses if present
    string = string.strip('()')
    # Split by comma and convert to appropriate type
    try:
        # Try converting to integers first
        return tuple(int(x.strip()) for x in string.split(','))
    except ValueError:
        # If integer conversion fails, keep as strings
        return tuple(x.strip() for x in string.split(','))


# TODO: for automation, load these from a json file
@dataclass
class BenchSetting:
    model: str
    tp: int
    device: str
    dataset: str
    req_rate: float
    interval_file: str
    num_requests: int = 200
    port: int = 10000
    speculative_model: str = None
    num_speculative_tokens: int = -1
    speculative_draft_tensor_parallel_size: int = -1
    dsd: bool = False
    acceptance_rate: float = None

    @staticmethod
    def get_head():
        return [
            "model", "tp", "device", "dataset", "req_rate", 
            "interval_file", "num_requests",
            "speculative_model", "num_speculative_tokens",
            "speculative_draft_tensor_parallel_size", "dsd"
        ]

    def get_value_list(self):
        return [
            self.model, self.tp, self.device, self.dataset, self.req_rate,
            self.interval_file, self.num_requests, self.speculative_model,
            self.num_speculative_tokens,
            self.speculative_draft_tensor_parallel_size, self.dsd
        ]


@dataclass
class BenchResult:
    setting: BenchSetting
    mean_TTFT: float
    mean_TPOT: float
    p99_TTFT: float
    p99_TPOT: float
    mean_e2el_latency: float
    p99_e2el_latency: float
    total_time: float
    throughput: float

    @staticmethod
    def get_head():
        return BenchSetting.get_head() + [
            "mean_TTFT",
            "mean_TPOT",
            "p99_TTFT",
            "p99_TPOT",
            "mean_e2el_latency",
            "p99_e2el_latency",
            "total_time",
            "throughput",
        ]

    def get_value_list(self):
        return self.setting.get_value_list() + [
            self.mean_TTFT,
            self.mean_TPOT,
            self.p99_TTFT,
            self.p99_TPOT,
            self.mean_e2el_latency,
            self.p99_e2el_latency,
            self.total_time,
            self.throughput,
        ]


class Util:

    @staticmethod
    def run_cmd(cmd, blocking=True, print_cmd=True):

        def set_new_pgroup():
            os.setpgrp()
        if print_cmd:
            print(cmd)
        if blocking:
            return subprocess.run(cmd,
                                  shell=True,
                                  capture_output=True,
                                  text=True)
        else:
            return subprocess.Popen(cmd, shell=True, preexec_fn=set_new_pgroup)

class BenchEngine:

    def __init__(self, setting: BenchSetting) -> None:
        self.backend_process: subprocess.Popen = None
        # launch backend
        cmd = (
            f"vllm serve {setting.model} "
            f" --tensor-parallel-size {setting.tp} "
            f" --disable-log-requests"
            # f" --max-model-len 40960"
            f" --port {setting.port}"
            f" --enable-chunked-prefill=False"
            f" --disable-async-output-proc")
        if setting.speculative_model:
            cmd = "VLLM_USE_FLASHINFER_SAMPLER=1 " + cmd
            cmd += f" --speculative-model {setting.speculative_model}"
            cmd += " --use-v2-block-manager"
        else:
            pass
            # cmd += " --enforce-eager"
        if setting.num_speculative_tokens >= 0:
            cmd += f" --num-speculative-tokens {setting.num_speculative_tokens}"
        if setting.speculative_draft_tensor_parallel_size > 0:
            cmd += " --speculative-draft-tensor-parallel-size " + \
                  f"{setting.speculative_draft_tensor_parallel_size}"
        if setting.speculative_model == "[ngram]":
            cmd += f" --ngram-prompt-lookup-max {args.ngram_prompt_lookup_max}"
            cmd += f" --ngram-prompt-lookup-min {args.ngram_prompt_lookup_min}"
        if setting.dsd:
            cmd += " --dsd"
        if setting.acceptance_rate:
            cmd += f" --acceptance-rate {setting.acceptance_rate}"

        print (f"Launching backend...")
        self.backend_process = Util.run_cmd(cmd, False)
        self.backend_cmd = cmd

    def check_server_up(self, port: int):
        # check if server is up
        time.sleep(10)
        cmd = f"curl http://localhost:{port}/health"
        while Util.run_cmd(cmd, True, False).returncode != 0:
            print("Server is not up yet, waiting for 10 seconds...")
            time.sleep(10)

    def bench(self, runs: List[BenchSetting]) -> List[BenchResult]:
        self.check_server_up(runs[0].port)
        print("============Start Benchmarking==================")
        return [self.bench_single(run) for run in runs]

    def bench_single(self, run: BenchSetting) -> BenchResult:
        out_values = [str(x) for x in run.get_value_list()]
        out_values = [x.replace("/", "_") for x in out_values]
        result_filename = f"bench_results/{':'.join(out_values)}.json"
        
        
        if run.interval_file is not None:
            interval_arr = np.loadtxt(run.interval_file)
            run.num_requests = len(interval_arr)
            print(f"Interval array: {interval_arr}")
        else:
            # We always run the server for two minutes
            run.num_requests = int(run.req_rate * 120)
        
        
        cmd = f"python benchmarks/benchmark_serving.py "
        cmd += f" --model {run.model} --request-rate {run.req_rate}"
        cmd += f" --num-prompts {run.num_requests}"
        cmd += f" --save-result "
        cmd += f" --result-filename {result_filename}"
        cmd += f" --port {run.port}"
        if run.interval_file is not None:
            cmd += f" --interval_file {run.interval_file}"
        if run.dataset in ["cnn_dailymail"]:
            cmd += f" --dataset-name {run.dataset}"
        elif run.dataset == "sonnet":
            cmd += f" --dataset-name {run.dataset}"
            cmd += f" --dataset-path {run.dataset_path}"
        elif run.dataset == "sharegpt_sonnet":
            cmd += f" --dataset-name {run.dataset}"
        else:
            cmd += f" --dataset {run.dataset}"
        
        with open(INFO_STR_PATH, "w") as f:
            f.write(f"Backend command: {self.backend_cmd}\n")
            f.write(f"benchmark_serving.py command: {cmd}\n")
        os.environ["INFO_STR_PATH"] = INFO_STR_PATH

        print(f"Running benchmark...")
        
        completed_process: CompletedProcess = Util.run_cmd(cmd, True)
        
        print("Waiting for 20 seconds for the server to dump the metrics...")
        time.sleep(20)
        os.remove(INFO_STR_PATH) 

        def process_output(completed_process: CompletedProcess):
            if completed_process.returncode != 0:
                print(f"[Error] {completed_process.stdout}")
                print(f"[Error] {completed_process.stderr}")
                return BenchResult(run, -1, -1, -1, -1, -1, -1, -1, -1)
            with open(result_filename, "r") as f:
                result = json.load(f)
                e2el_latencies = result["e2els"]
                e2el_latencies.sort()
                mean_e2el_latency = sum(e2el_latencies) / len(e2el_latencies)
                p99_e2el_latency = e2el_latencies[int(
                    len(e2el_latencies) * 0.99)]
                return BenchResult(
                    run,
                    result["mean_ttft_ms"],
                    result["mean_tpot_ms"],
                    result["p99_ttft_ms"],
                    result["p99_tpot_ms"],
                    mean_e2el_latency,
                    p99_e2el_latency,
                    result["duration"],
                    result["output_throughput"],
                )

        return process_output(completed_process)

    def dump_results(self, results: List[BenchResult], outfile: str) -> None:
        with open(outfile, "a") as f:
            writer = csv.writer(f)
            writer.writerow(BenchResult.get_head())
            for result in results:
                writer.writerow(result.get_value_list())

    def __del__(self):
        # stop backend
        print("==============Finish Benchmarking==============")
        if (self.backend_process.poll() is
                None):  # If poll() returns None, the process is still running
            print("Process is running, trying to kill...")
            os.killpg(self.backend_process.pid, signal.SIGINT)
            time.sleep(10)  # wait a bit for cleaning resources
            self.backend_process.terminate()
            self.backend_process.wait()
            time.sleep(1)
            if self.backend_process.poll() is not None:
                print(
                    f"Process {self.backend_process.pid} killed successfully.")
            else:
                print("Failed to kill process.")
        else:
            print("Process already terminated.")


def main(args):
    device = torch.cuda.get_device_name(0).replace(" ", "_")
    runs = []
    request_rates = []
    # All * 10 to generate non-integer request rates
    bench_setting = None
    tp = 4 if "70" in args.model else 1
    if args.speculative_model and "7B" not in args.speculative_model:
        speculative_draft_tensor_parallel_size = 1
    else:
        speculative_draft_tensor_parallel_size = -1
    runs = []
        
    if args.interval_file is None:
        request_rate_start, request_rate_end, step = parse_tuple(
                                                        args.request_rate_params)
        # All * 10 to generate non-integer request rates
        print (f"Generating request rates from {request_rate_start} to "
               f"{request_rate_end} with step {step}")
        for req_rate in range(request_rate_start * 10,
                            (request_rate_end + step) * 10, step * 10):
            req_rate = req_rate / 10.0
            bench_setting = BenchSetting(
                args.model,
                tp,
                device,
                args.dataset,
                req_rate,
                None,
                -1,
                args.port,  # port
                args.speculative_model,
                args.num_speculative_tokens,
                speculative_draft_tensor_parallel_size,
                args.dsd,
                args.acceptance_rate)
            runs.append(bench_setting)
    else:
        # Check if interval_file is actually a file
        interval_files = []
        if os.path.isfile(args.interval_file):
            interval_files.append(args.interval_file)
        else:
            for file in os.listdir(args.interval_file):
                interval_files.append(file)
        print (f"Found {len(interval_files)} interval files.")
        print (f"Files: {interval_files}")
        for file in interval_files:
            bench_setting = BenchSetting(
                args.model,
                tp,
                device,
                args.dataset,
                -1,
                file,
                -1,
                args.port,  # port
                args.speculative_model,
                args.num_speculative_tokens,
                speculative_draft_tensor_parallel_size,
                args.dsd)
            runs.append(bench_setting)
            
    with open(INFO_STR_PATH, "w") as f:
        f.write(f"Benchmark Script Initializing...\n")
    
    engine = BenchEngine(bench_setting)
    results = engine.bench(runs)
    engine.dump_results(results, f"bench_results/{args.result_file}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        type=str,
                        default="meta-llama/Meta-Llama-3.1-70B-Instruct")
    parser.add_argument("--speculative-model", type=str, default=None)
    parser.add_argument("--ngram-prompt-lookup-max", type=int, default=-1)
    parser.add_argument("--ngram-prompt-lookup-min", type=int, default=-1)
    parser.add_argument("--num-speculative-tokens", type=int, default=-1)
    parser.add_argument("--dsd", action="store_true")
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--result-file", type=str, default="all_results")
    parser.add_argument(
        "--dataset",
        type=str,
        default="/data/lily/ShareGPT_V3_unfiltered_cleaned_split.json",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/work/js_park/sonnet.txt",
    )
    parser.add_argument("--outfile", type=str, default="bench_results")
    parser.add_argument(
        "--request_rate_params",
        type=str,
        help="(start_request_rate,end_request_rate,step_size)." +
        "End_request_size is INCLUDED.",
        default="(2,12,2)",
    )
    parser.add_argument("--interval-file", type=str,
                        default=None)
    parser.add_argument("--acceptance-rate", type=float, default=None)
    args = parser.parse_args()

    main(args)
