# Sample from a dataset and collect token acceptance rate.
from typing import List

import numpy as np
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.engine.metrics_types import SpecDecodeWorkerMetrics

from ..benchmark_serving import sample_sharegpt_requests


def print_metric(spec_dec_metric: SpecDecodeWorkerMetrics):
    token_acc_rate = spec_dec_metric.draft_acceptance_rate
    print(f"Token acceptance rate: {token_acc_rate:.2f}")
    system_efficiency = spec_dec_metric.system_efficiency
    print(f"System efficiency: {system_efficiency:.2f}")
    draft_tokens = spec_dec_metric.draft_tokens
    print(f"Draft token: {draft_tokens}")
    emitted_tokens = spec_dec_metric.emitted_tokens
    print(f"Emitted tokens: {emitted_tokens}")
    accepted_tokens = spec_dec_metric.accepted_tokens
    print(f"Accepted tokens: {accepted_tokens}")
    num_spec_tokens = spec_dec_metric.num_spec_tokens
    print(f"Number of speculative tokens: {num_spec_tokens}")
    meidan_proposal_times = np.median(spec_dec_metric.poposal_times)
    print(f"Median proposal time: {meidan_proposal_times:.2f}")
    meidan_scoring_times = np.median(
        [x[-1] for x in spec_dec_metric.scoring_times])
    print(f"Median scoring time: {meidan_scoring_times:.2f}")
    print(spec_dec_metric.scoring_times)
    meidan_verification_times = np.median(spec_dec_metric.verification_times)
    print(f"Median verification time: {meidan_verification_times:.2f}")


if __name__ == "__main__":
    dataset_path = "/data/lily/ShareGPT_V3_unfiltered_cleaned_split.json"
    model = "lmsys/vicuna-7b-v1.5"
    # speculative_model = "JackFram/llama-68m"
    speculative_model = "eqhylxx/vicuna-160m"
    num_requests = 1000
    tokenizer = AutoTokenizer.from_pretrained(model)
    fixed_output_len = 100
    temperature = 0.0
    ignore_eos = True
    num_speculative_tokens = 5
    tensor_parallel_size = 1

    dataset = "uniform"

    # Sample dataset
    if dataset == "sharegpt":
        filtered_dataset = sample_sharegpt_requests(dataset_path,
                                                    num_requests,
                                                    tokenizer,
                                                    fixed_output_len=None)
        input_lens = [
            prompt_len
            for (prompt, prompt_len, output_len, _) in filtered_dataset
        ]
        output_lens = [
            output_len
            for (prompt, prompt_len, output_len, _) in filtered_dataset
        ]
        print(f"Average input length: {sum(input_lens) / len(input_lens)}")
        print(f"Average output length: {sum(output_lens) / len(output_lens)}")

        # Prepare prompts and sampling params
        prompts: List[str] = []
        sampling_params: List[SamplingParams] = []
        for (prompt, prompt_len, output_len, _) in filtered_dataset:
            prompts.append(prompt)
            sampling_params.append(
                SamplingParams(
                    n=1,
                    temperature=temperature,
                    ignore_eos=ignore_eos,
                    max_tokens=output_len,
                ))
    elif dataset == "uniform":
        input_len = 550
        output_len = 100
        prompt_ids = [[1] * input_len] * num_requests
        prompts = tokenizer.batch_decode(prompt_ids)
        sampling_params = [
            SamplingParams(
                n=1,
                temperature=temperature,
                ignore_eos=True,
                max_tokens=output_len,
            )
        ] * num_requests

    # Initialize engine
    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        seed=0,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=False,
        max_num_batched_tokens=40960,
        speculative_model=speculative_model,
        num_speculative_tokens=num_speculative_tokens,
        disable_log_stats=False,
        disable_async_output_proc=True,
        enforce_eager=False,
        # spec_decoding_acceptance_method="typical_acceptance_sampler"
    )

    llm.generate(prompts, sampling_params, use_tqdm=True)
    spec_dec_metric: SpecDecodeWorkerMetrics = \
        llm.llm_engine.spec_decode_metrics
    print_metric(spec_dec_metric)
