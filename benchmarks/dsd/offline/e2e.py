from vllm import LLM, SamplingParams
import time
import random
random.seed(0)
import json
from typing import List, Tuple, Optional
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from argparse import ArgumentParser

# Sample prompts.

output_len = 256
input_len = 2048 # filter input smaller than this size

# number of gpu blocks = 27977
batch_size = 256
target_model = "meta-llama/Meta-Llama-3-8B-Instruct"
draft_model = "turboderp/Qwama-0.5B-Instruct"
max_num_seqs = int(27977 * 16 / (input_len + output_len) * 0.9)
tp=1

# # number of gpu blocks = 25639
# batch_size = 512
# target_model = "meta-llama/Llama-3.1-70B-Instruct"
# draft_model = "meta-llama/Llama-3.2-1B-Instruct"
# tp = 4
# max_num_seqs = 350

# # number of gpu blocks = 6822
# This is bad because of the draft model has 
# very low token acceptance rate.
# batch_size = 128
# target_model = "lmsys/vicuna-7b-v1.5"
# draft_model = "eqhylxx/vicuna-160m"
# tp = 1
# max_num_seqs = 100

dataset = "sharegpt"
gpu_memory_utilization = 0.9
outfile = f"benchmarks/dsd/offline/{dataset}_result"
dataset_path = "/data/lily/ShareGPT_V3_unfiltered_cleaned_split.json"

def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    input_len: int,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int, None]]:
    prompt_ids = []
    # Load the dataset.
    with open(dataset_path, encoding='utf-8') as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    total_input_len = 0
    # Shuffle the dataset.
    random.shuffle(dataset)

    for i in range(len(dataset)):
        if len(prompt_ids) >= num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        total_input_len += prompt_len
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if prompt_len < input_len or (fixed_output_len is None and output_len < 4):
            # Prune too short sequences.
            continue
        prompt_ids.append(prompt_token_ids)

    print(f"==================Total input length: {total_input_len}")
    return prompt_ids

def prepare_input(dataset, input_len, output_len, batch_size):
    tokenizer = AutoTokenizer.from_pretrained(target_model)
    if dataset == "sharegpt":
        prompt_ids = sample_sharegpt_requests(
            dataset_path=dataset_path,
            num_requests=batch_size,
            tokenizer=tokenizer,
            input_len=input_len,
            fixed_output_len=None
        )
        sampling_params = SamplingParams(temperature=0,
                                 max_tokens=output_len,
                                 ignore_eos=True)
    else:
        raise ValueError(f"Invalid dataset: {dataset}")
    
    return prompt_ids, sampling_params

def get_output_len(outputs):
    output_len = 0
    for output in outputs:
        for co in output.outputs:
            output_len += len(co.token_ids)
    return output_len
    
def profile(llm, prompt_ids, sampling_params):
    start = time.time()
    outputs = llm.generate(prompt_token_ids=prompt_ids,
                        sampling_params=sampling_params)
    end = time.time()
    total_time = end - start
    total_len = get_output_len(outputs)
    print("=====Total generation length: ", total_len)
    tpt = total_len / total_time
    return total_time, tpt


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--org", action="store_true")
    parser.add_argument("--dsd", action="store_true")
    parser.add_argument("--vsd", type=int, default=0)
    args = parser.parse_args()
    
    prompt_ids, sampling_params = prepare_input(dataset, input_len, output_len, batch_size)

    with open(outfile, "a") as f:
        f.write(
            "draft_model, target_model, batch_size, dataset, num_spec_token, dsd, time, tpt\n"
        )
        
    # w/o SD
    if args.org:
        llm = LLM(model=target_model,
                gpu_memory_utilization=gpu_memory_utilization,
                disable_async_output_proc=True,
                tensor_parallel_size=tp,
                max_num_seqs=max_num_seqs)
        org_time, org_tpt = profile(llm, prompt_ids, sampling_params)
        del llm
        time.sleep(5)
        with open(outfile, "a") as f:
            f.write(
                f"{draft_model}, {target_model}, {batch_size}, {dataset}, -1, False, {org_time}, {org_tpt}\n"
            )
        
    # DSD
    if args.dsd:
        max_token = 6
        llm = LLM(model=target_model,
                        speculative_model=draft_model,
                        max_num_seqs=max_num_seqs - 5,
                        num_speculative_tokens=max_token,
                        gpu_memory_utilization=gpu_memory_utilization,
                        speculative_draft_tensor_parallel_size=1,
                        tensor_parallel_size=tp,
                        force_mqa=True,
                        dsd=True)
        dsd_time, dsd_tpt = profile(llm, prompt_ids, sampling_params)
        del llm
        time.sleep(5)

        with open(outfile, "a") as f:
            f.write(
                f"{draft_model}, {target_model}, {batch_size}, {dataset}, {max_token}, True, {dsd_time}, {dsd_tpt}\n"
            )
        
    # VSD
    if args.vsd > 0:
        llm = LLM(model=target_model,
                    speculative_model=draft_model,
                    max_num_seqs=max_num_seqs - 5,
                    num_speculative_tokens=args.vsd,
                    gpu_memory_utilization=gpu_memory_utilization,
                    speculative_draft_tensor_parallel_size=1,
                    tensor_parallel_size=tp,
                    force_mqa=True,
                    dsd=False)
        vsd_time, vsd_tpt = profile(llm, prompt_ids, sampling_params)
        del llm
        time.sleep(5)
        with open(outfile, "a") as f:
            f.write(
                f"{draft_model}, {target_model}, {batch_size}, {dataset}, {args.vsd}, False, {vsd_time}, {vsd_tpt}\n"
            )