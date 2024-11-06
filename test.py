from vllm import LLM, SamplingParams
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model='meta-llama/Llama-2-7b-hf',
          speculative_model='eqhylxx/vicuna-160m',
          disable_log_stats=False,
          num_speculative_tokens=5,
          use_v2_block_manager=True,
          acceptance_rate=0.7,
          dsd=True)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")