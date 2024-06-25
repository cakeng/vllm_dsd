from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, 
                                 top_p=1.0,
                                 max_tokens=15)

# Create an LLM.
llm = LLM(model="lmsys/vicuna-7b-v1.5",
          speculative_model="eqhylxx/vicuna-160m",
          num_speculative_tokens=5,
          use_v2_block_manager=True,
          enforce_eager=True,
          disable_log_stats=False
          )
# llm = LLM(model="facebook/opt-125m")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
print("======================================")
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
