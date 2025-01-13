from vllm.entrypoints.llm import LLM, SamplingParams
import os

prompts = [
    "This is just a test for LLM and cache in openai triton ",
]
sampling_params = SamplingParams(temperature=0.90, top_p=0.99, max_tokens=100)

llm = LLM(model="ibm-granite/granite-3.0-2b-base")
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}")
