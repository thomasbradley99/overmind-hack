
from llama_cpp import Llama
import os

model_path = "models/SmolVLM-Base.Q8_0.gguf"

try:
    print(f"Loading model from {model_path}...")
    # For SmolVLM, it might need chat_format="smolvlm" or similar if supported
    # But let's try basic load first.
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=0, # CPU only for now
    )
    print("Model loaded successfully!")
    
    # Check if it has vision capabilities
    # In llama-cpp-python, we usually check if clip_model_path was provided.
    # But some models have it integrated.
    
    prompt = "<|user|>\nDescribe this image.<|assistant|>\n"
    # Note: Without an image, we can't really test vision, 
    # but let's see if it can generate text.
    output = llm(prompt, max_tokens=10)
    print(f"Output: {output['choices'][0]['text']}")

except Exception as e:
    print(f"Error: {e}")
