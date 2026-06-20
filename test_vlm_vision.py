
from llama_cpp import Llama
import os

model_path = "models/SmolVLM-Base.Q8_0.gguf"

try:
    print(f"Loading model from {model_path}...")
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=0,
    )
    print("Model loaded successfully!")
    
    # Check if we can use images
    import base64
    # Just a dummy tiny image (1x1 black pixel)
    dummy_image = base64.b64encode(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82').decode('utf-8')
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{dummy_image}"}}
            ]
        }
    ]
    
    print("Testing chat completion with image...")
    # This will likely fail if vision is not properly initialized
    response = llm.create_chat_completion(messages=messages, max_tokens=20)
    print(f"Response: {response['choices'][0]['message']['content']}")

except Exception as e:
    print(f"Error: {e}")
