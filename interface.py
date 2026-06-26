import ctypes
import os
from typing import List

class AlienLLMInterface:
    def __init__(self, lib_path: str, checkpoint_path: str, d_model: int = 256):
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Shared library not found at {lib_path}")
        
        self.lib = ctypes.CDLL(lib_path)
        
        # Define argument and return types
        self.lib.create_model.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.create_model.restype = ctypes.c_void_p
        
        self.lib.destroy_model.argtypes = [ctypes.c_void_p]
        self.lib.destroy_model.restype = None
        
        self.lib.reset_model.argtypes = [ctypes.c_void_p]
        self.lib.reset_model.restype = None
        
        self.lib.generate_token.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.generate_token.restype = ctypes.c_char_p
        
        self.lib.encode_text.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
        self.lib.encode_text.restype = ctypes.c_int
        
        self.lib.process_tokens.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self.lib.process_tokens.restype = None

        self.lib.get_last_token_id.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.get_last_token_id.restype = ctypes.c_int
        
        # Create model instance
        self.instance = self.lib.create_model(checkpoint_path.encode('utf-8'), d_model)

    def __del__(self):
        if hasattr(self, 'instance') and self.instance:
            self.lib.destroy_model(self.instance)

    def reset(self):
        self.lib.reset_model(self.instance)

    def generate(self, prompt: str, max_tokens: int = 50):
        self.reset()
        
        # Encode prompt and process it
        tokens_array = (ctypes.c_int * len(prompt))()
        count = self.lib.encode_text(self.instance, prompt.encode('utf-8'), tokens_array)
        self.lib.process_tokens(self.instance, tokens_array, count)
        
        last_token_id = self.lib.get_last_token_id(self.instance, prompt.encode('utf-8'))
        
        generated_text = ""
        for _ in range(max_tokens):
            token_str = self.lib.generate_token(self.instance, last_token_id)
            if not token_str:
                break
            
            decoded_token = token_str.decode('utf-8')
            generated_text += decoded_token + " "
            
            # Update last_token_id (simplified for this demo)
            # In a real model, we'd get the actual ID of the generated token
            # Here we just use a placeholder or re-encode if necessary
            # For this bridge, we'll assume the C++ side handles state in process_token
            # but we need a token_id to pass back.
            # Let's re-encode the last token to get its ID
            last_token_id = self.lib.get_last_token_id(self.instance, decoded_token.encode('utf-8'))
            
            yield decoded_token

if __name__ == "__main__":
    # Test the interface
    lib_path = "./build/libalien_llm_lib.so"
    checkpoint_path = "./model_checkpoint.bin"
    
    try:
        model = AlienLLMInterface(lib_path, checkpoint_path)
        print("Model loaded successfully!")
        
        prompt = "Once upon a time"
        print(f"Prompt: {prompt}")
        print("Response: ", end="", flush=True)
        for token in model.generate(prompt, max_tokens=10):
            print(token, end=" ", flush=True)
        print("\n")
    except Exception as e:
        print(f"Error: {e}")
