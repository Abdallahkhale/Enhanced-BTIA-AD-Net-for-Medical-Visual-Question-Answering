import torch
import transformers
try:
    from transformers import AutoTokenizer
    print("Loading BioGPT Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT")
    print("Success.")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
