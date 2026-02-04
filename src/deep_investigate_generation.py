import torch
import sys
import os
sys.path.append(os.getcwd())
from models.generative_net import GenerativeMedVQA
from transformers import AutoTokenizer
import yaml
import os

def investigate():
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    print("Loading model...")
    model = GenerativeMedVQA(config['model'])
    
    # Load checkpoint if available (best or latest)
    checkpoint_path = 'checkpoints/model_epoch_1.pt' # Assuming epoch 1 saved
    if not os.path.exists(checkpoint_path):
        # Try finding any checkpoint
        checkpoints = [f for f in os.listdir('checkpoints') if f.endswith('.pt')]
        if checkpoints:
            checkpoint_path = os.path.join('checkpoints', checkpoints[-1])
            
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        print("No checkpoint found! Using random weights (results will be nonsense but logic can be tested).")
        
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(config['model']['text_decoder'])
    
    # Dummy inputs for simulation (since we don't want to load full dataset for quick test)
    # We'll valid inputs "Question: is there evidence of an aortic aneurysm? Answer:"
    
    questions = [
        "Question: is there evidence of an aortic aneurysm? Answer:",
        "Question: is there airspace consolidation on the left side? Answer:",
        "Question: is there any intraparenchymal abnormalities in the lung fields? Answer:"
    ]
    
    # Dummy image (random)
    image = torch.randn(1, 3, 224, 224).to(device)
    
    print("\n--- Testing Generation Parameters ---")
    
    configs_to_test = [
        {'name': 'Default', 'kwargs': {'max_new_tokens': 20}},
        {'name': 'Repetition Penalty 1.2', 'kwargs': {'max_new_tokens': 20, 'repetition_penalty': 1.2}},
        {'name': 'Repetition Penalty 1.5', 'kwargs': {'max_new_tokens': 20, 'repetition_penalty': 1.5}},
        {'name': 'Beam Search (num_beams=3)', 'kwargs': {'max_new_tokens': 20, 'num_beams': 3, 'early_stopping': True}},
        {'name': 'No Repeat Ngram Size 2', 'kwargs': {'max_new_tokens': 20, 'no_repeat_ngram_size': 2}},
    ]
    
    for q in questions:
        print(f"\nQuery: {q}")
        inputs = tokenizer(q, return_tensors="pt")
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        for cfg in configs_to_test:
            print(f"  Configuration: {cfg['name']}")
            try:
                out = model.generate(
                    images=image,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **cfg['kwargs']
                )
                pred = tokenizer.decode(out[0], skip_special_tokens=True)
                # Remove the prompt
                answer_only = pred.replace(q, "").strip()
                print(f"    Result: {answer_only}")
                
            except Exception as e:
                print(f"    Error: {e}")

if __name__ == "__main__":
    investigate()
