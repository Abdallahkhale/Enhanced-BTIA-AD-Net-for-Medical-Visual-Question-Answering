
import sys
import torch
import yaml
from pathlib import Path
import math

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.btia_ad_net import EnhancedBTIANet

def test_masking():
    print("Testing Answer Masking Logic...")
    
    # Mock config
    config = {
        'model': {
            'vision_encoder': 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
            'text_encoder': 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
            'hidden_dim': 768,
            'num_heads': 2,
            'num_attention_layers': 1,
            'dropout': 0.1,
            'top_k_answers': 5,
            'fusion_method': 'btia',
            'use_answer_distillation': False,
            'use_embedding_matching': False,
            'num_answers': 10  # 10 answers total
        }
    }
    
    # Initialize model (cpu is fine)
    model = EnhancedBTIANet(config['model'])
    model.eval()
    
    # Set indices
    yes_idx = 0
    no_idx = 1
    model.set_answer_indices(yes_idx, no_idx)
    model.yes_idx = yes_idx
    model.no_idx = no_idx
    
    print(f"Indices set: yes={yes_idx}, no={no_idx}")
    
    # Create dummy inputs
    B = 2
    images = torch.randn(B, 3, 224, 224)
    question_ids = torch.randint(0, 100, (B, 10))
    question_mask = torch.ones(B, 10)
    
    # is_closed: 0=Open, 1=Closed
    # Sample 0: Closed -> Should only allow 0, 1. Mask 2-9.
    # Sample 1: Open   -> Should mask 0, 1. Allow 2-9.
    is_closed = torch.tensor([1, 0])
    
    # Run model
    with torch.no_grad():
        outputs = model(
            images=images,
            question_ids=question_ids,
            question_mask=question_mask,
            is_closed=is_closed
        )
        logits = outputs['logits']
    
    print("\nChecking Closed Question (Sample 0)...")
    l0 = logits[0]
    print(f"Logits: {l0.tolist()}")
    if l0[yes_idx] > -100 and l0[no_idx] > -100:
        print("PASS: Yes/No allowed.")
    else:
        print("FAIL: Yes/No masked.")
        
    if l0[2].item() == -float('inf'):
        print("PASS: Open answers masked.")
    else:
        print(f"FAIL: Open answer not masked (val={l0[2].item()}).")

    print("\nChecking Open Question (Sample 1)...")
    l1 = logits[1]
    print(f"Logits: {l1.tolist()}")
    if l1[yes_idx] == -float('inf') and l1[no_idx] == -float('inf'):
        print("PASS: Yes/No masked.")
    else:
        print("FAIL: Yes/No not masked.")
        
    if l1[2] > -100:
        print("PASS: Open answers allowed.")
    else:
        print("FAIL: Open answers masked.")

if __name__ == "__main__":
    test_masking()
