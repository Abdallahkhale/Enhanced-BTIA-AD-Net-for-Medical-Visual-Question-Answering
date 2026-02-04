import sys
import torch
import yaml
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.btia_ad_net import create_model
from data.dataset import create_dataloaders
from src.utils import get_device

def inspect_predictions():
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = get_device()
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, test_loader, train_dataset = create_dataloaders(
        data_dir=config['data'].get('vqa_rad_dir', 'data').replace('/vqa_rad', ''),
        batch_size=8,
        num_workers=0,
        use_slake=False,
        use_pathvqa=True
    )
    
    # Update config
    config['model']['num_answers'] = train_dataset.num_answers
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    
    # Build cache
    print("Building answer cache...")
    model.build_answer_cache(train_dataset.get_answer_list(), device)
    
    # Load checkpoint
    ckpt_path = 'checkpoints/best_model.pt'
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print("\n Inspecting Open-Ended Predictions...")
    print("-" * 60)
    
    count = 0
    max_samples = 20
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['images'].to(device)
            question_ids = batch['question_ids'].to(device)
            question_mask = batch['question_mask'].to(device)
            targets = batch['answer_idx'].to(device)
            is_closed = batch['is_closed'].to(device)
            
            outputs = model(
                images=images,
                question_ids=question_ids,
                question_mask=question_mask
            )
            
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            # Get top 5
            top5_probs, top5_indices = torch.topk(probs, 5, dim=-1)
            
            for i in range(len(targets)):
                if not is_closed[i].item():
                    target_idx = targets[i].item()
                    pred_idx = predictions[i].item()
                    
                    target_ans = train_dataset.idx_to_answer[target_idx]
                    pred_ans = train_dataset.idx_to_answer[pred_idx]
                    
                    print(f"\nQ: {batch['questions'][i]}")
                    print(f"Target: '{target_ans}' (idx={target_idx})")
                    print(f"Pred:   '{pred_ans}' (idx={pred_idx})")
                    
                    print("Top 5 Predictions:")
                    for j in range(5):
                        idx = top5_indices[i, j].item()
                        prob = top5_probs[i, j].item()
                        ans = train_dataset.idx_to_answer[idx]
                        print(f"  {j+1}. '{ans}' ({prob:.4f})")
                    
                    count += 1
                    if count >= max_samples:
                        return

if __name__ == "__main__":
    inspect_predictions()
