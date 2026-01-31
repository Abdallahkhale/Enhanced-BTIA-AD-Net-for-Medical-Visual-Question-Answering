"""
Evaluation Script for Enhanced BTIA-AD Net
Detailed evaluation with per-category breakdown.
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.btia_ad_net import EnhancedBTIANet, create_model
from data.dataset import create_dataloaders, MedVQADataset
from src.utils import load_config, get_device, load_checkpoint


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_fp16: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive evaluation.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Target device
        use_fp16: Whether to use FP16
        
    Returns:
        Dictionary with detailed metrics
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_is_closed = []
    all_correct = []
    
    # For per-category analysis
    category_correct = defaultdict(list)
    
    for batch in tqdm(test_loader, desc="Evaluating"):
        images = batch['images'].to(device, non_blocking=True)
        question_ids = batch['question_ids'].to(device, non_blocking=True)
        question_mask = batch['question_mask'].to(device, non_blocking=True)
        targets = batch['answer_idx'].to(device, non_blocking=True)
        is_closed = batch['is_closed'].to(device, non_blocking=True)
        
        if use_fp16:
            with autocast():
                outputs = model(
                    images=images,
                    question_ids=question_ids,
                    question_mask=question_mask
                )
        else:
            outputs = model(
                images=images,
                question_ids=question_ids,
                question_mask=question_mask
            )
        
        predictions = torch.argmax(outputs['logits'], dim=-1)
        correct = (predictions == targets)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_is_closed.extend(is_closed.cpu().numpy())
        all_correct.extend(correct.cpu().numpy())
    
    # Convert to arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_is_closed = np.array(all_is_closed)
    all_correct = np.array(all_correct)
    
    # Overall metrics
    overall_accuracy = all_correct.mean()
    
    # Closed-ended accuracy
    closed_mask = all_is_closed.astype(bool)
    closed_accuracy = all_correct[closed_mask].mean() if closed_mask.any() else 0.0
    
    # Open-ended accuracy
    open_mask = ~closed_mask
    open_accuracy = all_correct[open_mask].mean() if open_mask.any() else 0.0
    
    results = {
        'overall_accuracy': overall_accuracy,
        'closed_accuracy': closed_accuracy,
        'open_accuracy': open_accuracy,
        'num_samples': len(all_predictions),
        'num_closed': closed_mask.sum(),
        'num_open': open_mask.sum()
    }
    
    return results


def print_results(results: Dict[str, Any]):
    """Print evaluation results in a formatted table."""
    print("\n" + "=" * 60)
    print("          VQA-RAD Evaluation Results")
    print("=" * 60)
    
    print(f"\n{'Metric':<30} {'Value':>15}")
    print("-" * 45)
    
    print(f"{'Overall Accuracy':<30} {results['overall_accuracy']*100:>14.2f}%")
    print(f"{'Closed-Ended Accuracy':<30} {results['closed_accuracy']*100:>14.2f}%")
    print(f"{'Open-Ended Accuracy':<30} {results['open_accuracy']*100:>14.2f}%")
    
    print("\n" + "-" * 45)
    print(f"{'Total Samples':<30} {results['num_samples']:>15}")
    print(f"{'Closed-Ended Samples':<30} {results['num_closed']:>15}")
    print(f"{'Open-Ended Samples':<30} {results['num_open']:>15}")
    
    print("\n" + "=" * 60)
    
    # Compare with paper baseline
    print("\nComparison with BTIA-AD Net Paper:")
    print("-" * 45)
    print(f"{'Metric':<25} {'Paper':>12} {'Ours':>12}")
    print("-" * 45)
    print(f"{'Overall':<25} {'70.22%':>12} {results['overall_accuracy']*100:>11.2f}%")
    print(f"{'Closed-Ended':<25} {'78.68%':>12} {results['closed_accuracy']*100:>11.2f}%")
    print(f"{'Open-Ended':<25} {'57.54%':>12} {results['open_accuracy']*100:>11.2f}%")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Enhanced BTIA-AD Net")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file path")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--output", type=str, default=None, help="Output file for results")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup device
    device = get_device()
    
    # Create test loader
    print("Loading test dataset...")
    _, test_loader, train_dataset = create_dataloaders(
        data_dir=config['data'].get('vqa_rad_dir', 'data').replace('/vqa_rad', ''),
        batch_size=config['training']['batch_size'],
        num_workers=0,  # Use 0 for evaluation
        use_slake=False,  # Only evaluate on VQA-RAD
        image_size=config['data'].get('image_size', 224),
        max_question_length=config['data'].get('max_question_length', 64)
    )
    
    # Update config with actual number of answers
    config['model']['num_answers'] = train_dataset.num_answers
    
    # Create model
    print("Loading model...")
    model = create_model(config)
    
    # Build answer cache
    model.build_answer_cache(train_dataset.get_answer_list(), device)
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.checkpoint, model)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    model = model.to(device)
    
    # Evaluate
    print("Starting evaluation...")
    results = evaluate(
        model, test_loader, device,
        use_fp16=config['training'].get('fp16', True)
    )
    
    # Print results
    print_results(results)
    
    # Save results if output specified
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
