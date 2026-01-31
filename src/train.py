"""
Training Script for Enhanced BTIA-AD Net
Supports FP16 training, gradient checkpointing, and RTX 5070 Ti optimizations.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.btia_ad_net import EnhancedBTIANet, create_model
from data.dataset import create_dataloaders, MedVQADataset
from src.utils import (
    set_seed, load_config, save_config, setup_logging,
    get_device, count_parameters, format_time,
    AverageMeter, EarlyStopping, save_checkpoint, load_checkpoint,
    compute_accuracy
)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    device: torch.device,
    epoch: int,
    config: Dict[str, Any],
    logger=None
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        optimizer: Optimizer
        scaler: Gradient scaler for FP16
        device: Target device
        epoch: Current epoch number
        config: Training configuration
        logger: Optional logger
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    
    loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()
    ad_loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    use_fp16 = config['training'].get('fp16', True)
    grad_accum_steps = config['training'].get('gradient_accumulation_steps', 1)
    ad_weight = config['training'].get('ad_loss_weight', 0.3)
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        images = batch['images'].to(device, non_blocking=True)
        question_ids = batch['question_ids'].to(device, non_blocking=True)
        question_mask = batch['question_mask'].to(device, non_blocking=True)
        targets = batch['answer_idx'].to(device, non_blocking=True)
        is_closed = batch['is_closed'].to(device, non_blocking=True)
        
        # Forward pass with mixed precision
        if use_fp16 and scaler is not None:
            with autocast():
                outputs = model(
                    images=images,
                    question_ids=question_ids,
                    question_mask=question_mask
                )
                
                # Compute loss
                losses = model.compute_loss(outputs, targets, ad_weight=ad_weight)
                loss = losses['loss'] / grad_accum_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(
                images=images,
                question_ids=question_ids,
                question_mask=question_mask
            )
            
            losses = model.compute_loss(outputs, targets, ad_weight=ad_weight)
            loss = losses['loss'] / grad_accum_steps
            
            loss.backward()
            
            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
        
        # Compute accuracy
        with torch.no_grad():
            acc = compute_accuracy(outputs['logits'], targets, is_closed)
        
        # Update meters
        batch_size = images.size(0)
        loss_meter.update(losses['loss'].item(), batch_size)
        cls_loss_meter.update(losses['cls_loss'].item(), batch_size)
        ad_loss_meter.update(losses['ad_loss'].item(), batch_size)
        acc_meter.update(acc['overall'], batch_size)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss_meter.avg:.4f}",
            'acc': f"{acc_meter.avg:.4f}"
        })
    
    return {
        'loss': loss_meter.avg,
        'cls_loss': cls_loss_meter.avg,
        'ad_loss': ad_loss_meter.avg,
        'accuracy': acc_meter.avg
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """
    Validate the model.
    
    Args:
        model: The model to validate
        val_loader: Validation data loader
        device: Target device
        config: Configuration
        
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    
    loss_meter = AverageMeter()
    overall_correct = 0
    closed_correct = 0
    open_correct = 0
    overall_total = 0
    closed_total = 0
    open_total = 0
    
    use_fp16 = config['training'].get('fp16', True)
    ad_weight = config['training'].get('ad_loss_weight', 0.3)
    
    for batch in tqdm(val_loader, desc="Validating"):
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
                losses = model.compute_loss(outputs, targets, ad_weight=ad_weight)
        else:
            outputs = model(
                images=images,
                question_ids=question_ids,
                question_mask=question_mask
            )
            losses = model.compute_loss(outputs, targets, ad_weight=ad_weight)
        
        # Compute predictions
        predictions = torch.argmax(outputs['logits'], dim=-1)
        correct = (predictions == targets)
        
        # Update totals
        batch_size = images.size(0)
        loss_meter.update(losses['loss'].item(), batch_size)
        
        overall_correct += correct.sum().item()
        overall_total += batch_size
        
        closed_mask = is_closed.bool()
        open_mask = ~closed_mask
        
        closed_correct += correct[closed_mask].sum().item()
        closed_total += closed_mask.sum().item()
        
        open_correct += correct[open_mask].sum().item()
        open_total += open_mask.sum().item()
        
        # Debug: Log first batch of open predictions
        if open_mask.sum() > 0 and overall_total <= batch_size * 2:
            open_preds = predictions[open_mask]
            open_targets = targets[open_mask]
            logger.debug(f"Open predictions sample: {open_preds[:5].tolist()}")
            logger.debug(f"Open targets sample: {open_targets[:5].tolist()}")
    
    # Compute final metrics
    overall_acc = overall_correct / max(overall_total, 1)
    closed_acc = closed_correct / max(closed_total, 1)
    open_acc = open_correct / max(open_total, 1)
    
    return {
        'loss': loss_meter.avg,
        'overall_accuracy': overall_acc,
        'closed_accuracy': closed_acc,
        'open_accuracy': open_acc
    }


def train(config: Dict[str, Any], args: argparse.Namespace):
    """
    Main training function.
    
    Args:
        config: Training configuration
        args: Command line arguments
    """
    # Setup
    set_seed(42)
    device = get_device()
    
    # Setup logging
    log_dir = Path(config['logging'].get('log_dir', 'logs'))
    logger = setup_logging(str(log_dir))
    logger.info(f"Starting training with config: {config}")
    
    # Setup TensorBoard
    writer = None
    if TENSORBOARD_AVAILABLE and config['logging'].get('use_tensorboard', True):
        writer = SummaryWriter(log_dir / 'tensorboard')
    
    # Create checkpoint directory
    checkpoint_dir = Path(config['training'].get('checkpoint_dir', 'checkpoints'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, train_dataset = create_dataloaders(
        data_dir=config['data'].get('vqa_rad_dir', 'data').replace('/vqa_rad', ''),
        batch_size=config['training']['batch_size'],
        num_workers=config['data'].get('num_workers', 4),
        use_slake=config['data'].get('use_slake_augmentation', True),
        use_pathvqa=config['data'].get('use_pathvqa_augmentation', False),
        image_size=config['data'].get('image_size', 224),
        max_question_length=config['data'].get('max_question_length', 64)
    )
    
    # Update config with actual number of answers
    config['model']['num_answers'] = train_dataset.num_answers
    logger.info(f"Number of answers: {train_dataset.num_answers}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config)
    
    # Move model to device FIRST
    model = model.to(device)
    
    # Build answer cache (model must be on device first)
    logger.info("Building answer embedding cache...")
    model.build_answer_cache(train_dataset.get_answer_list(), device)
    
    # Set class weights for balanced training
    if hasattr(train_dataset, 'get_class_weights'):
        logger.info("Setting class weights for balanced training...")
        class_weights = train_dataset.get_class_weights()
        model.set_class_weights(class_weights, device)
        logger.info(f"Class weights min: {class_weights.min():.4f}, max: {class_weights.max():.4f}")
    
    # Log parameter counts
    param_counts = count_parameters(model)
    logger.info(f"Model parameters: {param_counts}")
    
    # Create optimizer
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    
    # Different learning rates for pretrained vs new parameters
    param_groups = model.get_trainable_params()
    optimizer = AdamW([
        {'params': param_groups[0]['params'], 'lr': learning_rate * 0.1},  # Encoders
        {'params': param_groups[1]['params'], 'lr': learning_rate}  # New layers
    ], weight_decay=weight_decay)
    
    # Create scheduler
    num_epochs = config['training']['epochs']
    warmup_ratio = config['training'].get('warmup_ratio', 0.1)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[learning_rate * 0.1, learning_rate],
        total_steps=total_steps,
        pct_start=warmup_ratio,
        anneal_strategy='cos'
    )
    
    # Create gradient scaler for FP16
    scaler = GradScaler() if config['training'].get('fp16', True) else None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training'].get('early_stopping_patience', 10),
        mode='max'
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0.0
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['metrics'].get('overall_accuracy', 0.0)
    
    # Training loop
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"{'='*50}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch + 1, config, logger
        )
        
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        
        # Validate
        val_metrics = validate(model, val_loader, device, config)
        
        logger.info(
            f"Val - Loss: {val_metrics['loss']:.4f}, "
            f"Overall: {val_metrics['overall_accuracy']:.4f}, "
            f"Closed: {val_metrics['closed_accuracy']:.4f}, "
            f"Open: {val_metrics['open_accuracy']:.4f}"
        )
        
        # Update learning rate
        if scheduler is not None:
            # OneCycleLR updates per step, not per epoch
            pass
        
        # TensorBoard logging
        if writer is not None:
            writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
            writer.add_scalar('Train/Accuracy', train_metrics['accuracy'], epoch)
            writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
            writer.add_scalar('Val/Overall_Accuracy', val_metrics['overall_accuracy'], epoch)
            writer.add_scalar('Val/Closed_Accuracy', val_metrics['closed_accuracy'], epoch)
            writer.add_scalar('Val/Open_Accuracy', val_metrics['open_accuracy'], epoch)
        
        # Save checkpoint
        if config['training'].get('save_every_epoch', True):
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_metrics,
                str(checkpoint_dir / f"epoch_{epoch + 1}.pt")
            )
        
        # Save best model
        if val_metrics['overall_accuracy'] > best_acc:
            best_acc = val_metrics['overall_accuracy']
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_metrics,
                str(checkpoint_dir / "best_model.pt")
            )
            logger.info(f"New best model saved! Accuracy: {best_acc:.4f}")
        
        # Early stopping check
        if early_stopping(val_metrics['overall_accuracy']):
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break
    
    logger.info("\nTraining complete!")
    logger.info(f"Best validation accuracy: {best_acc:.4f}")
    
    if writer is not None:
        writer.close()
    
    return best_acc


def main():
    parser = argparse.ArgumentParser(description="Train Enhanced BTIA-AD Net")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file path")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--debug", action="store_true", help="Debug mode (1 epoch, small batch)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Apply overrides
    if args.debug:
        config['training']['epochs'] = 1
        config['training']['batch_size'] = 4
        config['data']['num_workers'] = 0
    
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    
    # Run training
    train(config, args)


if __name__ == "__main__":
    main()
