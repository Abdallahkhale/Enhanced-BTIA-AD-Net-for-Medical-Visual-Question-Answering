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
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.amp import GradScaler, autocast

from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.btia_ad_net import EnhancedBTIANet, create_model as create_discriminative_model
from models.generative_net import create_generative_model
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
        # Generative inputs
        if 'input_ids' in batch:
             input_ids = batch['input_ids'].to(device, non_blocking=True)
             attention_mask = batch['attention_mask'].to(device, non_blocking=True)
             labels = batch['labels'].to(device, non_blocking=True)
             question_ids = None # Not used
             question_mask = None
             targets = None
        else:
             # Discriminative inputs
             question_ids = batch['question_ids'].to(device, non_blocking=True)
             question_mask = batch['question_mask'].to(device, non_blocking=True)
             targets = batch['answer_idx'].to(device, non_blocking=True)
             
        is_closed = batch['is_closed'].to(device, non_blocking=True)
        
        # Forward pass with mixed precision
        if use_fp16 and scaler is not None:
            with autocast(device_type='cuda'):
                if 'input_ids' in batch:
                    # Generative
                    outputs = model(
                        images=images,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    losses = {'loss': outputs.loss}
                else:
                    # Discriminative
                    outputs = model(
                        images=images,
                        question_ids=question_ids,
                        question_mask=question_mask,
                        is_closed=is_closed
                    )
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
            if 'input_ids' in batch:
                # Generative
                outputs = model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                losses = {'loss': outputs.loss}
            else:
                outputs = model(
                    images=images,
                    question_ids=question_ids,
                    question_mask=question_mask,
                    is_closed=is_closed
                )
                losses = model.compute_loss(outputs, targets, ad_weight=ad_weight)
            
            loss = losses['loss'] / grad_accum_steps
            
            loss.backward()
            
            if (batch_idx + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
        # Compute accuracy (skip for generative)
        if targets is not None:
             with torch.no_grad():
                 acc = compute_accuracy(outputs['logits'], targets, is_closed)
             acc_val = acc['overall']
             cls_loss_val = losses.get('cls_loss', torch.tensor(0.0)).item()
             ad_loss_val = losses.get('ad_loss', torch.tensor(0.0)).item()
        else:
             # Generative Accuracy (Next Token Prediction) for TRAINING progress bar
             with torch.no_grad():
                 logits = outputs.logits
                 # Slice logits to remove visual prefix if needed
                 # labels are text only. logits are image+text.
                 if logits.shape[1] > labels.shape[1]:
                     n_vis = logits.shape[1] - labels.shape[1]
                     text_logits = logits[:, n_vis:, :]
                 else:
                     text_logits = logits
                     
                 shift_logits = text_logits[..., :-1, :].contiguous()
                 shift_labels = labels[..., 1:].contiguous()
                 
                 preds = torch.argmax(shift_logits, dim=-1)
                 mask = shift_labels != -100
                 
                 if mask.sum() > 0:
                     correct = (preds == shift_labels) & mask
                     acc_val = (correct.sum().float() / mask.sum().float()).item()
                 else:
                     acc_val = 0.0
                     
             cls_loss_val = 0.0
             ad_loss_val = 0.0
        
        # Update meters
        batch_size = images.size(0)
        loss_meter.update(losses['loss'].item(), batch_size)
        cls_loss_meter.update(cls_loss_val, batch_size)
        ad_loss_meter.update(ad_loss_val, batch_size)
        acc_meter.update(acc_val, batch_size)
        
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
    config: Dict[str, Any],
    logger=None
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
    
    for i, batch in enumerate(tqdm(val_loader, desc="Validating")):
        images = batch['images'].to(device, non_blocking=True)
        images = batch['images'].to(device, non_blocking=True)
        is_closed = batch['is_closed'].to(device, non_blocking=True)
        
        # Initialize discriminative inputs as None
        question_ids = None
        question_mask = None
        targets = None
        
        if 'input_ids' not in batch:
            question_ids = batch['question_ids'].to(device, non_blocking=True)
            question_mask = batch['question_mask'].to(device, non_blocking=True)
            targets = batch['answer_idx'].to(device, non_blocking=True)
        
        if use_fp16:
            with autocast(device_type='cuda'):
                if 'input_ids' in batch:
                    # Generative
                    input_ids = batch['input_ids'].to(device, non_blocking=True)
                    attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                    labels = batch['labels'].to(device, non_blocking=True)
                    
                    outputs = model(
                        images=images,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    losses = {'loss': outputs.loss}
                    
                    # Log predictions for the first batch
                    if logger and i == 0 and hasattr(model, 'generate'):
                         # Use tokenizer from dataset if available
                         tokenizer = val_loader.dataset.tokenizer
                         
                         # Generate for first 3 samples
                         logger.info("\n--- Generative Debug (Batch 0) ---")
                         
                         for k in range(len(images)): # Log all samples in the batch
                             # Find prompt length (where labels != -100)
                             # Labels are -100 for prompt, then valid for answer.
                             # But wait, input_ids are [Q_tokens, A_tokens]. 
                             # We want to give [Q_tokens] to generate.
                             # The labels are -100 for Q_tokens.
                             # So we find the first index where label != -100 (start of answer)
                             # OR just take the start of the answer.
                             
                             curr_labels = labels[k]
                             # Find index where labels != -100
                             answer_start = (curr_labels != -100).nonzero(as_tuple=True)[0]
                             
                             if len(answer_start) > 0:
                                 prompt_len = answer_start[0].item()
                             else:
                                 # Fallback if no answer (shouldn't happen in training)
                                 prompt_len = len(curr_labels) // 2
                                 
                             prompt_ids = input_ids[k:k+1, :prompt_len]
                             prompt_mask = attention_mask[k:k+1, :prompt_len]
                             
                             gen_out = model.generate(
                                 images=images[k:k+1],
                                 input_ids=prompt_ids,
                                 attention_mask=prompt_mask,
                                 max_new_tokens=20,
                                 repetition_penalty=1.2  # Fix "yes yes yes" loop
                             )
                             
                             gen_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
                             gen_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
                             # Robust answer extraction
                             # Since we passed prompt_ids, the prompt is at the beginning.
                             # But decoding prompt_ids again might differ slightly from gen_text due to spacing/merges.
                             # Let's just strip the known prompt string logic.
                             prompt_str = tokenizer.decode(prompt_ids[0], skip_special_tokens=True)
                             if gen_text.startswith(prompt_str):
                                 answer_only = gen_text[len(prompt_str):].strip()
                             else:
                                 # Fallback: display full generation if mismatch
                                 # (This often happens if prompt has whitespace differences)
                                 # Try stripping the first N chars slightly less accurately or just show full.
                                 answer_only = "[FULL] " + gen_text

                             gt_text = tokenizer.decode(input_ids[k], skip_special_tokens=True)
                             
                             q_type = "[Closed]" if is_closed[k].item() else "[Open]"
                             
                             logger.info(f"Sample {k+1} {q_type}:")
                             logger.info(f"  GT: {gt_text}")
                             logger.info(f"  Pred: {answer_only.strip()}")
                             
                         logger.info("----------------------------------\n")
                    
                    # Compute Generative Accuracy (Contains Match)
                    # We need to generate for the WHOLE batch to get metrics, 
                    # but that's slow. For now, let's just stick to loss as primary metric.
                    # Or do a quick greedy on tokens for accuracy?
                    # Let's use the provided labels to check if top-1 token matches?
                    # CausalLM accuracy is next-token prediction accuracy.
                    
                    # Flatten stats
                    # Flatten stats
                    logits = outputs.logits # [B, N_vis + L, V] - BioGPT includes image tokens
                    
                    # We must Slice logits to remove the visual prefix
                    # The labels [B, L] correspond to the TEXT part only (image part handled in forward)
                    # Wait, in forward() we CAT inputs: [Image, Text].
                    # So logits are [Image_Logits, Text_Logits].
                    # We need to slice off the first N_vis logits.
                    # N_vis = model.vis_hidden_dim? No, N_vis is sequence length.
                    # For ViT it's 197. For ResNet reshaped it's 2048? No, it's 7x7=49.
                    # Let's deduce N_vis from the shape difference.
                    # logits width = N_vis + L
                    # labels width = L
                    n_vis = logits.shape[1] - labels.shape[1]
                    
                    text_logits = logits[:, n_vis:, :] # [B, L, V]
                    
                    shift_logits = text_logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    # Flatten
                    preds = torch.argmax(shift_logits, dim=-1)
                    mask = shift_labels != -100
                    
                    # Initialize metrics dict
                    acc = {'overall': 0.0}
                    
                    if mask.sum() > 0:
                        correct = (preds == shift_labels) & mask
                        batch_acc = correct.sum().float() / mask.sum().float()
                        acc['overall'] = batch_acc.item()
                    else:
                        acc['overall'] = 0.0

                else:
                    # Discriminative
                    outputs = model(
                        images=images,
                        question_ids=question_ids,
                        question_mask=question_mask,
                        is_closed=is_closed
                    )
                    losses = model.compute_loss(outputs, targets, ad_weight=ad_weight)
        else:
            if 'input_ids' in batch:
                # Generative
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                
                outputs = model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                losses = {'loss': outputs.loss}
            else:
                outputs = model(
                    images=images,
                    question_ids=question_ids,
                    question_mask=question_mask,
                    is_closed=is_closed
                )
                losses = model.compute_loss(outputs, targets, ad_weight=ad_weight)
        
        # Compute predictions
        batch_size = images.size(0)
        loss_meter.update(losses['loss'].item(), batch_size)
        
        if 'input_ids' not in batch:
            # Discriminative metrics
            predictions = torch.argmax(outputs['logits'], dim=-1)
            correct = (predictions == targets)
            
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
                if logger is not None:
                    logger.debug(f"Open predictions sample: {open_preds[:5].tolist()}")
                    logger.debug(f"Open targets sample: {open_targets[:5].tolist()}")
        else:
            # Generative metrics - compute actual VQA accuracy
            # Generate answers and compare with ground truth
            if 'input_ids' in batch and hasattr(model, 'generate'):
                tokenizer = val_loader.dataset.tokenizer
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask_batch = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                
                for k in range(batch_size):
                    # Find prompt length (where labels != -100)
                    curr_labels = labels[k]
                    answer_start = (curr_labels != -100).nonzero(as_tuple=True)[0]
                    
                    if len(answer_start) > 0:
                        prompt_len = answer_start[0].item()
                    else:
                        prompt_len = len(curr_labels) // 2
                    
                    prompt_ids = input_ids[k:k+1, :prompt_len]
                    prompt_mask = attention_mask_batch[k:k+1, :prompt_len]
                    
                    # Generate prediction
                    with torch.no_grad():
                        gen_out = model.generate(
                            images=images[k:k+1],
                            input_ids=prompt_ids,
                            attention_mask=prompt_mask,
                            max_new_tokens=20
                        )
                    
                    # Decode generated text
                    gen_text = tokenizer.decode(gen_out[0], skip_special_tokens=True)
                    prompt_str = tokenizer.decode(prompt_ids[0], skip_special_tokens=True)
                    
                    # Extract answer only
                    if gen_text.startswith(prompt_str):
                        pred_answer = gen_text[len(prompt_str):].strip().lower()
                    else:
                        pred_answer = gen_text.strip().lower()
                    
                    # Get ground truth answer from input_ids (after prompt)
                    gt_tokens = input_ids[k, prompt_len:]
                    gt_text = tokenizer.decode(gt_tokens, skip_special_tokens=True).strip().lower()
                    
                    # Check for match (exact or contains)
                    # === IMPROVED FUZZY MATCHING ===
                    def normalize_answer(s):
                        s = s.lower().strip()
                        s = re.sub(r'[^\w\s]', '', s)  # Remove punctuation
                        s = re.sub(r'\s+', ' ', s)     # Normalize spaces
                        return s
                    
                    pred_norm = normalize_answer(pred_answer)
                    gt_norm = normalize_answer(gt_text)
                    
                    # Synonym mapping for medical yes/no answers
                    yes_syns = {'yes', 'true', 'positive', 'present', 'visible', 'seen'}
                    no_syns = {'no', 'false', 'negative', 'absent', 'not visible', 'not seen', 'none'}
                    
                    def check_synonym_match(p, g):
                        if p in yes_syns and g in yes_syns:
                            return True
                        if p in no_syns and g in no_syns:
                            return True
                        # Location synonyms
                        if ('left' in p and 'left' in g) or ('right' in p and 'right' in g):
                            return True
                        return False
                    
                    def word_overlap(p, g):
                        pw = set(p.split())
                        gw = set(g.split())
                        if len(gw) == 0:
                            return 0
                        return len(pw & gw) / len(gw)
                    
                    is_correct = (pred_norm == gt_norm) or \
                                 (gt_norm in pred_norm) or \
                                 (pred_norm in gt_norm and len(pred_norm) > 1) or \
                                 check_synonym_match(pred_norm, gt_norm) or \
                                 word_overlap(pred_norm, gt_norm) >= 0.5
                    
                    if is_correct:
                        overall_correct += 1
                        if is_closed[k].item():
                            closed_correct += 1
                        else:
                            open_correct += 1
                    
                    overall_total += 1
                    if is_closed[k].item():
                        closed_total += 1
                    else:
                        open_total += 1
    
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
    is_generative = config['model'].get('type') == 'generative'
    tokenizer_name = config['model'].get('text_decoder', "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    
    train_loader, val_loader, train_dataset = create_dataloaders(
        data_dir=config['data'].get('vqa_rad_dir', 'data').replace('/vqa_rad', ''),
        batch_size=config['training']['batch_size'],
        num_workers=config['data'].get('num_workers', 4),
        use_slake=config['data'].get('use_slake_augmentation', True),
        use_pathvqa=config['data'].get('use_pathvqa_augmentation', False),
        image_size=config['data'].get('image_size', 224),
        max_question_length=config['data'].get('max_question_length', 64),
        generative_mode=is_generative,
        tokenizer_name=tokenizer_name
    )
    
    # Update config with actual number of answers (only relevant for discriminative)
    if not is_generative:
        config['model']['num_answers'] = train_dataset.num_answers
        logger.info(f"Number of answers: {train_dataset.num_answers}")
    
    # Create model
    logger.info("Creating model...")
    if is_generative:
        model = create_generative_model(config)
    else:
        model = create_discriminative_model(config)
    
    # Move model to device FIRST
    model = model.to(device)
    
    # Build answer cache (only discriminative)
    if not is_generative:
        logger.info("Building answer embedding cache...")
        model.build_answer_cache(train_dataset.get_answer_list(), device)
        
        # Set class weights
        if hasattr(train_dataset, 'get_class_weights'):
             class_weights = train_dataset.get_class_weights()
             model.set_class_weights(class_weights, device)
    
    # Log parameter counts
    param_counts = count_parameters(model)
    logger.info(f"Model parameters: {param_counts}")
    
    # Two-stage training configuration (LLaVA approach)
    pretraining_epochs = config['model'].get('pretraining_epochs', 0) if is_generative else 0
    if pretraining_epochs > 0 and is_generative:
        logger.info(f"\n*** Two-Stage Training Enabled ***")
        logger.info(f"Stage 1: {pretraining_epochs} epochs training projection only (LLM frozen)")
        logger.info(f"Stage 2: Remaining epochs training projection + LLM")
        # Start in pretraining mode
        if hasattr(model, 'set_pretraining_mode'):
            model.set_pretraining_mode(True)
            # Log updated parameter counts after freezing
            new_param_counts = count_parameters(model)
            logger.info(f"After freezing LLM - Parameters: {new_param_counts}")
        else:
            logger.warning("Model does not have set_pretraining_mode method!")
    
    # Create optimizer
    learning_rate = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    
    # Different learning rates for pretrained vs new parameters
    param_groups = model.get_trainable_params()
    
    # Filter out empty param groups to avoid optimizer errors
    valid_param_groups = []
    if param_groups[0]['params']:  # LLM params (may be empty in pretraining mode)
        valid_param_groups.append({'params': param_groups[0]['params'], 'lr': learning_rate * 0.1})
    if param_groups[1]['params']:  # Projection params (always present)
        valid_param_groups.append({'params': param_groups[1]['params'], 'lr': learning_rate})
    
    optimizer = AdamW(valid_param_groups, weight_decay=weight_decay)
    
    # Create scheduler
    num_epochs = config['training']['epochs']
    warmup_ratio = config['training'].get('warmup_ratio', 0.1)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[pg['lr'] for pg in valid_param_groups],
        total_steps=total_steps,
        pct_start=warmup_ratio,
        anneal_strategy='cos'
    )
    
    # Create gradient scaler for FP16
    scaler = GradScaler('cuda') if config['training'].get('fp16', True) else None
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training'].get('early_stopping_patience', 10),
        mode='max'
    )
    
    # Resume from checkpoint if specified (via args or config)
    start_epoch = 0
    best_acc = 0.0
    
    resume_path = args.resume or config['training'].get('resume_from')
    if resume_path:
        import os
        if os.path.exists(resume_path):
            logger.info(f"Resuming from checkpoint: {resume_path}")
            # Load checkpoint
            checkpoint = torch.load(resume_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['metrics'].get('overall_accuracy', 0.0)
            logger.info(f"Resumed from epoch {checkpoint['epoch']}, best accuracy: {best_acc:.4f}")
            
            # If resuming from Stage 2, set up Stage 2 mode and optimizer
            if start_epoch > pretraining_epochs and is_generative:
                logger.info(f"Checkpoint is from Stage 2, setting up Stage 2 mode...")
                if hasattr(model, 'set_pretraining_mode'):
                    model.set_pretraining_mode(False)
                
                # Recreate optimizer with Stage 2 parameters
                param_groups = model.get_trainable_params()
                valid_param_groups = []
                if param_groups[0]['params']:
                    valid_param_groups.append({'params': param_groups[0]['params'], 'lr': learning_rate * 0.1})
                if param_groups[1]['params']:
                    valid_param_groups.append({'params': param_groups[1]['params'], 'lr': learning_rate})
                
                optimizer = AdamW(valid_param_groups, weight_decay=weight_decay)
                
                # Recreate scheduler for remaining epochs
                remaining_steps = len(train_loader) * (num_epochs - start_epoch)
                scheduler = OneCycleLR(
                    optimizer,
                    max_lr=[pg['lr'] for pg in valid_param_groups],
                    total_steps=remaining_steps,
                    pct_start=0.1,
                    anneal_strategy='cos'
                )
                logger.info(f"Stage 2 optimizer and scheduler recreated for epochs {start_epoch}-{num_epochs}")
        else:
            logger.warning(f"Checkpoint not found: {resume_path}, starting from scratch")
    
    # Training loop
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, num_epochs):
        # Check for stage transition (pretraining -> fine-tuning)
        if pretraining_epochs > 0 and epoch == pretraining_epochs and is_generative:
            logger.info(f"\n*** Stage 2: Unfreezing LLM decoder ***")
            if hasattr(model, 'set_pretraining_mode'):
                model.set_pretraining_mode(False)
            
            # Recreate optimizer with LLM parameters now included
            param_groups = model.get_trainable_params()
            valid_param_groups = []
            if param_groups[0]['params']:  # LLM params (now unfrozen)
                valid_param_groups.append({'params': param_groups[0]['params'], 'lr': learning_rate * 0.1})
            if param_groups[1]['params']:  # Projection params
                valid_param_groups.append({'params': param_groups[1]['params'], 'lr': learning_rate})
            
            optimizer = AdamW(valid_param_groups, weight_decay=weight_decay)
            
            # Recreate scheduler for remaining epochs
            remaining_steps = len(train_loader) * (num_epochs - epoch)
            scheduler = OneCycleLR(
                optimizer,
                max_lr=[pg['lr'] for pg in valid_param_groups],
                total_steps=remaining_steps,
                pct_start=0.1,
                anneal_strategy='cos'
            )
            logger.info(f"Optimizer recreated with {len(valid_param_groups)} parameter groups")
        
        stage_label = f"[Stage 1/Projection]" if epoch < pretraining_epochs and pretraining_epochs > 0 else "[Stage 2/Full]" if pretraining_epochs > 0 else ""
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{num_epochs} {stage_label}")
        logger.info(f"{'='*50}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch + 1, config, logger
        )
        
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        
        # Validate
        val_metrics = validate(model, val_loader, device, config, logger)
        
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
