"""
VQA-RAD and SLAKE Dataset Loader
Combined dataset handler with preprocessing for BiomedCLIP and PubMedBERT.
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from transformers import AutoTokenizer
import open_clip

from .download import download_vqa_rad, download_slake, download_pathvqa


class MedVQADataset(Dataset):
    """
    Combined Medical VQA Dataset supporting VQA-RAD, SLAKE, and PathVQA.
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        split: str = "train",
        use_slake: bool = True,
        use_pathvqa: bool = False,
        image_size: int = 224,
        max_question_length: int = 64,
        tokenizer_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        transform=None,
        auto_download: bool = True,
        oversample_factor: int = 1,
        generative_mode: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Base data directory
            split: Dataset split ('train' or 'test')
            use_slake: Whether to include SLAKE dataset
            use_pathvqa: Whether to include PathVQA dataset (32K+ extra samples!)
            image_size: Target image size
            max_question_length: Maximum question token length
            tokenizer_name: Tokenizer to use
            transform: Image transform (if None, uses BiomedCLIP default)
            auto_download: Auto-download datasets if not present
            generative_mode: Enable generative training (tokenized answers)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.use_slake = use_slake
        self.use_pathvqa = use_pathvqa
        self.image_size = image_size
        self.image_size = image_size
        self.max_question_length = max_question_length
        self.oversample_factor = oversample_factor
        self.generative_mode = generative_mode
        
        # Initialize tokenizer
        # For generative mode, set padding
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize image transform
        if transform is None:
            _, self.transform, _ = open_clip.create_model_and_transforms(
                'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
            )
        else:
            self.transform = transform
        
        # Auto-download if needed
        if auto_download:
            self._ensure_datasets()
        
        # Load data
        self.samples = []
        self._load_vqa_rad()
        
        if use_slake:
            self._load_slake()
        
        if use_pathvqa:
            self._load_pathvqa()
        
        # Build answer vocabulary
        self.answer_to_idx, self.idx_to_answer = self._build_answer_vocab()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        print(f"Answer vocabulary size: {len(self.answer_to_idx)}")
    
    def _ensure_datasets(self):
        """Ensure datasets are downloaded."""
        vqa_rad_dir = self.data_dir / "vqa_rad"
        slake_dir = self.data_dir / "slake"
        pathvqa_dir = self.data_dir / "pathvqa"
        
        if not vqa_rad_dir.exists():
            download_vqa_rad(str(vqa_rad_dir))
        
        if self.use_slake and not slake_dir.exists():
            download_slake(str(slake_dir))
        
        if self.use_pathvqa and not pathvqa_dir.exists():
            download_pathvqa(str(pathvqa_dir))
    
    def _load_vqa_rad(self):
        """Load VQA-RAD dataset."""
        vqa_rad_dir = self.data_dir / "vqa_rad"
        
        # Try to load from JSON
        json_path = vqa_rad_dir / f"{self.split}.json"
        
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            for item in data:
                # Oversample VQA-RAD if requested (to balance with PathVQA)
                # Only apply to training set
                count = self.oversample_factor if self.split == 'train' else 1
                
                for _ in range(count):
                    self.samples.append({
                        'image_path': str(vqa_rad_dir / item['image_path']),
                        'question': item['question'],
                        'answer': str(item['answer']).lower().strip(),
                        'question_type': item.get('question_type', 'unknown'),
                        'answer_type': item.get('answer_type', 'OPEN'),
                        'source': 'vqa_rad'
                    })
        else:
            print(f"Warning: VQA-RAD {self.split}.json not found at {json_path}")
            print("Please run 'python data/download.py' first")
    
    def _load_slake(self):
        """Load SLAKE dataset."""
        slake_dir = self.data_dir / "slake"
        
        # For SLAKE, we use English samples only and map to train/test
        json_path = slake_dir / f"{self.split}.json"
        
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            for item in data:
                # Handle both 'image_path' and 'image' fields
                img_field = item.get('image_path', item.get('image', ''))
                # Images are stored in 'imgs' subdirectory
                image_path = slake_dir / "imgs" / img_field
                
                # Skip if image doesn't exist
                if not image_path.exists():
                    continue
                    
                self.samples.append({
                    'image_path': str(image_path),
                    'question': item['question'],
                    'answer': str(item['answer']).lower().strip(),
                    'question_type': item.get('question_type', item.get('content_type', 'unknown')),
                    'answer_type': item.get('answer_type', 'OPEN'),
                    'source': 'slake'
                })
        else:
            print(f"Warning: SLAKE {self.split}.json not found at {json_path}")
    
    def _load_pathvqa(self):
        """Load PathVQA dataset (32,799 QA pairs!)."""
        pathvqa_dir = self.data_dir / "pathvqa"
        
        json_path = pathvqa_dir / f"{self.split}.json"
        
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            for item in data:
                self.samples.append({
                    'image_path': str(pathvqa_dir / item['image_path']),
                    'question': item['question'],
                    'answer': str(item['answer']).lower().strip(),
                    'question_type': item.get('question_type', 'pathology'),
                    'answer_type': item.get('answer_type', 'OPEN'),
                    'source': 'pathvqa'
                })
            print(f"Loaded {len(data)} PathVQA samples")
        else:
            print(f"Warning: PathVQA {self.split}.json not found at {json_path}")
    
    def _build_answer_vocab(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build answer vocabulary from training data."""
        # Count answer frequencies
        answer_counts = Counter(s['answer'] for s in self.samples)
        
        # Include ALL answers (no frequency filtering)
        # This is crucial for open-ended questions which often have unique answers
        answers = [ans for ans, count in answer_counts.most_common()]
        
        # Add special tokens
        answers = ['<UNK>'] + answers
        
        answer_to_idx = {ans: idx for idx, ans in enumerate(answers)}
        idx_to_answer = {idx: ans for ans, idx in answer_to_idx.items()}
        
        # Compute class weights (inverse frequency)
        self._answer_counts = answer_counts
        
        return answer_to_idx, idx_to_answer
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights inversely proportional to frequency.
        
        Returns:
            Tensor of shape [num_answers] with weights for each answer class
        """
        import numpy as np
        
        num_answers = len(self.answer_to_idx)
        weights = np.ones(num_answers)
        
        for ans, idx in self.answer_to_idx.items():
            count = self._answer_counts.get(ans, 1)
            # Strict Inverse frequency to heavily penalize frequent classes (yes/no)
            weights[idx] = 1.0 / count
        
        # Normalize so weights sum to num_classes
        weights = weights * num_answers / weights.sum()
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample."""
        sample = self.samples[idx]
        
        # Load and transform image
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {sample['image_path']}: {e}")
            # Return a black image as fallback
            image = torch.zeros(3, self.image_size, self.image_size)
        
        # Generative Mode Tokenization
        if self.generative_mode:
            questions = sample['question'] # Is this singular or batch?
            # It's definitely singular in getitem.
            
            # Determine if this is a closed (yes/no) or open question
            is_closed = sample.get('answer_type', 'OPEN').upper() in ['CLOSED', 'YES/NO'] or \
                       sample['answer'].lower() in ['yes', 'no']
            
            # Use different prompts for open vs closed questions
            # This teaches the model when to give yes/no vs descriptive answers
            if is_closed:
                # Closed-ended: expect yes/no answer
                prompt = f"Question: {sample['question']} Answer yes or no:"
            else:
                # Open-ended: expect descriptive answer
                # "Provide a detailed answer" triggers BioGPT's paper generation mode
                # "Answer:" is more constrained and effective for VQA
                prompt = f"Question: {sample['question']} Answer:"
            
            answer_text = f" {sample['answer']}"
            
            # 1. Tokenize Prompt
            prompt_enc = self.tokenizer(
                prompt,
                truncation=True, 
                max_length=self.max_question_length,
                add_special_tokens=False, # We'll manage special tokens manually if needed
                return_tensors='pt'
            )
            prompt_ids = prompt_enc['input_ids'].squeeze(0)
            
            # 2. Tokenize Answer + EOS
            answer_enc = self.tokenizer(
                answer_text + self.tokenizer.eos_token,
                truncation=True,
                max_length=32, # Max answer length
                add_special_tokens=False,
                return_tensors='pt'
            )
            answer_ids = answer_enc['input_ids'].squeeze(0)
            
            # 3. Concatenate
            input_ids = torch.cat([prompt_ids, answer_ids], dim=0)
            attention_mask = torch.ones_like(input_ids)
            
            # 4. Create Labels
            # Mask prompt with -100
            labels = input_ids.clone()
            labels[:len(prompt_ids)] = -100
            
            # 5. Pad to max_length
            max_len = self.max_question_length + 32
            if len(input_ids) < max_len:
                pad_len = max_len - len(input_ids)
                pad_ids = torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)
                input_ids = torch.cat([input_ids, pad_ids], dim=0)
                attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)], dim=0)
                labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)], dim=0)
            else:
                # Truncate if somehow too long (unlikely given chunks)
                input_ids = input_ids[:max_len]
                attention_mask = attention_mask[:max_len]
                labels = labels[:max_len]
            
            # If 'answer_type' is closed, mark it
            is_closed = sample.get('answer_type', 'OPEN').upper() in ['CLOSED', 'YES/NO'] or \
                       sample['answer'].lower() in ['yes', 'no']
            
            return {
                'images': image,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels, # For CausalLM loss
                'question': sample['question'],
                'answer_text': sample['answer'],
                'is_closed': torch.tensor(1 if is_closed else 0, dtype=torch.long)
            }

        # Discriminative Mode (Legacy)
        encoded = self.tokenizer(
            sample['question'],
            padding='max_length',
            truncation=True,
            max_length=self.max_question_length,
            return_tensors='pt'
        )
        
        # Get answer index
        answer = sample['answer']
        answer_idx = self.answer_to_idx.get(answer, 0)  # 0 is <UNK>
        
        # Determine if closed-ended (yes/no questions)
        is_closed = sample['answer_type'].upper() in ['CLOSED', 'YES/NO'] or \
                   answer.lower() in ['yes', 'no']
        
        return {
            'image': image,
            'question': sample['question'],
            'question_ids': encoded['input_ids'].squeeze(0),
            'question_mask': encoded['attention_mask'].squeeze(0),
            'answer': answer,
            'answer_idx': torch.tensor(answer_idx, dtype=torch.long),
            'answer_type': sample['answer_type'],
            'is_closed': torch.tensor(is_closed, dtype=torch.bool),
            'source': sample['source']
        }
    
    def get_answer_list(self) -> List[str]:
        """Get list of all answers in vocabulary order."""
        return [self.idx_to_answer[i] for i in range(len(self.idx_to_answer))]
    
    @property
    def num_answers(self) -> int:
        """Return number of unique answers."""
        return len(self.answer_to_idx)


def create_dataloaders(
    data_dir: str = "data",
    batch_size: int = 16,
    num_workers: int = 4,
    use_slake: bool = True,
    use_pathvqa: bool = False,
    image_size: int = 224,
    max_question_length: int = 64,
    generative_mode: bool = False,
    tokenizer_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
) -> Tuple[DataLoader, DataLoader, MedVQADataset]:
    """
    Create train and test dataloaders.
    
    Args:
        data_dir: Base data directory
        batch_size: Batch size
        num_workers: Number of data loading workers
        use_slake: Whether to include SLAKE dataset
        use_pathvqa: Whether to include PathVQA dataset (32K+ samples!)
        image_size: Target image size
        max_question_length: Maximum question token length
        generative_mode: Enable generative mode
        tokenizer_name: Tokenizer name
        
    Returns:
        train_loader, test_loader, train_dataset
    """
    # Create datasets
    train_dataset = MedVQADataset(
        data_dir=data_dir,
        split='train',
        use_slake=use_slake,
        use_pathvqa=use_pathvqa,
        image_size=image_size,
        max_question_length=max_question_length,
        oversample_factor=12 if use_pathvqa else 1,
        generative_mode=generative_mode,
        tokenizer_name=tokenizer_name
    )
    
    test_dataset = MedVQADataset(
        data_dir=data_dir,
        split='test',
        use_slake=False,  # Test only on VQA-RAD
        image_size=image_size,
        max_question_length=max_question_length,
        generative_mode=generative_mode,
        tokenizer_name=tokenizer_name
    )
    
    # CRITICAL: Merge test vocabulary into training vocabulary (Discriminative Only)
    # Generative doesn't care about answer_to_idx, but good to keep consistency
    if not generative_mode:
        all_answers = set(train_dataset.answer_to_idx.keys())
        for answer in test_dataset.answer_to_idx.keys():
            if answer not in all_answers:
                all_answers.add(answer)
        
        # Rebuild vocabulary with all answers (sorted for consistency)
        sorted_answers = ['<UNK>'] + sorted([a for a in all_answers if a != '<UNK>'])
        unified_answer_to_idx = {ans: idx for idx, ans in enumerate(sorted_answers)}
        unified_idx_to_answer = {idx: ans for ans, idx in unified_answer_to_idx.items()}
        
        # Apply unified vocabulary to both datasets
        train_dataset.answer_to_idx = unified_answer_to_idx
        train_dataset.idx_to_answer = unified_idx_to_answer
        test_dataset.answer_to_idx = unified_answer_to_idx
        test_dataset.idx_to_answer = unified_idx_to_answer
        
        print(f"Unified vocabulary size: {len(unified_answer_to_idx)}")
    
    # Create collator
    collator = VQACollator()
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collator
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator
    )
    
    return train_loader, test_loader, train_dataset


class VQACollator:
    """
    Custom collator for VQA batches.
    """
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples."""
        # Fix: access 'image' or 'images' consistently
        # dataset.__getitem__ returns 'images' if we fixed it, or 'image'?
        # Previous view showed 'images': image in generative mode block, but 'image': image in discrim mode.
        # Let's check `__getitem__` return keys again. 
        # Line 309 in Step 504: 'images': image
        # Line 267 in Step 458: 'image': image (Discrim)
        # So we handle both.
        
        images = torch.stack([s.get('images', s.get('image')) for s in batch])
        questions = [s['question'] for s in batch]
        
        result = {
            'images': images,
            'questions': questions
        }
        
        if 'input_ids' in batch[0]:
            # Generative
            result['input_ids'] = torch.stack([s['input_ids'] for s in batch])
            result['attention_mask'] = torch.stack([s['attention_mask'] for s in batch])
            result['labels'] = torch.stack([s['labels'] for s in batch])
        else:
            # Discriminative
            result['question_ids'] = torch.stack([s['question_ids'] for s in batch])
            result['question_mask'] = torch.stack([s['question_mask'] for s in batch])
            result['answer_idx'] = torch.stack([s['answer_idx'] for s in batch])
            
        # Common
        result['is_closed'] = torch.stack([s['is_closed'] for s in batch])
        
        return result
