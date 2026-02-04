# Custom batch sampler to group samples by question type
import torch
from torch.utils.data import Sampler
from typing import Iterator, List
from collections import defaultdict

class QuestionTypeBatchSampler(Sampler):
    """
    Custom sampler that creates batches containing only one question type.
    This avoids tensor dimension mismatches when collating.
    """
    def __init__(self, dataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Try to load cached indices
        import pickle
        import os
        cache_dir = "data/.cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create cache key based on dataset size and split
        split_name = getattr(dataset, 'split', 'unknown')
        cache_key = f"qtype_indices_{split_name}_{len(dataset)}.pkl"
        cache_path = os.path.join(cache_dir, cache_key)
        
        if os.path.exists(cache_path):
            print(f"Loading cached question type indices from {cache_path}")
            with open(cache_path, 'rb') as f:
                self.type_indices = pickle.load(f)
        else:
            # Group indices by question type
            self.type_indices = defaultdict(list)
            
            print("Grouping dataset by question type...")
            for idx in range(len(dataset)):
                # We need to peek at the sample to get the type
                # This is a bit expensive but only done once
                sample = dataset[idx]
                q_type = sample.get('question_type', 'rare_open')
                self.type_indices[q_type].append(idx)
            
            # Save to cache
            print(f"Saving question type indices to cache: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(dict(self.type_indices), f)
        
        print(f"Question type distribution:")
        for q_type, indices in self.type_indices.items():
            print(f"  {q_type}: {len(indices)} samples")
    
    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle indices within each type
        for q_type in self.type_indices:
            indices = self.type_indices[q_type].copy()
            torch.manual_seed(torch.initial_seed())
            perm = torch.randperm(len(indices)).tolist()
            self.type_indices[q_type] = [indices[i] for i in perm]
        
        # Create batches from each type
        batches = []
        for q_type, indices in self.type_indices.items():
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) > 0:  # Allow smaller last batch
                    batches.append(batch)
        
        # Shuffle batches
        torch.manual_seed(torch.initial_seed())
        perm = torch.randperm(len(batches)).tolist()
        batches = [batches[i] for i in perm]
        
        # Yield batches
        for batch in batches:
            yield batch
    
    def __len__(self) -> int:
        # Total number of batches
        total = 0
        for indices in self.type_indices.values():
            total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total
