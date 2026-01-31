"""
PubMedBERT Text Encoder
Domain-specific BERT trained on PubMed abstracts for medical text understanding.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
from transformers import AutoModel, AutoTokenizer


class PubMedBERTEncoder(nn.Module):
    """
    Text encoder using PubMedBERT.
    Trained on 14M+ PubMed abstracts with 3.2B words.
    
    Output: 768-dim text embeddings
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        freeze_encoder: bool = False,
        use_gradient_checkpointing: bool = True,
        max_length: int = 64,
        pooling_strategy: str = "cls"  # "cls", "mean", or "max"
    ):
        super().__init__()
        
        self.model_name = model_name
        self.hidden_dim = 768
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
        
        if use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
    
    def tokenize(
        self,
        texts: Union[str, List[str]],
        device: Optional[torch.device] = None
    ) -> dict:
        """
        Tokenize input texts.
        
        Args:
            texts: Single string or list of strings
            device: Target device for tensors
            
        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        if device is not None:
            encoded = {k: v.to(device) for k, v in encoded.items()}
        
        return encoded
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        texts: Optional[Union[str, List[str]]] = None,
        return_sequence: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through text encoder.
        
        Args:
            input_ids: Token IDs [B, L]
            attention_mask: Attention mask [B, L]
            texts: Raw text strings (tokenized internally if input_ids not provided)
            return_sequence: Whether to return full sequence
            
        Returns:
            pooled_features: [B, 768] Pooled text features
            sequence_output: [B, L, 768] Full sequence (optional)
        """
        # Tokenize if raw texts provided
        if input_ids is None and texts is not None:
            device = next(self.model.parameters()).device
            encoded = self.tokenize(texts, device)
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
        
        # Forward through BERT
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state  # [B, L, 768]
        
        # Apply pooling strategy
        if self.pooling_strategy == "cls":
            pooled_features = sequence_output[:, 0, :]  # [B, 768]
        elif self.pooling_strategy == "mean":
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
            sum_embeddings = torch.sum(sequence_output * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled_features = sum_embeddings / sum_mask  # [B, 768]
        elif self.pooling_strategy == "max":
            # Max pooling
            pooled_features = torch.max(sequence_output, dim=1)[0]  # [B, 768]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
        
        if return_sequence:
            return pooled_features, sequence_output
        
        return pooled_features, None
    
    def encode_answers(
        self,
        answers: List[str],
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Encode a list of answer strings.
        
        Args:
            answers: List of answer strings
            device: Target device
            
        Returns:
            answer_embeddings: [N, 768] Answer embeddings
        """
        if device is None:
            device = next(self.model.parameters()).device
        
        # Ensure model is on device
        self.model = self.model.to(device)
        
        encoded = self.tokenize(answers, device)
        
        with torch.no_grad():
            pooled, _ = self.forward(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask']
            )
        
        return pooled
    
    @property
    def output_dim(self) -> int:
        """Return the output dimension."""
        return self.hidden_dim
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return self.tokenizer.vocab_size


class AnswerEmbeddingCache:
    """
    Cache for answer embeddings to avoid recomputing them.
    """
    
    def __init__(self, encoder: PubMedBERTEncoder):
        self.encoder = encoder
        self.cache = {}
        self.answer_to_idx = {}
        self.idx_to_answer = {}
    
    def build_cache(self, answers: List[str], device: torch.device):
        """
        Build cache of answer embeddings.
        
        Args:
            answers: List of unique answers
            device: Target device
        """
        # Create mappings
        for idx, answer in enumerate(answers):
            self.answer_to_idx[answer] = idx
            self.idx_to_answer[idx] = answer
        
        # Encode all answers in batches
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(answers), batch_size):
            batch_answers = answers[i:i + batch_size]
            embeddings = self.encoder.encode_answers(batch_answers, device)
            all_embeddings.append(embeddings)
        
        self.cache['embeddings'] = torch.cat(all_embeddings, dim=0)  # [N, 768]
    
    def get_embeddings(self, answer_indices: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for answer indices.
        
        Args:
            answer_indices: [B] or [B, K] tensor of answer indices
            
        Returns:
            embeddings: [B, 768] or [B, K, 768] tensor of embeddings
        """
        return self.cache['embeddings'][answer_indices]
    
    @property
    def num_answers(self) -> int:
        """Return number of cached answers."""
        return len(self.answer_to_idx)
