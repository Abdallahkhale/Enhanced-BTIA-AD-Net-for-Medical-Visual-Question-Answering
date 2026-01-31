"""
Answer Distillation Network
Transforms open-ended questions into multiple-choice style by predicting Top-K candidates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


class AnswerDistillationNetwork(nn.Module):
    """
    Answer Distillation (AD) Network.
    
    Takes visual and question features, predicts distribution over answer vocabulary,
    and selects Top-K candidate answers with their embeddings.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 768,
        num_answers: int = 1000,
        top_k: int = 5,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_answers = num_answers
        self.top_k = top_k
        
        # Feature fusion for initial prediction
        self.fusion_layer = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Initial answer prediction head
        self.answer_classifier = nn.Linear(hidden_dim, num_answers)
        
        # Answer embedding projection
        self.answer_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Probability-weighted context generator
        self.context_generator = nn.Sequential(
            nn.Linear(hidden_dim + top_k, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        visual_features: torch.Tensor,
        question_features: torch.Tensor,
        answer_embeddings: Optional[torch.Tensor] = None,
        return_logits: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through Answer Distillation Network.
        
        Args:
            visual_features: [B, 768] Visual features from BiomedCLIP
            question_features: [B, 768] Question features from PubMedBERT
            answer_embeddings: [N, 768] Precomputed answer embeddings (optional)
            return_logits: Whether to return raw logits
            
        Returns:
            ad_logits: [B, N] Answer distribution logits
            top_k_indices: [B, K] Top-K answer indices
            top_k_probs: [B, K] Top-K answer probabilities
            answer_context: [B, 768] Probability-weighted answer context
        """
        batch_size = visual_features.size(0)
        
        # Concatenate visual and question features
        combined = torch.cat([visual_features, question_features], dim=-1)  # [B, 1536]
        
        # Fuse features
        fused = self.fusion_layer(combined)  # [B, 768]
        
        # Initial answer prediction
        ad_logits = self.answer_classifier(fused)  # [B, N]
        ad_probs = F.softmax(ad_logits, dim=-1)  # [B, N]
        
        # Get Top-K candidates
        top_k_probs, top_k_indices = torch.topk(ad_probs, self.top_k, dim=-1)  # [B, K]
        
        # Generate answer context
        if answer_embeddings is not None:
            # Get embeddings for Top-K answers
            # answer_embeddings: [N, 768]
            # top_k_indices: [B, K]
            top_k_embeddings = answer_embeddings[top_k_indices]  # [B, K, 768]
            
            # Project answer embeddings
            top_k_embeddings = self.answer_projection(top_k_embeddings)  # [B, K, 768]
            
            # Probability-weighted sum
            weighted_embeddings = (top_k_probs.unsqueeze(-1) * top_k_embeddings).sum(dim=1)  # [B, 768]
            
            # Combine with probabilities for richer context
            context_input = torch.cat([weighted_embeddings, top_k_probs], dim=-1)  # [B, 768+K]
            answer_context = self.context_generator(context_input)  # [B, 768]
        else:
            # Without precomputed embeddings, use fused features
            answer_context = fused
        
        if return_logits:
            return ad_logits, top_k_indices, top_k_probs, answer_context
        else:
            return ad_probs, top_k_indices, top_k_probs, answer_context
    
    def get_answer_distribution(
        self,
        visual_features: torch.Tensor,
        question_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Get full answer distribution without Top-K selection.
        
        Args:
            visual_features: [B, 768] Visual features
            question_features: [B, 768] Question features
            
        Returns:
            ad_logits: [B, N] Answer distribution logits
        """
        combined = torch.cat([visual_features, question_features], dim=-1)
        fused = self.fusion_layer(combined)
        return self.answer_classifier(fused)


class AdaptiveTopK(nn.Module):
    """
    Adaptive Top-K selection that adjusts K based on confidence.
    """
    
    def __init__(
        self,
        max_k: int = 10,
        min_k: int = 3,
        confidence_threshold: float = 0.9
    ):
        super().__init__()
        
        self.max_k = max_k
        self.min_k = min_k
        self.confidence_threshold = confidence_threshold
    
    def forward(
        self,
        probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Adaptively select Top-K candidates.
        
        Args:
            probs: [B, N] Answer probabilities
            
        Returns:
            top_k_probs: [B, max_k] Probabilities (padded)
            top_k_indices: [B, max_k] Indices (padded with -1)
            effective_k: [B] Actual K used for each sample
        """
        batch_size = probs.size(0)
        device = probs.device
        
        # Get all Top-max_k candidates
        full_top_probs, full_top_indices = torch.topk(probs, self.max_k, dim=-1)
        
        # Calculate cumulative probability
        cumsum = torch.cumsum(full_top_probs, dim=-1)
        
        # Find where cumulative probability exceeds threshold
        exceeds_threshold = cumsum >= self.confidence_threshold
        
        # Calculate effective K (at least min_k)
        effective_k = torch.sum(~exceeds_threshold, dim=-1) + 1
        effective_k = torch.clamp(effective_k, min=self.min_k, max=self.max_k)
        
        return full_top_probs, full_top_indices, effective_k
