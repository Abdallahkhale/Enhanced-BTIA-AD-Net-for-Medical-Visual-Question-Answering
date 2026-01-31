"""
Bi-Text-Image Attention (BTIA) Fusion Module
Multi-head self-attention and guided attention for cross-modal fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            query: [B, L_q, D] Query tensor
            key: [B, L_k, D] Key tensor (default: same as query)
            value: [B, L_v, D] Value tensor (default: same as key)
            attention_mask: [B, L_q, L_k] Attention mask
            
        Returns:
            output: [B, L_q, D] Output tensor
            attention_weights: [B, H, L_q, L_k] Attention weights
        """
        if key is None:
            key = query
        if value is None:
            value = key
        
        batch_size = query.size(0)
        
        # Project and reshape
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L_q, L_k]
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, v)  # [B, H, L_q, D_h]
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        output = self.out_proj(output)
        
        return output, attention_weights


class GuidedAttention(nn.Module):
    """
    Guided attention for cross-modal interaction.
    Query from one modality attends to key/value from another.
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with residual connection.
        
        Args:
            query: [B, L_q, D] or [B, D] Query tensor
            context: [B, L_c, D] or [B, D] Context tensor
            attention_mask: Optional attention mask
            
        Returns:
            output: [B, L_q, D] or [B, D] Output tensor
            attention_weights: Attention weights
        """
        # Handle 2D inputs
        squeeze_output = False
        if query.dim() == 2:
            query = query.unsqueeze(1)
            squeeze_output = True
        if context.dim() == 2:
            context = context.unsqueeze(1)
        
        # Cross-attention
        attended, weights = self.attention(query, context, context, attention_mask)
        
        # Residual connection and layer norm
        output = self.layer_norm(query + self.dropout(attended))
        
        if squeeze_output:
            output = output.squeeze(1)
        
        return output, weights


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism for combining multiple features.
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_inputs: int = 3
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_inputs = num_inputs
        
        # Gate generators for each input
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * num_inputs, hidden_dim),
                nn.Sigmoid()
            )
            for _ in range(num_inputs)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * num_inputs, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
    
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            *inputs: Variable number of [B, D] tensors
            
        Returns:
            output: [B, D] Gated fusion output
        """
        assert len(inputs) == self.num_inputs, f"Expected {self.num_inputs} inputs, got {len(inputs)}"
        
        # Concatenate all inputs
        concat = torch.cat(inputs, dim=-1)  # [B, D*num_inputs]
        
        # Generate gates
        gated_inputs = []
        for i, (gate, inp) in enumerate(zip(self.gates, inputs)):
            g = gate(concat)  # [B, D]
            gated_inputs.append(g * inp)
        
        # Combine gated inputs
        combined = torch.cat(gated_inputs, dim=-1)  # [B, D*num_inputs]
        output = self.output_proj(combined)  # [B, D]
        
        return output


class BiTextImageAttention(nn.Module):
    """
    Bi-Text-Image Attention (BTIA) fusion module.
    
    Integrates:
    - Image features
    - Question features
    - Answer candidate features
    
    Using multi-head self-attention and guided cross-attention.
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_gated_fusion: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_gated_fusion = use_gated_fusion
        
        # Input projections to ensure consistent dimensions
        self.image_proj = nn.Linear(hidden_dim, hidden_dim)
        self.question_proj = nn.Linear(hidden_dim, hidden_dim)
        self.answer_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Cross-modal attention layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                # Image -> Question attention
                'img_to_q': GuidedAttention(hidden_dim, num_heads, dropout),
                # Question -> Answer attention
                'q_to_ans': GuidedAttention(hidden_dim, num_heads, dropout),
                # Image -> Answer attention
                'img_to_ans': GuidedAttention(hidden_dim, num_heads, dropout),
                # Self-attention for each modality
                'img_self': MultiHeadSelfAttention(hidden_dim, num_heads, dropout),
                'q_self': MultiHeadSelfAttention(hidden_dim, num_heads, dropout),
                # Feed-forward networks
                'img_ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout)
                ),
                'q_ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout)
                ),
                # Layer norms
                'img_ln': nn.LayerNorm(hidden_dim),
                'q_ln': nn.LayerNorm(hidden_dim)
            })
            self.layers.append(layer)
        
        # Final fusion
        if use_gated_fusion:
            self.fusion = GatedFusion(hidden_dim, num_inputs=3)
        else:
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        image_features: torch.Tensor,
        question_features: torch.Tensor,
        answer_features: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass through BTIA fusion.
        
        Args:
            image_features: [B, D] Image features from BiomedCLIP
            question_features: [B, D] Question features from PubMedBERT
            answer_features: [B, D] Answer context from AD Network
            return_attention: Whether to return attention weights
            
        Returns:
            fused_features: [B, D] Fused multimodal features
            attention_dict: Dict of attention weights (optional)
        """
        # Project inputs
        img = self.image_proj(image_features)  # [B, D]
        q = self.question_proj(question_features)  # [B, D]
        ans = self.answer_proj(answer_features)  # [B, D]
        
        attention_dict = {} if return_attention else None
        
        # Apply attention layers
        for layer_idx, layer in enumerate(self.layers):
            # Image -> Question cross-attention
            img_q, attn_img_q = layer['img_to_q'](img, q)
            
            # Question -> Answer cross-attention
            q_ans, attn_q_ans = layer['q_to_ans'](q, ans)
            
            # Image -> Answer cross-attention
            img_ans, attn_img_ans = layer['img_to_ans'](img, ans)
            
            # Self-attention (need to add sequence dimension)
            img_self, _ = layer['img_self'](img.unsqueeze(1))
            img_self = img_self.squeeze(1)
            
            q_self, _ = layer['q_self'](q.unsqueeze(1))
            q_self = q_self.squeeze(1)
            
            # Combine with residuals
            img = layer['img_ln'](img + img_q + img_ans + img_self)
            q = layer['q_ln'](q + q_ans + q_self)
            
            # Feed-forward
            img = img + layer['img_ffn'](img)
            q = q + layer['q_ffn'](q)
            
            if return_attention:
                attention_dict[f'layer_{layer_idx}'] = {
                    'img_to_q': attn_img_q,
                    'q_to_ans': attn_q_ans,
                    'img_to_ans': attn_img_ans
                }
        
        # Final fusion
        if self.use_gated_fusion:
            fused = self.fusion(img, q, ans)
        else:
            fused = self.fusion(torch.cat([img, q, ans], dim=-1))
        
        # Output projection
        output = self.output_proj(fused)
        
        return output, attention_dict
    
    @property
    def output_dim(self) -> int:
        """Return output dimension."""
        return self.hidden_dim
