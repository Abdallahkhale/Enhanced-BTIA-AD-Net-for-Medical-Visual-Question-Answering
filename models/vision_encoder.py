"""
BiomedCLIP Vision Encoder
Wrapper for Microsoft's BiomedCLIP pretrained on PMC-15M medical images.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers import AutoModel, AutoProcessor
import open_clip


class BiomedCLIPVisionEncoder(nn.Module):
    """
    Vision encoder using BiomedCLIP's ViT-B/16.
    Pretrained on 15M medical figure-caption pairs from PubMed Central.
    
    Output: 768-dim visual embeddings + optional patch tokens
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        freeze_encoder: bool = False,
        use_gradient_checkpointing: bool = True,
        extract_layers: Optional[list] = None
    ):
        super().__init__()
        
        self.model_name = model_name
        self.hidden_dim = 768
        self.extract_layers = extract_layers or [-1]  # Last layer by default
        
        # Load BiomedCLIP model
        # BiomedCLIP uses open_clip format
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        
        # Extract only the vision encoder
        self.vision_encoder = self.model.visual
        
        # Safely delete text components to save memory (if they exist)
        text_attrs = ['text', 'token_embedding', 'positional_embedding', 'ln_final', 'text_projection']
        for attr in text_attrs:
            if hasattr(self.model, attr):
                try:
                    delattr(self.model, attr)
                except (AttributeError, RuntimeError):
                    pass  # Attribute may not be deletable
        
        if freeze_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        
        if use_gradient_checkpointing:
            self._enable_gradient_checkpointing()
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.vision_encoder, 'transformer'):
            # For ViT architecture
            if hasattr(self.vision_encoder.transformer, 'resblocks'):
                for block in self.vision_encoder.transformer.resblocks:
                    block.use_checkpoint = True
    
    def forward(
        self,
        images: torch.Tensor,
        return_patch_tokens: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through vision encoder.
        
        Args:
            images: Input images [B, 3, 224, 224]
            return_patch_tokens: Whether to return patch tokens
            
        Returns:
            cls_features: [B, 768] CLS token features
            patch_tokens: [B, 196, 768] Patch token features (optional)
        """
        # Get intermediate features if extracting from multiple layers
        if hasattr(self.vision_encoder, 'trunk'):
            # For timm-style ViT
            features = self.vision_encoder.trunk(images)
        else:
            # Standard forward
            features = self.vision_encoder(images)
        
        # Handle different output formats
        if isinstance(features, tuple):
            cls_features = features[0]
            patch_tokens = features[1] if len(features) > 1 else None
        elif len(features.shape) == 3:
            # [B, N, D] format - first token is CLS
            cls_features = features[:, 0, :]
            patch_tokens = features[:, 1:, :] if return_patch_tokens else None
        else:
            # [B, D] format - already CLS features
            cls_features = features
            patch_tokens = None
        
        return cls_features, patch_tokens
    
    def get_image_transform(self, is_training: bool = True):
        """Get the appropriate image transform."""
        return self.preprocess_train if is_training else self.preprocess_val
    
    @property
    def output_dim(self) -> int:
        """Return the output dimension."""
        return self.hidden_dim


class BiomedCLIPVisionEncoderHF(nn.Module):
    """
    Alternative implementation using HuggingFace transformers directly.
    Use this if open_clip is not available.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        freeze_encoder: bool = False,
        use_gradient_checkpointing: bool = True
    ):
        super().__init__()
        
        self.model_name = model_name
        self.hidden_dim = 768
        
        # Load model and processor from HuggingFace
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        if freeze_encoder:
            for param in self.model.vision_model.parameters():
                param.requires_grad = False
        
        if use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
    
    def forward(
        self,
        images: torch.Tensor,
        return_patch_tokens: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through vision encoder.
        
        Args:
            images: Input images [B, 3, 224, 224]
            return_patch_tokens: Whether to return patch tokens
            
        Returns:
            cls_features: [B, 768] CLS token features
            patch_tokens: [B, 196, 768] Patch token features (optional)
        """
        outputs = self.model.get_image_features(images, return_dict=True)
        
        if hasattr(outputs, 'last_hidden_state'):
            # Full sequence output
            hidden_states = outputs.last_hidden_state
            cls_features = hidden_states[:, 0, :]
            patch_tokens = hidden_states[:, 1:, :] if return_patch_tokens else None
        else:
            # Pooled output only
            cls_features = outputs
            patch_tokens = None
        
        return cls_features, patch_tokens
    
    @property
    def output_dim(self) -> int:
        return self.hidden_dim
