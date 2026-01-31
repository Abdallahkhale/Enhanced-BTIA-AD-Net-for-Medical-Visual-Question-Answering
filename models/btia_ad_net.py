"""
Enhanced BTIA-AD Net
Complete model combining BiomedCLIP, PubMedBERT, Answer Distillation, and BTIA Fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any

from .vision_encoder import BiomedCLIPVisionEncoder
from .text_encoder import PubMedBERTEncoder, AnswerEmbeddingCache
from .answer_distillation import AnswerDistillationNetwork
from .btia_fusion import BiTextImageAttention


class EnhancedBTIANet(nn.Module):
    """
    Enhanced BTIA-AD Net for Medical Visual Question Answering.
    
    Architecture:
    1. BiomedCLIP for visual feature extraction
    2. PubMedBERT for question encoding
    3. Answer Distillation Network for Top-K candidate selection
    4. Bi-Text-Image Attention for multimodal fusion
    5. Classification head for final answer prediction
    """
    
    def __init__(
        self,
        vision_encoder_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        text_encoder_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        num_answers: int = 1000,
        hidden_dim: int = 768,
        num_heads: int = 8,
        num_attention_layers: int = 2,
        top_k: int = 5,
        dropout: float = 0.3,
        freeze_vision: bool = False,
        freeze_text: bool = False,
        use_gradient_checkpointing: bool = True
    ):
        super().__init__()
        
        self.num_answers = num_answers
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        
        # Vision Encoder (BiomedCLIP)
        self.vision_encoder = BiomedCLIPVisionEncoder(
            model_name=vision_encoder_name,
            freeze_encoder=freeze_vision,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        
        # Text Encoder (PubMedBERT)
        self.text_encoder = PubMedBERTEncoder(
            model_name=text_encoder_name,
            freeze_encoder=freeze_text,
            use_gradient_checkpointing=use_gradient_checkpointing
        )
        
        # Answer Embedding Cache
        self.answer_cache = None
        
        # Answer Distillation Network
        self.answer_distillation = AnswerDistillationNetwork(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_answers=num_answers,
            top_k=top_k,
            dropout=dropout
        )
        
        # Bi-Text-Image Attention Fusion
        self.btia_fusion = BiTextImageAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_attention_layers,
            dropout=dropout,
            use_gated_fusion=True
        )
        
        # Classification Head (for backward compatibility)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_answers)
        )
        
        # Embedding Matching Head (Solution 3: projects to answer embedding space)
        self.embedding_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Temperature for softmax scaling in embedding matching
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        
        # Mode flag: True = use embedding matching, False = use classification
        self.use_embedding_matching = True
        
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """Initialize classifier weights."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def build_answer_cache(self, answers: List[str], device: torch.device):
        """
        Build the answer embedding cache.
        
        Args:
            answers: List of unique answer strings
            device: Target device
        """
        self.answer_cache = AnswerEmbeddingCache(self.text_encoder)
        self.answer_cache.build_cache(answers, device)
        
        # Update num_answers
        self.num_answers = len(answers)
        
        # Update answer distillation
        self.answer_distillation.num_answers = len(answers)
        self.answer_distillation.answer_classifier = nn.Linear(
            self.hidden_dim, len(answers)
        ).to(device)
        
        # Initialize class weights (uniform by default)
        self.class_weights = None
    
    def set_class_weights(self, weights: torch.Tensor, device: torch.device):
        """
        Set class weights for balanced training.
        
        Args:
            weights: Tensor of shape [num_answers] with weights for each class
            device: Target device
        """
        self.class_weights = weights.to(device)
    
    def forward(
        self,
        images: torch.Tensor,
        questions: Optional[List[str]] = None,
        question_ids: Optional[torch.Tensor] = None,
        question_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model.
        
        Args:
            images: [B, 3, 224, 224] Input images
            questions: List of question strings (if using raw text)
            question_ids: [B, L] Tokenized question IDs
            question_mask: [B, L] Attention mask for questions
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing:
                - logits: [B, N] Final answer logits
                - ad_logits: [B, N] Answer distillation logits
                - top_k_indices: [B, K] Top-K answer indices
                - top_k_probs: [B, K] Top-K probabilities
                - attention: Optional attention weights
        """
        # 1. Extract visual features
        visual_features, _ = self.vision_encoder(images)  # [B, 768]
        
        # 2. Extract question features
        if questions is not None:
            question_features, _ = self.text_encoder(texts=questions)  # [B, 768]
        else:
            question_features, _ = self.text_encoder(
                input_ids=question_ids,
                attention_mask=question_mask
            )  # [B, 768]
        
        # 3. Get answer embeddings
        answer_embeddings = None
        if self.answer_cache is not None:
            answer_embeddings = self.answer_cache.cache['embeddings']  # [N, 768]
        
        # 4. Answer Distillation
        ad_logits, top_k_indices, top_k_probs, answer_context = self.answer_distillation(
            visual_features,
            question_features,
            answer_embeddings
        )
        
        # 5. Bi-Text-Image Attention Fusion
        fused_features, attention = self.btia_fusion(
            visual_features,
            question_features,
            answer_context,
            return_attention=return_attention
        )  # [B, 768]
        
        # 6. Final Prediction - Use embedding matching or classification
        if self.use_embedding_matching and answer_embeddings is not None:
            # Project fused features to embedding space
            projected_features = self.embedding_projection(fused_features)  # [B, 768]
            
            # L2 normalize for cosine similarity
            projected_features = F.normalize(projected_features, p=2, dim=-1)
            normalized_answer_embeddings = F.normalize(answer_embeddings, p=2, dim=-1)
            
            # Compute cosine similarity: [B, N]
            # Temperature-scaled similarity for sharper distributions
            similarity = torch.matmul(projected_features, normalized_answer_embeddings.t())
            logits = similarity / self.temperature.clamp(min=0.01)
        else:
            # Traditional classification
            logits = self.classifier(fused_features)  # [B, N]
        
        outputs = {
            'logits': logits,
            'ad_logits': ad_logits,
            'top_k_indices': top_k_indices,
            'top_k_probs': top_k_probs,
            'visual_features': visual_features,
            'question_features': question_features,
            'fused_features': fused_features
        }
        
        if self.use_embedding_matching and answer_embeddings is not None:
            outputs['projected_features'] = projected_features
        
        if return_attention:
            outputs['attention'] = attention
        
        return outputs
    
    def predict(
        self,
        images: torch.Tensor,
        questions: List[str]
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Make predictions for input images and questions.
        
        Args:
            images: [B, 3, 224, 224] Input images
            questions: List of question strings
            
        Returns:
            predictions: [B] Predicted answer indices
            answers: List of predicted answer strings
        """
        self.eval()
        
        with torch.no_grad():
            outputs = self.forward(images, questions)
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=-1)
        
        # Convert to answer strings
        if self.answer_cache is not None:
            answers = [
                self.answer_cache.idx_to_answer.get(idx.item(), '<UNK>')
                for idx in predictions
            ]
        else:
            answers = [str(idx.item()) for idx in predictions]
        
        return predictions, answers
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        ad_weight: float = 0.3
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        
        Args:
            outputs: Model outputs from forward()
            targets: [B] Ground truth answer indices
            ad_weight: Weight for answer distillation loss
            
        Returns:
            Dictionary containing:
                - loss: Total loss
                - cls_loss: Classification loss
                - ad_loss: Answer distillation loss
        """
        # Use Focal Loss to address class imbalance
        # This down-weights easy yes/no predictions and focuses on harder answers
        cls_loss = self._focal_loss(outputs['logits'], targets, gamma=2.0, alpha=0.25)
        
        # Answer distillation loss (also use focal loss)
        ad_loss = self._focal_loss(outputs['ad_logits'], targets, gamma=2.0, alpha=0.25)
        
        # Embedding matching loss (contrastive)
        embedding_loss = torch.tensor(0.0, device=targets.device)
        if self.use_embedding_matching and 'projected_features' in outputs:
            embedding_loss = self._contrastive_embedding_loss(
                outputs['projected_features'], 
                targets
            )
        
        # Total loss
        if self.use_embedding_matching:
            # For embedding mode: emphasize embedding loss
            total_loss = cls_loss + ad_weight * ad_loss + 0.5 * embedding_loss
        else:
            total_loss = cls_loss + ad_weight * ad_loss
        
        return {
            'loss': total_loss,
            'cls_loss': cls_loss,
            'ad_loss': ad_loss,
            'embedding_loss': embedding_loss
        }
    
    def _contrastive_embedding_loss(
        self,
        projected_features: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Contrastive loss to pull features towards correct answer embeddings.
        
        Args:
            projected_features: [B, 768] Projected and normalized features
            targets: [B] Target answer indices
            
        Returns:
            Contrastive loss value
        """
        if self.answer_cache is None:
            return torch.tensor(0.0, device=projected_features.device)
        
        # Get answer embeddings
        answer_embeddings = self.answer_cache.cache['embeddings']  # [N, 768]
        normalized_answer_embeddings = F.normalize(answer_embeddings, p=2, dim=-1)
        
        # Get target embeddings
        target_embeddings = normalized_answer_embeddings[targets]  # [B, 768]
        
        # Positive similarity (should be high)
        positive_sim = (projected_features * target_embeddings).sum(dim=-1)  # [B]
        
        # Contrastive loss: maximize positive similarity using InfoNCE-style loss
        # logits = projected_features @ all_answer_embeddings.T / temperature
        logits = torch.matmul(projected_features, normalized_answer_embeddings.t())
        logits = logits / self.temperature.clamp(min=0.01)
        
        # Cross entropy loss treats this as classification
        loss = F.cross_entropy(logits, targets)
        
        return loss
    
    def _focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        gamma: float = 2.0,
        alpha: float = 0.25
    ) -> torch.Tensor:
        """
        Focal Loss for addressing class imbalance.
        
        FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
        
        Args:
            logits: [B, N] Predicted logits
            targets: [B] Target indices
            gamma: Focusing parameter (higher = more focus on hard examples)
            alpha: Weighting factor
            
        Returns:
            Focal loss value
        """
        # Use class weights if available
        if self.class_weights is not None:
            ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction='none')
        else:
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Get probability of correct class
        probs = F.softmax(logits, dim=-1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** gamma
        
        # Apply focal weight
        focal_loss = alpha * focal_weight * ce_loss
        
        return focal_loss.mean()
    
    def get_trainable_params(self) -> List[Dict[str, Any]]:
        """
        Get parameter groups for optimizer with different learning rates.
        
        Returns:
            List of parameter dictionaries
        """
        # Lower learning rate for pretrained encoders
        encoder_params = list(self.vision_encoder.parameters()) + \
                        list(self.text_encoder.parameters())
        
        # Higher learning rate for new modules
        new_params = list(self.answer_distillation.parameters()) + \
                    list(self.btia_fusion.parameters()) + \
                    list(self.classifier.parameters())
        
        return [
            {'params': encoder_params, 'lr_scale': 0.1},
            {'params': new_params, 'lr_scale': 1.0}
        ]
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count model parameters.
        
        Returns:
            Dictionary with parameter counts
        """
        def count_module(module):
            return sum(p.numel() for p in module.parameters())
        
        def count_trainable(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        return {
            'total': count_module(self),
            'trainable': count_trainable(self),
            'vision_encoder': count_module(self.vision_encoder),
            'text_encoder': count_module(self.text_encoder),
            'answer_distillation': count_module(self.answer_distillation),
            'btia_fusion': count_module(self.btia_fusion),
            'classifier': count_module(self.classifier)
        }


def create_model(config: dict) -> EnhancedBTIANet:
    """
    Create model from config dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized model
    """
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    
    return EnhancedBTIANet(
        vision_encoder_name=model_config.get('vision_encoder', 
            "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"),
        text_encoder_name=model_config.get('text_encoder',
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"),
        num_answers=model_config.get('num_answers', 1000),
        hidden_dim=model_config.get('hidden_dim', 768),
        num_heads=model_config.get('num_heads', 8),
        num_attention_layers=model_config.get('num_attention_layers', 2),
        top_k=model_config.get('top_k_answers', 5),
        dropout=model_config.get('dropout', 0.3),
        use_gradient_checkpointing=training_config.get('gradient_checkpointing', True)
    )
