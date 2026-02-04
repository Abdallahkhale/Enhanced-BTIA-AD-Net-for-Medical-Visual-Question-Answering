
import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoModelForCausalLM, AutoTokenizer,
    VisionEncoderDecoderModel, ViTModel, GPT2Model
)
from typing import Optional, Dict, List, Any
from peft import LoraConfig, get_peft_model, TaskType

class GenerativeMedVQA(nn.Module):
    """
    Generative Medical VQA Model (Mini-LLaVA SOTA Architecture).
    
    Components:
    - Vision Encoder: BiomedCLIP (PubMedBERT + ViT) -> Use Vision Tower
    - Connector: Linear Projection / Cross-Attention
    - Text Decoder: BioGPT (Microsoft) or DistilGPT2 (for efficiency)
    
    Architecture:
    [Image] -> [ViT] -> [Visual Tokens] -> [Projection] -> [LLM Decoder] <- [Question Tokens]
                                                                     |
                                                                  [Answer Tokens]
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # 1. Vision Encoder: BiomedCLIP Vision Tower
        vision_model_name = config.get('vision_encoder', "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
        print(f"Loading Vision Encoder: {vision_model_name}")
        # Note: BiomedCLIP is a CLIP model, we need the vision tower. 
        # But loading the whole CLIP model is huge. 
        # Efficient way: Load ViT directly if possible, or load CLIP and extract.
        # Let's try loading as generic model first.
        try:
            self.vision_encoder = AutoModel.from_pretrained(vision_model_name)
            # If it's CLIP, it has .vision_model
            if hasattr(self.vision_encoder, "vision_model"):
                self.vision_encoder = self.vision_encoder.vision_model
        except Exception as e:
            print(f"Warning: Could not load specific vision model, falling back to safe generic loading: {e}")
            # Fallback (e.g. if name is internal path)
            self.vision_encoder = AutoModel.from_pretrained("microsoft/resnet-50") # Placeholder
            
        # Freeze vision encoder?
        if config.get('freeze_vision', False):
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
                
        # 2. Text Decoder: BioGPT or GPT2 with LoRA
        decoder_model_name = config.get('text_decoder', "microsoft/BioGPT")
        print(f"Loading Text Decoder: {decoder_model_name}")
        self.decoder = AutoModelForCausalLM.from_pretrained(decoder_model_name)
        
        # Apply LoRA for parameter-efficient fine-tuning (prevents overfitting)
        use_lora = config.get('use_lora', True)
        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,  # Rank of the low-rank matrices
                lora_alpha=32,  # Scaling factor
                lora_dropout=0.1,  # Dropout for regularization
                target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],  # Target attention layers
                inference_mode=False,
            )
            self.decoder = get_peft_model(self.decoder, lora_config)
            self.decoder.print_trainable_parameters()
            print("LoRA enabled: Training only adapter layers")
            self._use_lora = True
        else:
            self._use_lora = False
            if config.get('freeze_text', False):
                for param in self.decoder.parameters():
                    param.requires_grad = False
        
        # Connector: Project visual dimension to decoder dimension
        # ViT Output: [B, 197, 768] (for ViT-Base)
        # BioGPT Embed: 1024 (BioGPT-Large) or 768?
        # Let's check dims dynamically or hardcode for known models
        # Determine vision hidden dimension dynamically
        # Create a dummy input to check output shape
        with torch.no_grad():
             dummy_images = torch.ones(1, 3, 224, 224)
             try:
                 dummy_out = self.vision_encoder(pixel_values=dummy_images)
                 if hasattr(dummy_out, 'last_hidden_state'):
                     last_hidden = dummy_out.last_hidden_state
                     # If 4D: [B, C, H, W] -> Channels is dim 1
                     if len(last_hidden.shape) == 4:
                         self.vis_hidden_dim = last_hidden.shape[1]
                     # If 3D: [B, L, C] -> Channels is dim 2
                     elif len(last_hidden.shape) == 3:
                         self.vis_hidden_dim = last_hidden.shape[2]
                     else:
                         self.vis_hidden_dim = config.get('vis_hidden_dim', 768)
                 else:
                     # Fallback if no last_hidden_state
                     if hasattr(dummy_out, 'pooler_output'):
                          self.vis_hidden_dim = dummy_out.pooler_output.shape[1]
                     else:
                          self.vis_hidden_dim = config.get('vis_hidden_dim', 768)
             except Exception as e:
                 print(f"Warning: Could not determine vision dim dynamically: {e}")
                 self.vis_hidden_dim = config.get('vis_hidden_dim', 768)
                 
        print(f"Detected Vision Hidden Dim: {self.vis_hidden_dim}")
        
        self.text_hidden_dim = self.decoder.config.hidden_size
        
        # Improved projection: 2-layer MLP with GELU (LLaVA-1.5 style)
        # This provides better cross-modal alignment than simple linear projection
        self.connector = nn.Sequential(
            nn.Linear(self.vis_hidden_dim, self.text_hidden_dim),
            nn.GELU(),
            nn.Linear(self.text_hidden_dim, self.text_hidden_dim)
        )
        
        # Support for two-stage training (LLaVA approach)
        # Stage 1: Freeze LLM, train only projection (pretraining_mode=True)
        # Stage 2: Unfreeze LLM, train projection + LLM (pretraining_mode=False)
        self._pretraining_mode = False
        
    def forward(
        self, 
        images: torch.Tensor, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Causal LM training.
        
        Args:
            images: [B, C, H, W]
            input_ids: [B, L] (Question + Answer tokens)
            attention_mask: [B, L]
            labels: [B, L] (Masked labels for loss, -100 for question parts)
        """
        # 1. Encode Images
        # [B, 3, 224, 224] -> [B, N_vis, D_vis]
        # ViT output usually is object with last_hidden_state
        vision_outputs = self.vision_encoder(pixel_values=images)
        if hasattr(vision_outputs, 'last_hidden_state'):
            image_embeds = vision_outputs.last_hidden_state 
            
            # Handle 4D output (ResNet-like): [B, C, H, W] -> [B, H*W, C]
            if len(image_embeds.shape) == 4:
                # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
                b, c, h, w = image_embeds.shape
                image_embeds = image_embeds.view(b, c, -1).permute(0, 2, 1)
        else:
             # Fallback to pooler_output (single token per image)
             image_embeds = vision_outputs.pooler_output.unsqueeze(1) # [B, 1, C]
        
        # 2. Project to Text Dimension
        # [B, N_vis, D_text]
        image_embeds = self.connector(image_embeds)
        
        # 3. Concatenate? No, standard LLaVA/Causal approach:
        # We can pass image_embeds as `encoder_hidden_states` if using Encoder-Decoder.
        # BUT BioGPT is Decoder-ONLY.
        # So we must PREPEND visual tokens to the input_ids embeddings.
        
        # Embed text inputs
        input_embeds = self.decoder.get_input_embeddings()(input_ids) # [B, L, D_text]
        
        # Concatenate: [Image_Tokens, Text_Tokens]
        # [B, N_vis + L, D_text]
        inputs_embeds = torch.cat([image_embeds, input_embeds], dim=1)
        
        # Extend attention mask
        # [B, N_vis] ones + [B, L] original mask
        batch_size = images.shape[0]
        vis_mask = torch.ones((batch_size, image_embeds.shape[1]), device=images.device)
        attention_mask = torch.cat([vis_mask, attention_mask], dim=1)
        
        # Extend labels?
        # Image tokens should be ignored (-100)
        if labels is not None:
             vis_labels = torch.full((batch_size, image_embeds.shape[1]), -100, device=images.device, dtype=labels.dtype)
             labels = torch.cat([vis_labels, labels], dim=1)
        
        # 4. Decoder Forward
        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs

    def generate(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 20,
        **kwargs
    ):
        """
        Generate text answers.
        """
        # Encode images
        vision_outputs = self.vision_encoder(pixel_values=images)
        if hasattr(vision_outputs, 'last_hidden_state'):
            image_embeds = vision_outputs.last_hidden_state 
            # Handle 4D output (ResNet-like): [B, C, H, W] -> [B, H*W, C]
            if len(image_embeds.shape) == 4:
                b, c, h, w = image_embeds.shape
                image_embeds = image_embeds.view(b, c, -1).permute(0, 2, 1)
        else:
             image_embeds = vision_outputs.pooler_output.unsqueeze(1)

        image_embeds = self.connector(image_embeds)
        
        # Prepare inputs
        # We need to construct the prompt embeddings: [Image, Question]
        # Then let it auto-regressively generate.
        
        # Only passed embedding of [Image, Question]
        input_embeds = self.decoder.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([image_embeds, input_embeds], dim=1)
        
        # Attention mask
        batch_size = images.shape[0]
        vis_mask = torch.ones((batch_size, image_embeds.shape[1]), device=images.device)
        attention_mask = torch.cat([vis_mask, attention_mask], dim=1)
        
        outputs = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        
        return outputs

    def set_pretraining_mode(self, enable: bool):
        """
        Enable/disable pretraining mode (Stage 1 vs Stage 2).
        
        Stage 1 (pretraining): Freeze LLM decoder, train only projection.
        Stage 2 (fine-tuning): Unfreeze LLM decoder/LoRA, train both.
        
        Args:
            enable: True for Stage 1 (projection only), False for Stage 2.
        """
        self._pretraining_mode = enable
        
        if enable:
            # Stage 1: Freeze LLM decoder (including LoRA if used)
            for param in self.decoder.parameters():
                param.requires_grad = False
            print("Pretraining mode ENABLED: LLM decoder frozen, training projection only")
        else:
            # Stage 2: Unfreeze LoRA adapters (or all if not using LoRA)
            if hasattr(self, '_use_lora') and self._use_lora:
                # Only unfreeze LoRA parameters
                for name, param in self.decoder.named_parameters():
                    if 'lora' in name.lower():
                        param.requires_grad = True
                print("Pretraining mode DISABLED: LoRA adapters unfrozen, training projection + LoRA")
            else:
                for param in self.decoder.parameters():
                    param.requires_grad = True
                print("Pretraining mode DISABLED: LLM decoder unfrozen, training projection + LLM")
    
    def get_trainable_params(self):
        """
        Get trainable parameters split into groups.
        
        Returns:
            List of dictionaries with 'params' and other keys.
        """
        # Group 1: LLM decoder parameters (lower learning rate)
        # Only include if not in pretraining mode
        group1 = []
        if hasattr(self, 'decoder'):
            trainable_decoder_params = [p for p in self.decoder.parameters() if p.requires_grad]
            if trainable_decoder_params:
                group1.extend(trainable_decoder_params)
             
        # Group 2: Projection layer parameters (higher learning rate)
        # Always trained
        group2 = list(self.connector.parameters())
        
        return [
            {'params': group1},  # LLM (lower LR, may be empty in pretraining mode)
            {'params': group2}   # Projection (higher LR)
        ]

def create_generative_model(config: Dict[str, Any]):
    return GenerativeMedVQA(config.get('model', {}))
