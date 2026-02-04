import torch
import sys
import os
sys.path.append(os.getcwd())
from models.generative_net import GenerativeMedVQA
import yaml

def check_dims():
    config = {
        'model': {
            'vision_encoder': 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
            'text_decoder': 'microsoft/BioGPT',
            'vis_hidden_dim': 768,
            'freeze_vision': True,
            'freeze_text': True
        }
    }
    
    print("Creating model...")
    model = GenerativeMedVQA(config['model'])
    model.eval()
    
    print("Checking vision encoder...")
    print(f"Vision Encoder type: {type(model.vision_encoder)}")
    
    # Create dummy input
    images = torch.randn(2, 3, 224, 224) # Batch of 2
    
    print("Running vision encoder forward...")
    with torch.no_grad():
        vision_outputs = model.vision_encoder(pixel_values=images)
        print(f"Vision Outputs type: {type(vision_outputs)}")
        if hasattr(vision_outputs, 'last_hidden_state'):
            print(f"last_hidden_state shape: {vision_outputs.last_hidden_state.shape}")
        if hasattr(vision_outputs, 'pooler_output'):
            print(f"pooler_output shape: {vision_outputs.pooler_output.shape}")
        
        # Try to replicate the error
        image_embeds = vision_outputs.last_hidden_state
        print(f"image_embeds shape: {image_embeds.shape}")
        
        try:
             projected = model.connector(image_embeds)
             print(f"Connector output shape: {projected.shape}")
        except Exception as e:
             print(f"Connector failed: {e}")

if __name__ == "__main__":
    check_dims()
