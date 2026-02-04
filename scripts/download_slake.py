"""
Script to download the SLAKE dataset from HuggingFace
"""
from datasets import load_dataset
import json
import os
from pathlib import Path
from PIL import Image
import io

def download_slake():
    print("Downloading SLAKE dataset from HuggingFace...")
    
    # Load dataset
    dataset = load_dataset("BoKelvin/SLAKE")
    
    # Create directories
    base_dir = Path("data/slake")
    images_dir = base_dir / "imgs"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    train_data = []
    test_data = []
    val_data = []
    
    # Process each split
    for split_name, split_data in [("train", dataset.get("train")), 
                                    ("validation", dataset.get("validation")),
                                    ("test", dataset.get("test"))]:
        if split_data is None:
            continue
            
        print(f"Processing {split_name} split: {len(split_data)} samples")
        
        for i, item in enumerate(split_data):
            try:
                # Save image
                if 'image' in item and item['image'] is not None:
                    img = item['image']
                    img_name = item.get('img_name', f"{split_name}_{i}.jpg")
                    img_path = images_dir / img_name
                    
                    if not img_path.exists():
                        img.save(str(img_path))
                
                # Create data entry
                entry = {
                    "image": item.get('img_name', f"{split_name}_{i}.jpg"),
                    "question": item.get('question', ''),
                    "answer": item.get('answer', ''),
                    "answer_type": item.get('answer_type', 'OPEN'),
                    "content_type": item.get('content_type', ''),
                    "modality": item.get('modality', 'CT'),
                    "q_lang": item.get('q_lang', 'en')
                }
                
                # Only include English questions
                if entry.get('q_lang', 'en') == 'en':
                    if split_name == "train":
                        train_data.append(entry)
                    elif split_name == "validation":
                        val_data.append(entry)
                    else:
                        test_data.append(entry)
                        
            except Exception as e:
                print(f"Error processing item {i}: {e}")
                continue
    
    # Save JSON files
    with open(base_dir / "train.json", "w") as f:
        json.dump(train_data, f, indent=2)
    
    with open(base_dir / "validation.json", "w") as f:
        json.dump(val_data, f, indent=2)
        
    with open(base_dir / "test.json", "w") as f:
        json.dump(test_data, f, indent=2)
    
    print(f"\nDownload complete!")
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Images saved to: {images_dir}")

if __name__ == "__main__":
    download_slake()
