"""
Dataset download utilities for VQA-RAD and SLAKE.
Auto-downloads from Hugging Face Hub.
"""

import os
import json
from pathlib import Path
from typing import Optional, Tuple
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import shutil
from tqdm import tqdm


def download_vqa_rad(
    data_dir: str = "data/vqa_rad",
    force_download: bool = False
) -> Path:
    """
    Download VQA-RAD dataset from Hugging Face.
    
    Args:
        data_dir: Directory to save dataset
        force_download: Force re-download even if exists
        
    Returns:
        Path to dataset directory
    """
    data_path = Path(data_dir)
    
    if data_path.exists() and not force_download:
        print(f"VQA-RAD already exists at {data_path}")
        return data_path
    
    print("Downloading VQA-RAD dataset from Hugging Face...")
    data_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Try the flaviagiammarino version first
        dataset = load_dataset("flaviagiammarino/vqa-rad", trust_remote_code=True)
        
        # Save dataset splits
        for split in dataset.keys():
            split_path = data_path / f"{split}.json"
            
            # Convert to list of dicts
            data = []
            for item in tqdm(dataset[split], desc=f"Processing {split}"):
                data.append({
                    'image': item.get('image'),  # PIL Image
                    'question': item.get('question'),
                    'answer': item.get('answer'),
                    'question_type': item.get('question_type', 'unknown'),
                    'answer_type': item.get('answer_type', 'unknown')
                })
            
            # Save images
            images_path = data_path / "images" / split
            images_path.mkdir(parents=True, exist_ok=True)
            
            qa_data = []
            for idx, item in enumerate(tqdm(data, desc=f"Saving {split} images")):
                if item['image'] is not None:
                    img_name = f"image_{idx:05d}.jpg"
                    img_path = images_path / img_name
                    item['image'].save(img_path)
                    
                    # Infer answer type from answer content
                    answer = str(item['answer']).lower().strip()
                    if item['answer_type'] != 'unknown':
                        answer_type = item['answer_type']
                    elif answer in ['yes', 'no']:
                        answer_type = 'CLOSED'
                    else:
                        answer_type = 'OPEN'
                    
                    qa_data.append({
                        'image_path': str(img_path.relative_to(data_path)),
                        'question': item['question'],
                        'answer': item['answer'],
                        'question_type': item['question_type'],
                        'answer_type': answer_type
                    })
            
            # Save QA pairs
            with open(split_path, 'w') as f:
                json.dump(qa_data, f, indent=2)
        
        print(f"VQA-RAD downloaded successfully to {data_path}")
        
    except Exception as e:
        print(f"Error downloading from flaviagiammarino/vqa-rad: {e}")
        print("Trying alternative source...")
        
        # Fallback: create empty structure for manual download
        (data_path / "images").mkdir(exist_ok=True)
        
        info = {
            "dataset": "VQA-RAD",
            "download_instructions": [
                "1. Visit: https://osf.io/89kps/",
                "2. Download VQA_RAD_Dataset.zip",
                "3. Extract to this directory",
                "4. Move images to data/vqa_rad/images/",
                "5. Place JSON files in data/vqa_rad/"
            ],
            "alternative_sources": [
                "https://paperswithcode.com/dataset/vqa-rad",
                "https://github.com/aioz-ai/MICCAI21_MMQ"
            ]
        }
        
        with open(data_path / "DOWNLOAD_INSTRUCTIONS.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"Created download instructions at {data_path}")
    
    return data_path


def download_slake(
    data_dir: str = "data/slake",
    force_download: bool = False
) -> Path:
    """
    Download SLAKE dataset from Hugging Face.
    
    Args:
        data_dir: Directory to save dataset
        force_download: Force re-download even if exists
        
    Returns:
        Path to dataset directory
    """
    data_path = Path(data_dir)
    
    if data_path.exists() and not force_download:
        print(f"SLAKE already exists at {data_path}")
        return data_path
    
    print("Downloading SLAKE dataset from Hugging Face...")
    data_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download from HuggingFace
        dataset = load_dataset("BoKelvin/SLAKE", trust_remote_code=True)
        
        for split in dataset.keys():
            split_path = data_path / f"{split}.json"
            images_path = data_path / "images" / split
            images_path.mkdir(parents=True, exist_ok=True)
            
            qa_data = []
            for idx, item in enumerate(tqdm(dataset[split], desc=f"Processing {split}")):
                # Save image if present
                if 'image' in item and item['image'] is not None:
                    img_name = f"image_{idx:05d}.jpg"
                    img_full_path = images_path / img_name
                    item['image'].save(img_full_path)
                    
                    qa_data.append({
                        'image_path': str(img_full_path.relative_to(data_path)),
                        'question': item.get('question', ''),
                        'answer': item.get('answer', ''),
                        'question_type': item.get('q_type', 'unknown'),
                        'answer_type': item.get('a_type', 'unknown'),
                        'modality': item.get('modality', 'unknown')
                    })
            
            with open(split_path, 'w') as f:
                json.dump(qa_data, f, indent=2)
        
        print(f"SLAKE downloaded successfully to {data_path}")
        
    except Exception as e:
        print(f"Error downloading SLAKE: {e}")
        
        # Create placeholder
        info = {
            "dataset": "SLAKE",
            "download_instructions": [
                "1. Visit: https://www.med-vqa.com/slake/",
                "2. Submit request form",
                "3. Download and extract to this directory"
            ]
        }
        
        with open(data_path / "DOWNLOAD_INSTRUCTIONS.json", 'w') as f:
            json.dump(info, f, indent=2)
    
    return data_path


def download_pathvqa(
    data_dir: str = "data/pathvqa",
    force_download: bool = False
) -> Path:
    """
    Download PathVQA dataset from Hugging Face.
    PathVQA contains 32,799 QA pairs from 4,998 pathology images.
    
    Args:
        data_dir: Directory to save dataset
        force_download: Force re-download even if exists
        
    Returns:
        Path to dataset directory
    """
    data_path = Path(data_dir)
    
    if data_path.exists() and not force_download:
        print(f"PathVQA already exists at {data_path}")
        return data_path
    
    print("Downloading PathVQA dataset from Hugging Face...")
    data_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download from HuggingFace
        dataset = load_dataset("flaviagiammarino/path-vqa", trust_remote_code=True)
        
        for split_name in dataset.keys():
            # Map split names
            out_split = "train" if "train" in split_name else "test"
            split_path = data_path / f"{out_split}.json"
            images_path = data_path / "images" / out_split
            images_path.mkdir(parents=True, exist_ok=True)
            
            qa_data = []
            for idx, item in enumerate(tqdm(dataset[split_name], desc=f"Processing {split_name}")):
                if 'image' in item and item['image'] is not None:
                    img_name = f"pathvqa_{idx:05d}.jpg"
                    img_full_path = images_path / img_name
                    item['image'].save(img_full_path)
                    
                    qa_data.append({
                        'image_path': str(img_full_path.relative_to(data_path)),
                        'question': item.get('question', ''),
                        'answer': str(item.get('answer', '')).lower().strip(),
                        'question_type': 'pathology',
                        'answer_type': 'OPEN'  # PathVQA has mostly open-ended
                    })
            
            # Append if file exists, otherwise create
            if split_path.exists():
                with open(split_path, 'r') as f:
                    existing_data = json.load(f)
                qa_data = existing_data + qa_data
            
            with open(split_path, 'w') as f:
                json.dump(qa_data, f, indent=2)
        
        print(f"PathVQA downloaded successfully to {data_path}")
        
    except Exception as e:
        print(f"Error downloading PathVQA: {e}")
        
        info = {
            "dataset": "PathVQA",
            "download_instructions": [
                "1. Visit: https://huggingface.co/datasets/flaviagiammarino/path-vqa",
                "2. Download manually if automatic fails"
            ]
        }
        
        with open(data_path / "DOWNLOAD_INSTRUCTIONS.json", 'w') as f:
            json.dump(info, f, indent=2)
    
    return data_path


def download_all_datasets(
    base_dir: str = "data",
    force_download: bool = False,
    include_pathvqa: bool = True
) -> Tuple[Path, Path]:
    """
    Download all datasets.
    
    Args:
        base_dir: Base directory for datasets
        force_download: Force re-download
        include_pathvqa: Include PathVQA dataset
        
    Returns:
        Tuple of (vqa_rad_path, slake_path)
    """
    vqa_rad_path = download_vqa_rad(
        os.path.join(base_dir, "vqa_rad"),
        force_download
    )
    
    slake_path = download_slake(
        os.path.join(base_dir, "slake"),
        force_download
    )
    
    if include_pathvqa:
        pathvqa_path = download_pathvqa(
            os.path.join(base_dir, "pathvqa"),
            force_download
        )
    
    return vqa_rad_path, slake_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download VQA datasets")
    parser.add_argument("--data_dir", type=str, default="data", help="Base data directory")
    parser.add_argument("--dataset", type=str, default="all", 
                       choices=["all", "vqa_rad", "slake"], help="Dataset to download")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    
    args = parser.parse_args()
    
    if args.dataset == "all":
        download_all_datasets(args.data_dir, args.force)
    elif args.dataset == "vqa_rad":
        download_vqa_rad(os.path.join(args.data_dir, "vqa_rad"), args.force)
    elif args.dataset == "slake":
        download_slake(os.path.join(args.data_dir, "slake"), args.force)
