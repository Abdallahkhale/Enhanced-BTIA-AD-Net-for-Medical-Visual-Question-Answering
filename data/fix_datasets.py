"""
Comprehensive Dataset Fixer
Fixes all identified issues:
1. VQA-RAD: Infer question_type from question content
2. PathVQA: Infer answer_type from answers (yes/no = CLOSED)
3. SLAKE: Download images from alternative source or skip
"""

import json
import re
from pathlib import Path
from collections import Counter


def infer_question_type(question: str) -> str:
    """Infer question type from question content."""
    q = question.lower().strip()
    
    # Yes/No questions (closed-ended)
    if q.startswith(('is ', 'are ', 'does ', 'do ', 'was ', 'were ', 'has ', 'have ', 'can ', 'could ', 'should ', 'would ')):
        return 'YESNO'
    
    # Choice questions
    if ' or ' in q and q.endswith('?'):
        return 'CHOICE'
    
    # What questions
    if q.startswith('what'):
        if 'plane' in q or 'view' in q or 'projection' in q:
            return 'PLANE'
        if 'modality' in q or 'type of image' in q or 'imaging' in q:
            return 'MODALITY'
        if 'organ' in q or 'structure' in q or 'anatomy' in q:
            return 'ORGAN'
        if 'abnormal' in q or 'finding' in q or 'patholog' in q or 'condition' in q:
            return 'ABNORMALITY'
        return 'ATTRIBUTE'
    
    # Where questions
    if q.startswith('where') or 'location' in q:
        return 'POSITION'
    
    # How questions
    if q.startswith('how'):
        return 'ATTRIBUTE'
    
    # Which questions
    if q.startswith('which'):
        return 'ATTRIBUTE'
    
    # Count questions
    if q.startswith('how many') or 'count' in q or 'number of' in q:
        return 'COUNT'
    
    # Size questions
    if 'size' in q or 'large' in q or 'small' in q:
        return 'SIZE'
    
    # Color questions
    if 'color' in q or 'colour' in q:
        return 'COLOR'
    
    return 'OTHER'


def infer_answer_type(answer: str) -> str:
    """Infer answer type from answer content."""
    a = str(answer).lower().strip()
    
    if a in ['yes', 'no']:
        return 'CLOSED'
    return 'OPEN'


def fix_vqa_rad(data_dir: str = "data/vqa_rad"):
    """Fix VQA-RAD dataset: add proper question_type and answer_type."""
    data_path = Path(data_dir)
    
    print("\n=== Fixing VQA-RAD ===")
    
    for split in ['train', 'test']:
        json_path = data_path / f"{split}.json"
        
        if not json_path.exists():
            print(f"  {split}.json not found, skipping")
            continue
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        question_types = Counter()
        answer_types = Counter()
        
        for item in data:
            # Fix question_type
            if item.get('question_type', 'unknown') == 'unknown':
                item['question_type'] = infer_question_type(item.get('question', ''))
            question_types[item['question_type']] += 1
            
            # Fix answer_type  
            if item.get('answer_type', 'unknown') == 'unknown':
                item['answer_type'] = infer_answer_type(item.get('answer', ''))
            answer_types[item['answer_type']] += 1
        
        # Save
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n  {split}.json ({len(data)} samples):")
        print(f"    Question types: {dict(question_types)}")
        print(f"    Answer types: {dict(answer_types)}")


def fix_pathvqa(data_dir: str = "data/pathvqa"):
    """Fix PathVQA dataset: infer answer_type from answers."""
    data_path = Path(data_dir)
    
    print("\n=== Fixing PathVQA ===")
    
    for split in ['train', 'test']:
        json_path = data_path / f"{split}.json"
        
        if not json_path.exists():
            print(f"  {split}.json not found, skipping")
            continue
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        answer_types = Counter()
        
        for item in data:
            # Infer answer_type from answer
            item['answer_type'] = infer_answer_type(item.get('answer', ''))
            
            # Also infer question_type
            if item.get('question_type') in ['unknown', 'pathology', None]:
                item['question_type'] = infer_question_type(item.get('question', ''))
            
            answer_types[item['answer_type']] += 1
        
        # Save
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n  {split}.json ({len(data)} samples):")
        print(f"    Answer types: {dict(answer_types)}")


def check_slake(data_dir: str = "data/slake"):
    """Check SLAKE status and provide solution."""
    data_path = Path(data_dir)
    
    print("\n=== Checking SLAKE ===")
    
    if not data_path.exists():
        print("  SLAKE folder does not exist")
        return False
    
    # Check if train.json has content
    train_json = data_path / "train.json"
    if train_json.exists():
        with open(train_json, 'r') as f:
            data = json.load(f)
        if len(data) == 0:
            print("  ❌ SLAKE train.json is EMPTY")
            print("\n  REASON: The HuggingFace 'BoKelvin/SLAKE' dataset")
            print("          contains only metadata (questions/answers)")
            print("          but NOT the actual images!")
            print("\n  SOLUTION OPTIONS:")
            print("  1. Download SLAKE manually from: https://www.med-vqa.com/slake/")
            print("  2. Or skip SLAKE and use VQA-RAD + PathVQA only")
            print("\n  For now, SLAKE will be DISABLED in training.")
            return False
    
    return True


def print_summary(data_dirs: dict):
    """Print final dataset summary."""
    print("\n" + "="*60)
    print("DATASET SUMMARY AFTER FIXES")
    print("="*60)
    
    total_train = 0
    total_test = 0
    
    for name, path in data_dirs.items():
        train_json = Path(path) / "train.json"
        test_json = Path(path) / "test.json"
        
        train_count = 0
        test_count = 0
        
        if train_json.exists():
            with open(train_json, 'r') as f:
                train_count = len(json.load(f))
        if test_json.exists():
            with open(test_json, 'r') as f:
                test_count = len(json.load(f))
        
        status = "✅" if train_count > 0 else "❌"
        print(f"\n{status} {name}:")
        print(f"   Train: {train_count:,} samples")
        print(f"   Test: {test_count:,} samples")
        
        total_train += train_count
        total_test += test_count
    
    print(f"\n{'='*60}")
    print(f"TOTAL: {total_train:,} train + {total_test:,} test = {total_train + total_test:,} samples")
    print("="*60)


def main():
    print("="*60)
    print("COMPREHENSIVE DATASET FIXER")
    print("="*60)
    
    # Fix each dataset
    fix_vqa_rad("data/vqa_rad")
    fix_pathvqa("data/pathvqa")
    check_slake("data/slake")
    
    # Print summary
    print_summary({
        'VQA-RAD': 'data/vqa_rad',
        'PathVQA': 'data/pathvqa',
        'SLAKE': 'data/slake'
    })
    
    print("\n✅ All fixable issues have been resolved!")
    print("\nNext step: Run training with:")
    print("  python src/train.py --config config/config.yaml")


if __name__ == "__main__":
    main()
