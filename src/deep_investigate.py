"""
Deep Investigation: Why Open-Ended Accuracy is 0%

This script analyzes:
1. Vocabulary overlap between train/test
2. Answer distribution
3. Model predictions (if checkpoint exists)
4. Potential bugs in evaluation
"""

import json
from pathlib import Path
from collections import Counter
import sys


def load_dataset(json_path):
    """Load a dataset JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_vocabulary_coverage():
    """Analyze if test open-ended answers are in training vocabulary."""
    print("\n" + "="*70)
    print("VOCABULARY COVERAGE ANALYSIS")
    print("="*70)
    
    # Load all datasets
    vqa_rad_train = load_dataset("data/vqa_rad/train.json")
    vqa_rad_test = load_dataset("data/vqa_rad/test.json")
    
    pathvqa_train = []
    pathvqa_test = []
    
    if Path("data/pathvqa/train.json").exists():
        pathvqa_train = load_dataset("data/pathvqa/train.json")
    if Path("data/pathvqa/test.json").exists():
        pathvqa_test = load_dataset("data/pathvqa/test.json")
    
    # Combine training data
    all_train = vqa_rad_train + pathvqa_train
    all_test = vqa_rad_test + pathvqa_test
    
    print(f"\nTotal training samples: {len(all_train)}")
    print(f"Total test samples: {len(all_test)}")
    
    # Build training vocabulary (all answers)
    train_answers = set()
    train_open_answers = set()
    train_answer_counts = Counter()
    
    for item in all_train:
        answer = str(item.get('answer', '')).lower().strip()
        train_answers.add(answer)
        train_answer_counts[answer] += 1
        
        if item.get('answer_type') == 'OPEN':
            train_open_answers.add(answer)
    
    print(f"\nTraining vocabulary size (all): {len(train_answers)}")
    print(f"Training vocabulary size (open only): {len(train_open_answers)}")
    
    # Analyze test open-ended coverage
    test_open_samples = [item for item in all_test if item.get('answer_type') == 'OPEN']
    test_open_answers = set(str(item.get('answer', '')).lower().strip() for item in test_open_samples)
    
    print(f"\nTest open-ended samples: {len(test_open_samples)}")
    print(f"Test unique open answers: {len(test_open_answers)}")
    
    # Check coverage
    covered = test_open_answers & train_answers
    not_covered = test_open_answers - train_answers
    
    coverage_pct = len(covered) / len(test_open_answers) * 100 if test_open_answers else 0
    
    print(f"\n>>> CRITICAL: Test open answers IN training vocab: {len(covered)}/{len(test_open_answers)} ({coverage_pct:.1f}%)")
    
    if not_covered:
        print(f"\n⚠️ Test open answers NOT in training vocab (first 20):")
        for ans in list(not_covered)[:20]:
            print(f"   '{ans}'")
    
    # Check how often covered answers appear in training
    print(f"\n📊 Training frequency of COVERED test answers:")
    for ans in list(covered)[:10]:
        print(f"   '{ans}': {train_answer_counts[ans]} times")
    
    return coverage_pct, len(covered), len(test_open_answers)


def analyze_answer_distribution():
    """Analyze the distribution of answers in training data."""
    print("\n" + "="*70)
    print("ANSWER DISTRIBUTION ANALYSIS")
    print("="*70)
    
    # Load training data
    train_data = load_dataset("data/vqa_rad/train.json")
    if Path("data/pathvqa/train.json").exists():
        train_data += load_dataset("data/pathvqa/train.json")
    
    # Count by type
    closed_answers = Counter()
    open_answers = Counter()
    
    for item in train_data:
        answer = str(item.get('answer', '')).lower().strip()
        if item.get('answer_type') == 'CLOSED':
            closed_answers[answer] += 1
        else:
            open_answers[answer] += 1
    
    print(f"\nClosed-ended answers: {sum(closed_answers.values())} samples, {len(closed_answers)} unique")
    print(f"Open-ended answers: {sum(open_answers.values())} samples, {len(open_answers)} unique")
    
    print("\n📊 Top 10 Closed answers:")
    for ans, count in closed_answers.most_common(10):
        print(f"   '{ans}': {count}")
    
    print("\n📊 Top 10 Open answers:")
    for ans, count in open_answers.most_common(10):
        print(f"   '{ans}': {count}")
    
    # Class imbalance
    total = sum(closed_answers.values()) + sum(open_answers.values())
    closed_pct = sum(closed_answers.values()) / total * 100
    open_pct = sum(open_answers.values()) / total * 100
    
    print(f"\n⚖️ Class Balance: CLOSED {closed_pct:.1f}% vs OPEN {open_pct:.1f}%")
    
    # Answer frequency distribution for open
    print(f"\n📈 Open answer frequency distribution:")
    freq_1 = len([a for a, c in open_answers.items() if c == 1])
    freq_2_5 = len([a for a, c in open_answers.items() if 2 <= c <= 5])
    freq_6_plus = len([a for a, c in open_answers.items() if c > 5])
    
    print(f"   Appears 1 time: {freq_1} answers")
    print(f"   Appears 2-5 times: {freq_2_5} answers")
    print(f"   Appears 6+ times: {freq_6_plus} answers")
    
    return len(open_answers)


def check_test_evaluation():
    """Check specifically what test data we're evaluating on."""
    print("\n" + "="*70)
    print("TEST SET ANALYSIS")
    print("="*70)
    
    # VQA-RAD test
    vqa_rad_test = load_dataset("data/vqa_rad/test.json")
    
    vqa_open = [item for item in vqa_rad_test if item.get('answer_type') == 'OPEN']
    vqa_closed = [item for item in vqa_rad_test if item.get('answer_type') == 'CLOSED']
    
    print(f"\nVQA-RAD Test:")
    print(f"  Total: {len(vqa_rad_test)}")
    print(f"  CLOSED: {len(vqa_closed)}")
    print(f"  OPEN: {len(vqa_open)}")
    
    if vqa_open:
        print(f"\n  Sample OPEN questions from VQA-RAD test:")
        for item in vqa_open[:5]:
            print(f"    Q: {item['question'][:60]}...")
            print(f"    A: {item['answer']}")
            print()
    
    return len(vqa_open)


def check_dataset_loader():
    """Check if dataset loader is correctly handling answer types."""
    print("\n" + "="*70)
    print("DATASET LOADER CHECK")
    print("="*70)
    
    # Read the dataset.py file to understand evaluation
    dataset_path = Path("data/dataset.py")
    
    if dataset_path.exists():
        content = dataset_path.read_text()
        
        # Check for answer_type handling
        if 'answer_type' in content:
            print("✅ dataset.py contains 'answer_type' handling")
        else:
            print("❌ dataset.py does NOT contain 'answer_type' handling!")
        
        # Check for vocabulary building
        if 'answer_vocab' in content.lower() or 'answer2idx' in content:
            print("✅ dataset.py contains vocabulary building")
        else:
            print("❌ dataset.py does NOT contain vocabulary building!")
    
    # Check train.py for evaluation logic
    train_path = Path("src/train.py")
    
    if train_path.exists():
        content = train_path.read_text()
        
        # Look for separate open/closed accuracy calculation
        if 'open' in content.lower() and 'accuracy' in content.lower():
            print("✅ train.py calculates open accuracy separately")
            
            # Find the evaluation section
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'open' in line.lower() and 'acc' in line.lower():
                    # Print context
                    start = max(0, i-3)
                    end = min(len(lines), i+4)
                    print(f"\n  Code around line {i+1}:")
                    for j in range(start, end):
                        marker = ">>>" if j == i else "   "
                        print(f"  {marker} {j+1}: {lines[j][:80]}")
                    break


def main():
    print("="*70)
    print("DEEP INVESTIGATION: Why Open-Ended Accuracy = 0%")
    print("="*70)
    
    # Run all analyses
    coverage_pct, covered, total_open = analyze_vocabulary_coverage()
    num_open_answers = analyze_answer_distribution()
    num_test_open = check_test_evaluation()
    check_dataset_loader()
    
    # Summary and recommendations
    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    
    print("\n🔍 KEY FINDINGS:")
    
    if coverage_pct < 30:
        print(f"   ❌ CRITICAL: Only {coverage_pct:.1f}% of test open answers are in training vocab!")
        print("      → The model CAN'T predict answers it never saw during training")
    
    if num_open_answers > 1000:
        print(f"   ⚠️ WARNING: {num_open_answers} unique open answers - very sparse!")
        print("      → Each answer has very few training examples")
    
    print("\n💡 RECOMMENDED SOLUTIONS:")
    print("   1. Merge test vocabulary into training vocabulary")
    print("   2. Use embedding-based answer matching (not classification)")
    print("   3. Add more training data with diverse answers")
    print("   4. Use answer generation instead of classification for open-ended")


if __name__ == "__main__":
    main()
