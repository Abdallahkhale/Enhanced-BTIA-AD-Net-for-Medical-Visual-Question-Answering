import json
from collections import Counter

# Load PathVQA dataset
print("=== PathVQA Dataset Analysis ===\n")
with open('data/pathvqa/train.json', 'r') as f:
    data = json.load(f)

# 1. Check for duplicates
total_samples = len(data)
unique_questions = len(set(d['question'] for d in data))
unique_qa_pairs = len(set((d['question'], d['answer']) for d in data))

print(f"Total samples: {total_samples}")
print(f"Unique questions: {unique_questions}")
print(f"Unique Q-A pairs: {unique_qa_pairs}")
print(f"Duplicate questions: {total_samples - unique_questions}")
print(f"Duplicate Q-A pairs: {total_samples - unique_qa_pairs}\n")

# 2. Analyze answer vocabulary
answers = [d['answer'] for d in data]
unique_answers = set(answers)
answer_counts = Counter(answers)

print(f"=== Answer Statistics ===")
print(f"Total answers: {len(answers)}")
print(f"Unique answers: {len(unique_answers)}\n")

# 3. Check yes/no vs open
yes_no_count = sum(1 for a in answers if a.lower() in ['yes', 'no'])
open_count = total_samples - yes_no_count

print(f"Yes/No questions: {yes_no_count} ({yes_no_count/total_samples*100:.1f}%)")
print(f"Open questions: {open_count} ({open_count/total_samples*100:.1f}%)\n")

# 4. Top answers
print(f"=== Top 30 Most Common Answers ===")
for i, (ans, count) in enumerate(answer_counts.most_common(30), 1):
    print(f"{i}. '{ans}': {count} times ({count/total_samples*100:.1f}%)")

# 5. Answer length distribution
answer_lengths = [len(a.split()) for a in answers]
print(f"\n=== Answer Length Distribution ===")
print(f"Average words per answer: {sum(answer_lengths)/len(answer_lengths):.2f}")
print(f"1 word answers: {sum(1 for l in answer_lengths if l == 1)}")
print(f"2-3 word answers: {sum(1 for l in answer_lengths if 2 <= l <= 3)}")
print(f"4-10 word answers: {sum(1 for l in answer_lengths if 4 <= l <= 10)}")
print(f">10 word answers: {sum(1 for l in answer_lengths if l > 10)}")

# 6. Can we use classification?
print(f"\n=== Classification Feasibility ===")
print(f"If we take top 100 answers, coverage: {sum(count for _, count in answer_counts.most_common(100))/total_samples*100:.1f}%")
print(f"If we take top 500 answers, coverage: {sum(count for _, count in answer_counts.most_common(500))/total_samples*100:.1f}%")
print(f"If we take top 1000 answers, coverage: {sum(count for _, count in answer_counts.most_common(1000))/total_samples*100:.1f}%")
