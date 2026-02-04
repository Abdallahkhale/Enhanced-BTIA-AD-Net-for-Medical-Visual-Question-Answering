# Helper function to process generative samples with tri-head routing
def process_generative_sample_trihead(sample, tokenizer, max_question_length):
    """
    Process a sample for tri-head VQA architecture.
    Returns different data based on question type.
    """
    import torch
    from data.answer_vocab import TOP_100_ANSWERS, ANSWER_TO_IDX
    
    # Determine question type
    answer_lower = sample['answer'].lower()
    
    if answer_lower in ['yes', 'no']:
        question_type = 'yes_no'
        label = 1 if answer_lower == 'yes' else 0
    elif sample['answer'] in TOP_100_ANSWERS:
        question_type = 'common_open'
        label = ANSWER_TO_IDX[sample['answer']]
    else:
        question_type = 'rare_open'
        label = None
    
    # Create prompts (no special characters)
    if question_type == 'yes_no':
        prompt = f"Based on the medical image, {sample['question']} Answer yes or no."
    else:
        prompt = f"Based on the medical image, {sample['question']} Brief answer:"
    
    # Tokenization
    if question_type in ['yes_no', 'common_open']:
        # Classification: only question
        enc = tokenizer(
            prompt,
            truncation=True,
            max_length=max_question_length,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'question_type': question_type,
            'answer_text': sample['answer']
        }
    else:
        # Generation: question + answer
        answer_text = f" {sample['answer']}"
        
        prompt_enc = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
        answer_enc = tokenizer(answer_text + tokenizer.eos_token, add_special_tokens=False, return_tensors='pt')
        
        prompt_ids = prompt_enc['input_ids'].squeeze(0)
        answer_ids = answer_enc['input_ids'].squeeze(0)
        
        input_ids = torch.cat([prompt_ids, answer_ids], dim=0)
        attention_mask = torch.ones_like(input_ids)
        
        labels = input_ids.clone()
        labels[:len(prompt_ids)] = -100
        
        # Pad
        max_len = max_question_length + 32
        if len(input_ids) < max_len:
            pad_len = max_len - len(input_ids)
            input_ids = torch.cat([input_ids, torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
        else:
            input_ids = input_ids[:max_len]
            attention_mask = attention_mask[:max_len]
            labels = labels[:max_len]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': labels,
            'question_type': question_type,
            'answer_text': sample['answer']
        }
