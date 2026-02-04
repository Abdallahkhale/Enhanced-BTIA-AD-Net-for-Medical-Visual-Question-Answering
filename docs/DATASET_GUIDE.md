# Medical VQA Dataset & Usage Guide

## Overview

This Medical Visual Question Answering (VQA) system is trained on medical imaging datasets to answer questions about radiology images. The model combines visual understanding (BiomedCLIP) with language generation (BioGPT) to provide medically-relevant answers.

---

## Training Datasets

### 1. VQA-RAD (Radiology)
- **Size**: ~3,500 question-answer pairs
- **Imaging Modalities**: CT, MRI, X-ray
- **Body Regions**: Head, chest, abdomen
- **Focus**: Clinical radiology interpretation

### 2. PathVQA (Pathology)
- **Size**: ~32,000 question-answer pairs
- **Imaging Types**: Histopathology slides, tissue samples
- **Focus**: Pathological tissue analysis

---

## Question Types

### Closed Questions (Yes/No)
Best accuracy (~50%+). Examples:

| Question Pattern | Example |
|------------------|---------|
| Presence detection | "Is there evidence of a fracture?" |
| Abnormality check | "Are there any abnormalities in the lungs?" |
| Visibility confirmation | "Is the liver visible in this image?" |
| Condition verification | "Is this a normal chest X-ray?" |

### Open Questions (Free-form)
More challenging (~20-40% accuracy). Examples:

| Question Pattern | Example |
|------------------|---------|
| Location identification | "Where is the lesion located?" |
| Organ/structure naming | "What organ is shown in this image?" |
| Side identification | "Which side shows the abnormality?" |
| Modality identification | "What imaging modality was used?" |

---

## Best Practices for Asking Questions

### ✅ DO:
- Ask specific, focused questions
- Use medical terminology when appropriate
- Ask about visible structures and findings
- Keep questions concise

### ❌ DON'T:
- Ask for diagnoses or treatment recommendations
- Expect detailed explanations or reports
- Ask about patient history or demographics
- Rely on this for clinical decision-making

---

## Use Cases

### 1. Medical Education
- Teaching radiology interpretation
- Training medical students on image analysis
- Interactive learning tool

### 2. Research Assistance
- Quick analysis of research imaging data
- Preliminary screening of large datasets
- Hypothesis generation

### 3. Clinical Support (Non-Diagnostic)
- Assisting in image documentation
- Quick verification of image contents
- Training data annotation

---

## Supported Imaging Types

| Modality | Body Region | Example Questions |
|----------|-------------|-------------------|
| X-ray | Chest | "Is there cardiomegaly?" |
| CT | Head | "Is there evidence of hemorrhage?" |
| MRI | Brain | "What structure is abnormal?" |
| X-ray | Abdomen | "Are the kidneys visible?" |

---

## Limitations

> [!WARNING]
> **Not for Clinical Diagnosis**
> This model is for research/educational purposes only. Do not use for actual medical diagnosis.

- Accuracy varies by question complexity
- Open-ended questions have lower accuracy
- May not recognize rare conditions
- Performance depends on image quality

---

## Example Interactions

**Input Image**: Chest X-ray
```
Q: Is there any consolidation in the lungs?
A: no

Q: Is the heart size normal?
A: yes

Q: Which lung shows the abnormality?  
A: left lung
```

**Input Image**: Brain MRI
```
Q: Is this an axial image?
A: yes

Q: What structures are visible?
A: brain, ventricles
```
