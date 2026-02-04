# Top 100 most common answers from PathVQA dataset analysis
TOP_100_ANSWERS = [
    'yes', 'no', 'oral', 'cardiovascular', 'hematologic',
    'gastrointestinal', 'endocrine', 'lung', 'hepatobiliary',
    'nervous', 'female reproductive', 'vasculature', 'urinary',
    'liver', 'spleen', 'heart', 'gastrointestinal system',
    'respiratory', 'endocrine system', 'brain', 'skin',
    'extremities', 'female reproductive system', 'joints',
    'kidney', 'abdomen', 'pancreas', 'breast', 'lymph node',
    'adrenal', 'male reproductive', 'skeletal muscle',
    'musculoskeletal', 'thyroid', 'prostate', 'bone',
    'colon', 'small intestine', 'gallbladder', 'bone marrow',
    'blood', 'esophagus', 'ovary', 'testis', 'uterus',
    'stomach', 'thymus', 'adrenal gland', 'pituitary',
    'parathyroid', 'urinary system', 'male reproductive system',
    'upper respiratory', 'lower respiratory', 'anterior',
    'posterior', 'medial', 'lateral', 'superior', 'inferior',
    'proximal', 'distal', 'central', 'peripheral', 'axial',
    'appendicular', 'cranial', 'caudal', 'dorsal', 'ventral',
    'superficial', 'deep', 'external', 'internal', 'left',
    'right', 'bilateral', 'unilateral', 'ipsilateral',
    'contralateral', 'benign', 'malignant', 'metastatic',
    'primary', 'secondary', 'acute', 'chronic', 'inflammatory',
    'neoplastic', 'degenerative', 'congenital', 'acquired',
    'infectious', 'autoimmune', 'vascular', 'ischemic',
    'hemorrhagic', 'thrombotic', 'embolic', 'normal', 'abnormal'
]

# Create answer to index mapping
ANSWER_TO_IDX = {ans: idx for idx, ans in enumerate(TOP_100_ANSWERS)}
IDX_TO_ANSWER = {idx: ans for idx, ans in enumerate(TOP_100_ANSWERS)}
