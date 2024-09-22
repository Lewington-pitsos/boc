import os
import torch
import pandas as pd
import numpy as np
from transformer_lens import HookedTransformer
from sae_lens import SAE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
import json
from scipy.sparse import csr_matrix

# Set up device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load credentials
with open('.credentials.json') as f:
    creds = json.load(f)
os.environ['HF_TOKEN'] = creds['HF_TOKEN']

# Load the data
df = pd.read_csv('data/questions.csv')
df = df.sample(frac=0.001, random_state=42).reset_index(drop=True)
print(df.shape)
print(df['is_duplicate'].value_counts())

# Combine 'question1' and 'question2' into a list of unique documents
docs = pd.concat([df['question1'], df['question2']]).unique().tolist()
print(f"Total unique documents: {len(docs)}")

# Load the model and SAE
sae_id = "blocks.8.hook_resid_pre"
release = "gpt2-small-res-jb"
model_name = "gpt2"

model = HookedTransformer.from_pretrained(model_name, device=device)

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release=release,
    sae_id=sae_id,
    device=device
)

# Function to generate concept indices per document
def generate_concept_indices(feature_acts, attention_mask, k=5):
    # feature_acts: [batch_size, seq_len, num_concepts]
    # attention_mask: [batch_size, seq_len]
    topk_indices = torch.topk(feature_acts, k=k, dim=-1).indices  # [batch_size, seq_len, k]
    # Mask out padding tokens
    attention_mask = attention_mask.bool()  # [batch_size, seq_len]
    batch_size, seq_len, k = topk_indices.shape
    concepts_list = []
    for i in range(batch_size):
        valid_positions = attention_mask[i].nonzero(as_tuple=False).squeeze()
        valid_topk = topk_indices[i][valid_positions]  # [valid_seq_len, k]
        concepts = valid_topk.flatten().cpu().numpy()
        concepts_list.append(concepts)
    return concepts_list

# Process documents and generate concept indices
def process_documents_to_concept_indices(docs, model, sae, batch_size=8, k=5, device='cuda'):
    concept_indices_list = []
    tokenizer = model.tokenizer

    with torch.no_grad():
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            inputs = tokenizer(
                batch,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=90
            ).to(device)
            _, cache = model.run_with_cache(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                prepend_bos=True,
                stop_at_layer=18
            )
            feature_acts = sae.encode(cache[sae.cfg.hook_name])
            batch_concept_indices = generate_concept_indices(
                feature_acts.cpu(),
                inputs.attention_mask.cpu(),
                k=k
            )
            concept_indices_list.extend(batch_concept_indices)
    return concept_indices_list

# Map documents to indices
doc_to_index = {doc: idx for idx, doc in enumerate(docs)}
df['q1idx'] = df['question1'].map(doc_to_index)
df['q2idx'] = df['question2'].map(doc_to_index)

# Function to compute results
def compute_results(df, concept_indices_list, num_concepts):
    mapping = dict(zip(df['q1idx'], df['q2idx']))
    q1_indices = list(mapping.keys())
    q2_indices = [mapping[q1] for q1 in q1_indices]

    # Build term-frequency matrix
    data = []
    rows = []
    cols = []
    for doc_idx, concepts in enumerate(concept_indices_list):
        counts = {}
        for concept in concepts:
            counts[concept] = counts.get(concept, 0) + 1
        for concept, count in counts.items():
            data.append(count)
            rows.append(doc_idx)
            cols.append(concept)
    num_docs = len(concept_indices_list)
    tf_matrix = csr_matrix((data, (rows, cols)), shape=(num_docs, num_concepts))

    # Compute TF-IDF
    transformer = TfidfTransformer()
    tfidf_matrix = transformer.fit_transform(tf_matrix)

    query_vectors = tfidf_matrix[q1_indices]
    n_queries = len(q1_indices)

    similarities = cosine_similarity(query_vectors, tfidf_matrix)
    correct_similarities = similarities[np.arange(n_queries), q2_indices]

    per_query_data = []

    for i in range(n_queries):
        q1_idx = q1_indices[i]
        q2_idx = q2_indices[i]
        q1_text = docs[q1_idx]
        q2_text = docs[q2_idx]
        sim = correct_similarities[i]
        per_query_data.append({
            'q1': q1_text,
            'q2': q2_text,
            'similarity': sim
        })

    return per_query_data

# Evaluate for different values of k
ks = [None, 1, 2]  # None for bag of words, 1 for bag of concepts
all_results = []
all_per_query_data = {}

for k in ks:
    if k is None:
        # Use original documents and TfidfVectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(docs)
        # Map documents to indices
        doc_to_index = {doc: idx for idx, doc in enumerate(docs)}
        df['q1idx'] = df['question1'].map(doc_to_index)
        df['q2idx'] = df['question2'].map(doc_to_index)
        mapping = dict(zip(df['q1idx'], df['q2idx']))
        q1_indices = list(mapping.keys())
        q2_indices = [mapping[q1] for q1 in q1_indices]
        query_vectors = tfidf_matrix[q1_indices]
        n_queries = len(q1_indices)
        similarities = cosine_similarity(query_vectors, tfidf_matrix)
        correct_similarities = similarities[np.arange(n_queries), q2_indices]
        per_query_data = []
        for i in range(n_queries):
            q1_idx = q1_indices[i]
            q2_idx = q2_indices[i]
            q1_text = docs[q1_idx]
            q2_text = docs[q2_idx]
            sim = correct_similarities[i]
            per_query_data.append({
                'q1': q1_text,
                'q2': q2_text,
                'similarity': sim
            })
        all_per_query_data['None'] = per_query_data
    else:
        # Process documents to get concept indices
        concept_indices_list = process_documents_to_concept_indices(
            docs, model, sae, batch_size=64, k=k, device=device
        )
        num_concepts = sae.cfg.d_sae
        per_query_data = compute_results(df, concept_indices_list, num_concepts)
        all_per_query_data[k] = per_query_data

# Store highest and lowest scoring question pairs for each k
results_dict = {}

for k in ks:
    key = 'None' if k is None else k
    per_query_data = all_per_query_data[key]
    sorted_data = sorted(per_query_data, key=lambda x: x['similarity'])
    lowest = sorted_data[:20]
    highest = sorted_data[-20:]
    results_dict[key] = {
        'highest': highest,
        'lowest': lowest
    }

# Optionally, save the results_dict to a JSON file
with open('results.json', 'w') as f:
    json.dump(results_dict, f, indent=4)

# Plot the results (if you still want to plot average scores)
# Note: Since we modified the compute_results function, you may need to adjust the plotting code accordingly.
