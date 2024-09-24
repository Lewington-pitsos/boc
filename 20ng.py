import os
import time
import torch
import pandas as pd
import numpy as np
from transformer_lens import HookedTransformer
from sae_lens import SAE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
import json
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, BisectingKMeans, DBSCAN
from sklearn.decomposition import NMF  # Added for NMF
from sklearn.preprocessing import normalize  # Added for normalization

# Set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load your credentials (make sure you have a valid .credentials.json file with 'HF_TOKEN')
with open('.credentials.json') as f:
    creds = json.load(f)
os.environ['HF_TOKEN'] = creds['HF_TOKEN']

# Model configurations
release = 'gemma-scope-2b-pt-res'
sae_id = 'layer_19/width_65k/average_l0_21'
model_name = 'gemma-2-2b'
layer = 20
batch_size = 8

# Load the models
model = HookedTransformer.from_pretrained(model_name, device=device)
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release=release,
    sae_id=sae_id,
    device=device
)

# Function to generate concept indices
def generate_concept_indices(feature_acts, attention_mask, k=5):
    topk = torch.topk(feature_acts, k=k, dim=-1)  # [batch_size, seq_len, k]
    topk_indices = topk.indices
    attention_mask = attention_mask.bool()  # [batch_size, seq_len]
    batch_size, seq_len, k = topk_indices.shape
    concepts_list = []
    for i in range(batch_size):
        valid_positions = attention_mask[i].nonzero(as_tuple=False).squeeze()
        valid_topk_indices = topk_indices[i][valid_positions]  # [valid_seq_len, k]
        valid_topk_values = topk.values[i][valid_positions]  # [valid_seq_len, k]

        # Filter indices where the corresponding value is >= 0
        valid_indices = valid_topk_indices[valid_topk_values >= 0]

        concepts = valid_indices.flatten().cpu().numpy()
        concepts_list.append(concepts)
    return concepts_list

# Process documents and generate concept indices
def process_documents_to_concept_indices(texts, model, sae, layer, batch_size=8, k=5, device='cuda'):
    concept_indices_list = []
    tokenizer = model.tokenizer

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_start_time = time.time()
            batch = texts[i:i+batch_size]
            inputs = tokenizer(
                batch,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=90  # Adjust max_length for longer documents
            ).to(device)
            _, cache = model.run_with_cache(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                prepend_bos=True,
                stop_at_layer=layer
            )
            feature_acts = sae.encode(cache[sae.cfg.hook_name])
            batch_concept_indices = generate_concept_indices(
                feature_acts.cpu(),
                inputs.attention_mask.cpu(),
                k=k
            )
            concept_indices_list.extend(batch_concept_indices)
            batch_end_time = time.time()
            print(f"Processing batch {i+1} - {i+batch_size} in {batch_end_time - batch_start_time:.2f} seconds")
    return concept_indices_list

if __name__ == '__main__':
    # Load the 20 Newsgroups dataset
    newsgroups = fetch_20newsgroups(subset='all')
    documents = newsgroups.data
    labels = newsgroups.target

    print(f"Number of documents: {len(documents)}")

    # Sample 1% of all documents
    np.random.seed(42)
    idx = np.random.choice(len(documents), size=int(0.01 * len(documents)), replace=False)
    documents = [documents[i] for i in idx]
    labels = [labels[i] for i in idx]

    print(f"Number of documents (sampled): {len(documents)}")

    num_clusters = 20
    results = {}

    # -------------------- Clustering using TF-IDF --------------------

    print("\nClustering using TF-IDF")
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    X_tfidf = vectorizer.fit_transform(documents)

    # NMF clustering on TF-IDF vectors
    nmf_model_tfidf = NMF(n_components=num_clusters, random_state=42)
    W_tfidf = nmf_model_tfidf.fit_transform(X_tfidf)
    W_tfidf_normalized = normalize(W_tfidf, norm='l1', axis=1)
    labels_nmf_tfidf = np.argmax(W_tfidf_normalized, axis=1)

    # Evaluate NMF clustering
    ari_nmf_tfidf = adjusted_rand_score(labels, labels_nmf_tfidf)
    print(f"Adjusted Rand Index using NMF on TF-IDF: {ari_nmf_tfidf:.3f}")

    # Store result
    results['NMF_TF-IDF'] = ari_nmf_tfidf

    # -------------------- Clustering using Sentence Embeddings --------------------

    print("\nClustering using Sentence Embeddings")
    # Obtain Sentence Embeddings
    model_st = SentenceTransformer('all-MiniLM-L6-v2')
    X_embeddings = model_st.encode(documents, show_progress_bar=True)

    # KMeans clustering on Sentence Embeddings
    kmeans_emb = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans_emb.fit(X_embeddings)

    # Evaluate Embedding clustering
    ari_emb = adjusted_rand_score(labels, kmeans_emb.labels_)
    print(f"Adjusted Rand Index using Sentence Embeddings: {ari_emb:.3f}")

    # Store result
    results['sentence_embeddings'] = ari_emb

    # -------------------- Clustering using Bag of Concepts --------------------

    k_values = [16, 4, 1]
    for k in k_values:
        print(f"\nClustering using Bag of Concepts with k={k}")
        # Process documents
        doc_concept_indices_list = process_documents_to_concept_indices(
            documents, model, sae, layer, batch_size=batch_size, k=k, device=device
        )
        num_concepts = sae.cfg.d_sae
        # Build term-frequency matrices for documents
        data = []
        rows = []
        cols = []
        for doc_idx, concepts in enumerate(doc_concept_indices_list):
            counts = {}
            for concept in concepts:
                counts[concept] = counts.get(concept, 0) + 1
            for concept, count in counts.items():
                data.append(count)
                rows.append(doc_idx)
                cols.append(concept)
        num_docs = len(doc_concept_indices_list)
        doc_tf_matrix = csr_matrix((data, (rows, cols)), shape=(num_docs, num_concepts))

        # Compute TF-IDF for documents
        transformer = TfidfTransformer()
        doc_tfidf_matrix = transformer.fit_transform(doc_tf_matrix)

        # NMF clustering on Bag of Concepts TF-IDF
        X_concepts_tfidf = doc_tfidf_matrix
        nmf_model_concepts = NMF(n_components=num_clusters, random_state=42)
        W_concepts = nmf_model_concepts.fit_transform(X_concepts_tfidf)
        W_concepts_normalized = normalize(W_concepts, norm='l1', axis=1)
        labels_nmf_concepts = np.argmax(W_concepts_normalized, axis=1)
        ari_nmf_concepts = adjusted_rand_score(labels, labels_nmf_concepts)
        print(f"Adjusted Rand Index using NMF on Bag of Concepts (k={k}): {ari_nmf_concepts:.3f}")

        # Store results
        results[f'NMF_Bag_of_Concepts_k_{k}'] = ari_nmf_concepts

    # -------------------- Compare Results --------------------

    print("\nAdjusted Rand Index Scores:")
    for method, ari in results.items():
        print(f"{method}: {ari:.3f}")

    # Optional: Visualize clusters using t-SNE for Sentence Embeddings
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_embeddings)

