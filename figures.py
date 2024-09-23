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
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer

# Set up device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load credentials
with open('.credentials.json') as f:
    creds = json.load(f)
os.environ['HF_TOKEN'] = creds['HF_TOKEN']

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
def process_documents_to_concept_indices(texts, model, sae, batch_size=8, k=5, device='cuda'):
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
                max_length=512  # Adjust max_length for longer documents
            ).to(device)
            _, cache = model.run_with_cache(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                prepend_bos=True,
                stop_at_layer=9
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

# Function to load documents from XML file
def load_documents(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        xml_data = f.read()
    root = ET.fromstring(xml_data)
    docs = []
    for doc in root.findall('doc'):
        docno = doc.find('docno').text.strip()

        text = doc.find('text').text.strip()
        docs.append({'docno': docno, 'text': text})
    return pd.DataFrame(docs)

# Function to load queries from XML file
def load_queries(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    queries = []
    for top in root.findall('top'):
        num = top.find('num').text.strip()
        title = top.find('title').text.strip()
        queries.append({'num': num, 'title': title})
    return pd.DataFrame(queries)

# Function to load relevance judgments
def load_qrels(file_path):
    qrels = pd.read_csv(
        file_path, 
        delim_whitespace=True, 
        names=['topic', 'iter', 'docno', 'relevancy']
    )
    qrels = qrels.astype({'topic': str, 'docno': str})
    return qrels

if __name__ == '__main__':
    if not os.path.exists('cruft'):
        os.makedirs('cruft')

    parent_dir = 'data/figures/'
    docs_df = load_documents(parent_dir + 'all.xml')
    queries_df = load_queries(parent_dir + 'qry.xml')
    qrels = load_qrels(parent_dir + 'trec.txt')

    # Prepare documents and queries
    doc_texts = docs_df['text'].tolist()
    doc_ids = docs_df['docno'].tolist()
    query_texts = queries_df['title'].tolist()
    query_ids = queries_df['num'].tolist()

    # Map doc IDs to indices
    doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    docs_df['index'] = docs_df.index
    queries_df['index'] = queries_df.index

    # Process documents and queries to get concept indices
    k_values = [None, 8, 4, 2, 1]  # None for bag of words, others for bag of concepts
    results = {}

    batch_size = 32

    for k in k_values:
        if k is None:
            # Bag of words approach
            vectorizer = TfidfVectorizer()
            doc_vectors = vectorizer.fit_transform(doc_texts)
            query_vectors = vectorizer.transform(query_texts)
        else:
            # Bag of concepts approach
            if not os.path.exists(f'cruft/fig_doc_tfidf_matrix_k_{k}.npy'):                
                print(f"Processing documents with k={k}")
                # Process documents
                doc_concept_indices_list = process_documents_to_concept_indices(
                    doc_texts, model, sae, batch_size=batch_size, k=k, device=device
                )
                # Process queries
                print(f"Processing queries with k={k}")
                query_concept_indices_list = process_documents_to_concept_indices(
                    query_texts, model, sae, batch_size=batch_size, k=k, device=device
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

                # Build term-frequency matrices for queries
                data = []
                rows = []
                cols = []
                for query_idx, concepts in enumerate(query_concept_indices_list):
                    counts = {}
                    for concept in concepts:
                        counts[concept] = counts.get(concept, 0) + 1
                    for concept, count in counts.items():
                        data.append(count)
                        rows.append(query_idx)
                        cols.append(concept)
                num_queries = len(query_concept_indices_list)
                query_tf_matrix = csr_matrix((data, (rows, cols)), shape=(num_queries, num_concepts))

                # Compute TF-IDF for documents and queries
                transformer = TfidfTransformer()
                doc_tfidf_matrix = transformer.fit_transform(doc_tf_matrix)
                query_tfidf_matrix = transformer.transform(query_tf_matrix)

                # save the matrices
                np.save(f'cruft/fig_doc_tfidf_matrix_k_{k}.npy', doc_tfidf_matrix.toarray())
                np.save(f'cruft/fig_query_tfidf_matrix_k_{k}.npy', query_tfidf_matrix.toarray())
            else:    
                doc_tfidf_matrix = np.load(f'cruft/fig_doc_tfidf_matrix_k_{k}.npy')
                query_tfidf_matrix = np.load(f'cruft/fig_query_tfidf_matrix_k_{k}.npy')

        # Compute similarity between queries and documents
        if k is None:
            # Using bag of words vectors
            similarities = cosine_similarity(query_vectors, doc_vectors)
        else:
            # Using bag of concepts vectors
            similarities = cosine_similarity(query_tfidf_matrix, doc_tfidf_matrix)

        # Evaluate performance
        all_results = []
        ns = [1, 3, 5, 10]
        for idx, query_id in enumerate(query_ids):
            query_text = queries_df.loc[idx, 'title']
            scores = similarities[idx]
            # Rank documents by similarity scores
            ranked_indices = np.argsort(scores)[::-1]
            ranked_doc_ids = [doc_ids[i] for i in ranked_indices]
            # Get relevancies from qrels
            relevant_docs = qrels[(qrels['topic'] == query_id) & (qrels['relevancy'] == 1)]['docno'].tolist()
            # Evaluate Precision at k
            precisions = {}
            for n_eval in ns:
                retrieved_docs = ranked_doc_ids[:n_eval]
                retrieved_relevancies = [1 if doc_id in relevant_docs else 0 for doc_id in retrieved_docs]
                precision_at_k = sum(retrieved_relevancies) / n_eval
                precisions[f'precision_at_{n_eval}'] = precision_at_k
            # Get top-ranked document
            top_doc_id = ranked_doc_ids[0]
            # Find the index of the top document
            top_doc_idx = docs_df[docs_df['docno'] == top_doc_id].index[0]
            top_doc_text = docs_df.loc[top_doc_idx, 'text']
            # Optional: Limit the length of the document text for readability
            max_text_length = 500  # Adjust as needed
            if len(top_doc_text) > max_text_length:
                top_doc_text = top_doc_text[:max_text_length] + '...'
            # Store results
            result = {
                'query_id': query_id,
                'query_text': query_text,
                'top_doc_id': top_doc_id,
                'top_doc_text': top_doc_text
            }
            result.update(precisions)
            all_results.append(result)

        # Convert results to DataFrame
        results_df = pd.DataFrame(all_results)

        # Compute average Precision at ks
        avg_precisions = {}
        for n_eval in ns:
            avg_precision_at_k = results_df[f'precision_at_{n_eval}'].mean()
            avg_precisions[f'avg_precision_at_{n_eval}'] = avg_precision_at_k
            print(f'Average Precision at {n_eval} for k={k}: {avg_precision_at_k:.4f}')

        # Store results
        results_key = 'bag_of_words' if k is None else f'bag_of_concepts_k_{k}'
        results[results_key] = {
            'results_df': results_df,
            'avg_precisions': avg_precisions
        }

        # Sort results by Precision at 5
        sorted_results = results_df.sort_values(by='precision_at_5', ascending=False)

        # Get top 10 and bottom 10 queries
        top_10 = sorted_results.head(10)
        bottom_10 = sorted_results.tail(10)

        # Convert to list of dictionaries
        top_10_list = top_10.to_dict(orient='records')
        bottom_10_list = bottom_10.to_dict(orient='records')

        # Save to JSON
        output = {
            'average_precisions': avg_precisions,
            'top_10_queries': top_10_list,
            'bottom_10_queries': bottom_10_list
        }

        output_filename = f'cruft/fig_queries_performance_{results_key}.json'
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

    # Optionally, compare the average precisions across different methods
    comparison_df = pd.DataFrame({
        method: pd.Series(data['avg_precisions']) for method, data in results.items()
    })
    print("\nComparison of Average Precisions:")
    print(comparison_df)

    # Save comparison to JSON
    comparison_df.to_json('cruft/fig_average_precisions_comparison.json', orient='index', indent=4)
