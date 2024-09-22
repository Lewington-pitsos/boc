import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

# Function to load documents from XML file
def load_documents(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        xml_data = f.read()
    # Wrap the xml data with a root element
    xml_data = '<root>' + xml_data + '</root>'
    root = ET.fromstring(xml_data)
    docs = []
    for doc in root.findall('doc'):
        docno = doc.find('docno').text.strip()
        text_elements = [elem.text.strip() for elem in doc if elem.tag != 'docno' and elem.text]
        text = ' '.join(text_elements)
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
    parent_dir = 'data/cranfield/'
    docs = load_documents(parent_dir + 'cran.all.1400.xml')
    queries = load_queries(parent_dir + 'cran.qry.xml')
    qrels = load_qrels(parent_dir + 'cranqrel.trec.txt')

    # Prepare documents for vectorization
    doc_texts = docs['text'].tolist()
    doc_ids = docs['docno'].tolist()

    # Prepare queries for vectorization
    query_texts = queries['title'].tolist()
    query_ids = queries['num'].tolist()

    # Vectorize documents and queries using TF-IDF
    vectorizer = TfidfVectorizer()
    # Fit on documents
    doc_vectors = vectorizer.fit_transform(doc_texts)
    # Transform queries using the same vectorizer
    query_vectors = vectorizer.transform(query_texts)

    # Compute similarity between queries and documents
    results = []
    for idx, query_vec in enumerate(query_vectors):
        query_id = query_ids[idx]
        query_text = queries.loc[idx, 'title']
        # Compute cosine similarity
        scores = cosine_similarity(query_vec, doc_vectors)[0]
        # Rank documents by similarity scores
        ranked_indices = np.argsort(scores)[::-1]
        ranked_doc_ids = [doc_ids[i] for i in ranked_indices]
        # Get relevancies from qrels
        relevant_docs = qrels[(qrels['topic'] == query_id) & (qrels['relevancy'] == 1)]['docno'].tolist()
        # Evaluate Precision at 5, 10, and 50
        precisions = {}
        ks = [1, 3, 5, 10]
        for k in ks:
            retrieved_docs = ranked_doc_ids[:k]
            retrieved_relevancies = [1 if doc_id in relevant_docs else 0 for doc_id in retrieved_docs]
            precision_at_k = sum(retrieved_relevancies) / k
            precisions[f'precision_at_{k}'] = precision_at_k
        # Get top-ranked document
        top_doc_id = ranked_doc_ids[0]
        # Find the index of the top document
        top_doc_idx = docs[docs['docno'] == top_doc_id].index[0]
        top_doc_text = docs.loc[top_doc_idx, 'text']
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
        results.append(result)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Compute average Precision at 5, 10, and 50

    for k in ks:
        avg_precision_at_k = results_df[f'precision_at_{k}'].mean()
        print(f'Average Precision at {k}: {avg_precision_at_k:.4f}')

    # Sort results by Precision at 10
    sorted_results = results_df.sort_values(by='precision_at_5', ascending=False)

    # Get top 10 and bottom 10 queries
    top_10 = sorted_results.head(10)
    bottom_10 = sorted_results.tail(10)

    # Convert to list of dictionaries
    top_10_list = top_10.to_dict(orient='records')
    bottom_10_list = bottom_10.to_dict(orient='records')

    # Save to JSON
    output = {'top_10_queries': top_10_list, 'bottom_10_queries': bottom_10_list}

    with open('queries_performance.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)