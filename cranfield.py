import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
        # Compute cosine similarity
        scores = cosine_similarity(query_vec, doc_vectors)[0]
        # Rank documents by similarity scores
        ranked_indices = np.argsort(scores)[::-1]
        ranked_doc_ids = [doc_ids[i] for i in ranked_indices]
        # Get relevancies from qrels
        relevant_docs = qrels[(qrels['topic'] == query_id) & (qrels['relevancy'] == 1)]['docno'].tolist()
        # Evaluate Precision at 5, 10, and 50
        precisions = {}
        for k in [5, 10, 50]:
            retrieved_docs = ranked_doc_ids[:k]
            retrieved_relevancies = [1 if doc_id in relevant_docs else 0 for doc_id in retrieved_docs]
            precision_at_k = sum(retrieved_relevancies) / k
            precisions[f'precision_at_{k}'] = precision_at_k
        # Store results
        result = {'query_id': query_id}
        result.update(precisions)
        results.append(result)

    # Compute average Precision at 5, 10, and 50
    avg_precision_at_5 = sum([res['precision_at_5'] for res in results]) / len(results)
    avg_precision_at_10 = sum([res['precision_at_10'] for res in results]) / len(results)
    avg_precision_at_50 = sum([res['precision_at_50'] for res in results]) / len(results)

    print(f'Average Precision at 5: {avg_precision_at_5:.4f}')
    print(f'Average Precision at 10: {avg_precision_at_10:.4f}')
    print(f'Average Precision at 50: {avg_precision_at_50:.4f}')