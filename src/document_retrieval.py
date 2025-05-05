from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
import faiss
import numpy as np
import json
import re
from rerankers import Reranker

nltk.download('punkt')

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# --- Preload and preprocess dense corpus ---
def minimal_clean(text): 
    text = text.lower()
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

with open("final_corrected_extracted_data.json", "r", encoding="utf-8") as file:
    corpus = json.load(file)

dense_pages, dense_page_metadata = [], {}
dense_metadata_keys = []

for doc_name, doc_data in corpus.items():
    for page in doc_data["extracted_text"]:
        text = minimal_clean(page["text"])
        page_num = page["page"]
        if text:
            dense_pages.append(text)
            dense_page_metadata[(doc_name, page_num)] = {
                "source_link": doc_data.get("source_link", ""),
                "local_path": doc_data.get("local_path", ""),
                "text": text
            }
            dense_metadata_keys.append((doc_name, page_num))

# Compute dense embeddings and build FAISS index
page_embeddings = model.encode(dense_pages, convert_to_numpy=True, show_progress_bar=True)
page_embeddings = page_embeddings / np.linalg.norm(page_embeddings, axis=1, keepdims=True)

dense_index = faiss.IndexFlatIP(page_embeddings.shape[1])
dense_index.add(page_embeddings)

# --- Hybrid Retrieval ---
def hybrid_retrieve(query, cleaned_extracted_data="cleaned_extracted_data.json"):

    with open(cleaned_extracted_data, "r", encoding="utf-8") as file:
        corpus = json.load(file)

    page_texts, page_metadata = [], {}
    for doc_name, doc_data in corpus.items():
        for page in doc_data.get("extracted_text", []):
            text = page.get("text", "").strip()
            page_num = page.get("page", 0)
            if text:
                page_texts.append(word_tokenize(text.lower()))
                page_metadata[(doc_name, page_num)] = {
                    "source_link": doc_data.get("source_link", ""),
                    "local_path": doc_data.get("local_path", ""),
                    "text": text
                }

    bm25 = BM25Okapi(page_texts)

    def bm25_retrieval(query, top_k=50):
        tokenized_query = word_tokenize(query.lower())
        if not tokenized_query:
            return []
        scores = bm25.get_scores(tokenized_query)
        min_score, max_score = min(scores), max(scores)
        normalized_scores = [(s - min_score) / (max_score - min_score) if max_score != min_score else 1.0 for s in scores]
        ranked_results = sorted(zip(page_metadata.keys(), normalized_scores), key=lambda x: x[1], reverse=True)
        return ranked_results[:top_k]

    def dense_retrieval(query, top_k=50):
        query_embedding = model.encode([minimal_clean(query)], convert_to_numpy=True)
        query_embedding /= np.linalg.norm(query_embedding)
        similarity_scores, indices = dense_index.search(query_embedding, top_k)
        results = [
            (dense_metadata_keys[idx], similarity_scores[0][i])
            for i, idx in enumerate(indices[0])
        ]
        return results

    def fuse_scores(bm25_results, dense_results, alpha=0.4, top_k=50):
        fused_scores = {}

        for (doc, page), score in bm25_results:
            fused_scores[(doc, page)] = alpha * score

        for (doc, page), score in dense_results:
            if (doc, page) in fused_scores:
                fused_scores[(doc, page)] += (1 - alpha) * score
            else:
                fused_scores[(doc, page)] = (1 - alpha) * score

        final_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        result_dict = {}
        for (doc_name, page_num), score in final_results:
            metadata = dense_page_metadata.get((doc_name, page_num), {})
            result_dict[f"{doc_name} - Page {page_num}"] = {
                "document_name": doc_name,
                "source_link": metadata.get("source_link", "N/A"),
                "local_link": metadata.get("local_path", "N/A"),
                "page_number": page_num,
                "text": metadata.get("text", "N/A"),
                "score": score
            }

        return result_dict

    bm25_results = bm25_retrieval(query)
    dense_results = dense_retrieval(query)
    hybrid_results = fuse_scores(bm25_results, dense_results)

    return hybrid_results


# --- Reranking ---
def reranking(query, hybrid_results):
    def rerank_results_with_flashrank(query, retrieved_results):
        ranker = Reranker('flashrank')

        docs = [doc_data["text"] for doc_data in retrieved_results.values()]
        doc_ids = list(retrieved_results.keys())

        k = 30
        docs = docs[:k]
        doc_ids = doc_ids[:k]
        retrieved_subset = {doc_id: retrieved_results[doc_id] for doc_id in doc_ids}

        ranked_results = ranker.rank(query=query, docs=docs, doc_ids=doc_ids)

        for result in ranked_results:
            doc_id = result.doc_id
            retrieved_subset[doc_id]["rerank_score"] = result.score

        reranked_results = dict(
            sorted(retrieved_subset.items(), key=lambda x: x[1]["rerank_score"], reverse=True)
        )

        return reranked_results

    return rerank_results_with_flashrank(query, hybrid_results)
