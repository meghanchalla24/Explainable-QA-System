import streamlit as st
from visual_plots import display_BertRecallScore_Plot,display_BertF1Score_Plot,display_BertPrecisionScore_Plot,kdeplot,boxplot_metrics,plot_bertscore_all,plot_rogue,boxplot_rogue,plot_rouge_vs_length,plot_bleu
from generator import generate_answer
from document_retrieval import hybrid_retrieve,reranking

from documents_statistics import get_statistics

from boolean_retrieval import build_inverted_index,evaluate_query
import json
import time

import os
os.environ["TOGETHER_API_KEY"] = "ce142861c7d89ac010c16d12d7da1ca855a95f7cb4bc405d799c8dcca06a40b5"


# Sidebar for navigation
st.sidebar.title("Navigation")
tab = st.sidebar.radio("Select a tab", ("Query Search", "View Statistics", "View Plots","Boolean Retrieval"))


# Content for Tab 1
if tab == "Query Search":
    st.title("Question Answering Assistant")
    st.write("Enter your query below and get the answer based on Institute Regulations.")

    # Input field
    query = st.text_input("🔎 Query", placeholder="e.g., What are the guidelines for availing casual leaves by a PhD student?")

    # Always show the button
    run_button = st.button("Run")

    # Only run logic when query is present AND button clicked
    if run_button:
        if query.strip() == "":
            st.warning("⚠️ Please enter a query before running.")
        else:
            with st.spinner("Retrieving documents..."):
                hybrid_results = hybrid_retrieve(query)
                time.sleep(1)  # optional artificial delay

            with st.spinner("Reranking retrieved documents..."):
                reranked_results = reranking(query, hybrid_results)
                print("✅ Documents have been reranked, now generating final answer...")

            top_answers = [doc_data["text"] for _, doc_data in list(reranked_results.items())[:5]]

            with st.spinner("Generating final answer..."):
                final_answer = generate_answer(query, top_answers)

            st.subheader("✅ Final Answer")
            st.write(final_answer)

            st.markdown("---")
            st.subheader("📚 References")
            
            # Displaying document details in the reference section
            for rank, (doc_id, doc_data) in enumerate(list(reranked_results.items())[:5], start=1):
                st.markdown(f"**{rank}.** {doc_data['document_name']} - Page {doc_data['page_number']}")
                #(Score: {doc_data['rerank_score']:.4f})
                st.markdown(f"   📄 Source: {doc_data['source_link']}")
                st.markdown(f"   📂 Local Path: {doc_data['local_link']}")
                st.markdown(f"   📝 Excerpt: {doc_data['text'][:300]}...")  # Excerpt first 300 characters (or adjust as needed)
                st.markdown("___")

elif tab == "View Statistics":
    st.title("Document Statistics Before Prepocessing")
    get_statistics("extracted_data_with_links.json")

    st.title("Document Statistics After Prepocessing")
    get_statistics("cleaned_extracted_data.json")

elif tab=="View Plots":
    display_BertRecallScore_Plot()
    display_BertPrecisionScore_Plot()
    display_BertF1Score_Plot()
    kdeplot()
    boxplot_metrics()
    plot_bertscore_all()
    plot_rogue()
    boxplot_rogue()
    plot_rouge_vs_length()
    plot_bleu()

elif tab == "Boolean Retrieval":
    st.title("Boolean Retrieval over Institute Regulations")

    # Load cleaned PDF data
    with open("cleaned_extracted_data.json", "r", encoding="utf-8") as f:
        pdf_data = json.load(f)

    # Build inverted index
    inverted_index, all_doc_ids = build_inverted_index(pdf_data)

    # Input query
    query = st.text_input("Enter Boolean Query (e.g., fellowship AND NOT regulation)")

    if st.button("Search"):
        if not query.strip():
            st.warning("Please enter a valid query.")
        else:
            results = evaluate_query(query, inverted_index, all_doc_ids)

            if not results:
                st.warning("No results found for this query.")
            else:
                st.success(f"Found {len(results)} matching pages.")
                for pdf_file, page_num in results:
                    link = pdf_data.get(pdf_file, {}).get("source_link", "Link not available")
                    st.markdown(f"**{pdf_file} - Page {page_num}**  \n[🔗 Source Link]({link})", unsafe_allow_html=True)

