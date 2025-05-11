import os
import sys
import nltk
sys.path.append(os.path.dirname(__file__)) 
import streamlit as st
from visual_plots import display_BertRecallScore_Plot,display_BertF1Score_Plot,display_BertPrecisionScore_Plot,kdeplot,boxplot_metrics,plot_bertscore_all,plot_rogue,boxplot_rogue,plot_rouge_vs_length,plot_bleu
from generator import generate_answer
from document_retrieval import hybrid_retrieve,reranking

from documents_statistics import get_statistics


import json
import time

os.environ["TOGETHER_API_KEY"] = "410dde8215d41aae1af97f1c1424c6d32e2b0bb3accc1b6b2d700bdb0e2cb830"


# Sidebar for navigation
st.sidebar.title("Navigation")
tab = st.sidebar.radio("Select a tab", ("Query Search", "View Statistics", "View Plots"))


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
                hybrid_results = hybrid_retrieve(query) # either change the default file location in the 'hybrid_retrieve' function or pass on your desired file location as an extra parameter
                
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
                #st.markdown(f"   📂 Local Path: {doc_data['local_link']}") -- > this is helpful when running on local host
                st.markdown(f"   📝 Excerpt: {doc_data['text'][:300]}...")  # Excerpt first 300 characters (or adjust as needed)
                st.markdown("___")

elif tab == "View Statistics":
    st.title("Document Statistics Before Prepocessing")
    get_statistics(r"data/raw/raw_json/extracted_data_with_links.json") # change your path accordingly

    st.title("Document Statistics After Prepocessing")
    get_statistics(r"data/processed/cleaned_extracted_data.json") # change your path accordingly

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


