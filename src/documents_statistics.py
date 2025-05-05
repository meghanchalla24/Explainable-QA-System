import json
from collections import Counter
import streamlit as st
import pandas as pd

def get_statistics(input_file):
    """Compute and display statistics for PDFs before preprocessing (Streamlit version)."""

    # Load extracted data
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            pdf_data = json.load(f)
    except Exception as e:
        st.error(f"Error loading JSON file: {e}")
        return

    # Initialize statistics
    total_pdfs = len(pdf_data)
    scanned_pdfs = sum(1 for data in pdf_data.values() if data["is_scanned"])
    normal_pdfs = total_pdfs - scanned_pdfs
    total_words_per_pdf = {}
    total_pages_per_pdf = {}
    all_words = []

    for pdf_file, data in pdf_data.items():
        total_words = 0
        extracted_text = data.get("extracted_text", [])

        if isinstance(extracted_text, str):
            extracted_text = [{"page": 1, "text": extracted_text}]

        total_pages = len(extracted_text)

        for page in extracted_text:
            if isinstance(page, dict) and "text" in page:
                words = page["text"].split()
                total_words += len(words)
                all_words.extend(words)

        total_words_per_pdf[pdf_file] = total_words
        total_pages_per_pdf[pdf_file] = total_pages

    avg_words_per_pdf = sum(total_words_per_pdf.values()) / total_pdfs if total_pdfs else 0
    avg_words_per_page = sum(total_words_per_pdf.values()) / sum(total_pages_per_pdf.values()) if total_pages_per_pdf else 0
    total_pages_all_pdfs = sum(total_pages_per_pdf.values())
    total_words_all_pdfs = sum(total_words_per_pdf.values())

    most_text_pdf = max(total_words_per_pdf, key=total_words_per_pdf.get, default="N/A")
    most_pages_pdf = max(total_pages_per_pdf, key=total_pages_per_pdf.get, default="N/A")

    common_words = Counter(all_words).most_common(10)
    freq_sum = sum(freq for _, freq in common_words)

    # --- Streamlit Display ---
    st.subheader("📊 Basic Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total PDFs", total_pdfs)
    col2.metric("Scanned PDFs", scanned_pdfs)
    col3.metric("Normal PDFs", normal_pdfs)

    col4, col5, col6 = st.columns(3)
    col4.metric("Total Pages", total_pages_all_pdfs)
    col5.metric("Total Words", total_words_all_pdfs)
    col6.metric("Avg Words/Page", f"{avg_words_per_page:.2f}")

    st.divider()

    st.subheader("📄 Page-Level Analysis")
    st.write(f"**PDF with the most text**: `{most_text_pdf}` ({total_words_per_pdf.get(most_text_pdf, 0)} words)")
    st.write(f"**PDF with the most pages**: `{most_pages_pdf}` ({total_pages_per_pdf.get(most_pages_pdf, 0)} pages)")

    st.divider()

    st.subheader("📝 Most Frequent Words")
    df_freq = pd.DataFrame(common_words, columns=["Word", "Frequency"])
    st.dataframe(df_freq, use_container_width=True)
    st.caption(f"Sum of top-10 frequent word occurrences: **{freq_sum}**")

