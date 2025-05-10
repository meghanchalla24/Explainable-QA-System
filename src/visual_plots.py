import json
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import pandas as pd
import nltk
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

nltk.download('punkt')


"""The below function stores the bert scores in a json file. We dont want to run it again and again so run once and save the scores"""
# from bert_score import score
# # Extract reference and candidate texts
# references = [item['answer'] for item in data]
# candidates = [item['generated_answer'] for item in data]

# # Compute BERTScores (use 'precision', 'recall', 'f1')
# P, R, F1 = score(candidates, references, lang='en', verbose=True)

# bert_scores = {
#     "f1": bert_f1_scores,
#     "precision": bert_P_scores,
#     "recall": bert_R_scores
# }

# with open("bert_scores.json", "w") as f:
#     json.dump(bert_scores, f)



# Load the updated JSON file
with open(r"data\output\bert_scores.json", "r") as f:
    bert_scores = json.load(f)

with open(r"D:\Information_retrieval_project\data\processed\testset_with_generated.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

f1_scores = bert_scores["f1"]
precision_scores = bert_scores["precision"]
recall_scores = bert_scores["recall"]


def display_BertRecallScore_Plot():
    st.title("BERTScore Recall Distribution")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="🔼 Maximum Recall", value=f"{max(recall_scores):.4f}")

    with col2:
        st.metric(label="🔽 Minimum Recall", value=f"{min(recall_scores):.4f}")

    with col3:
        st.metric(label="📊 Mean Recall", value=f"{sum(recall_scores) / len(recall_scores):.4f}")


    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(recall_scores, bins=15, kde=True, color='goldenrod', edgecolor='black', ax=ax)
    ax.set_title('BERTScore Recall Distribution', fontsize=15)
    ax.set_xlabel('BERTScore Recall', fontsize=13)
    ax.set_ylabel('Frequency', fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    st.pyplot(fig)


def display_BertF1Score_Plot():
    st.title("BERTScore F1 Distribution")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="🔼 Maximum F1 Score", value=f"{max(f1_scores):.4f}")

    with col2:
        st.metric(label="🔽 Minimum F1 Score", value=f"{min(f1_scores):.4f}")

    with col3:
        st.metric(label="📊 Mean F1 Score", value=f"{sum(f1_scores) / len(f1_scores):.4f}")



    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(f1_scores, bins=15, kde=True, color='mediumseagreen', edgecolor='black', ax=ax)
    ax.set_title('BERTScore F1 Distribution', fontsize=15)
    ax.set_xlabel('BERTScore F1', fontsize=13)
    ax.set_ylabel('Frequency', fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    st.pyplot(fig)



def display_BertPrecisionScore_Plot():
    st.title("BERTScore Precision Distribution")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="🔼 Maximum Precision", value=f"{max(precision_scores):.4f}")

    with col2:
        st.metric(label="🔽 Minimum Precision", value=f"{min(precision_scores):.4f}")

    with col3:
        st.metric(label="📊 Mean Precision", value=f"{sum(precision_scores) / len(precision_scores):.4f}")



    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(precision_scores, bins=15, kde=True, color='skyblue', edgecolor='black', ax=ax)
    ax.set_title('BERTScore Precision Distribution', fontsize=15)
    ax.set_xlabel('BERTScore Precision', fontsize=13)
    ax.set_ylabel('Frequency', fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    st.pyplot(fig)




def kdeplot():
    st.title("The BERT Score Density Plot")

    fig, ax = plt.subplots(figsize=(12, 7))

    # Precision KDE
    sns.kdeplot(precision_scores, fill=True, label='Precision', color='skyblue', linewidth=2, ax=ax)
    p_kde = sns.kdeplot(precision_scores, ax=ax).get_lines()[-1].get_data()
    p_x, p_y = p_kde
    p_peak_x = p_x[np.argmax(p_y)]
    p_peak_y = max(p_y)
    ax.axvline(p_peak_x, color='skyblue', linestyle='--', linewidth=1)
    ax.text(p_peak_x, p_peak_y + 0.05, f'Peak: {p_peak_x:.3f}', color='skyblue', fontsize=10, ha='center')

    # Recall KDE
    sns.kdeplot(recall_scores, fill=True, label='Recall', color='goldenrod', linewidth=2, ax=ax)
    r_kde = sns.kdeplot(recall_scores, ax=ax).get_lines()[-1].get_data()
    r_x, r_y = r_kde
    r_peak_x = r_x[np.argmax(r_y)]
    r_peak_y = max(r_y)
    ax.axvline(r_peak_x, color='goldenrod', linestyle='--', linewidth=1)
    ax.text(r_peak_x, r_peak_y + 0.05, f'Peak: {r_peak_x:.3f}', color='goldenrod', fontsize=10, ha='center')

    # F1 Score KDE
    sns.kdeplot(f1_scores, fill=True, label='F1 Score', color='mediumseagreen', linewidth=2, ax=ax)
    f_kde = sns.kdeplot(f1_scores, ax=ax).get_lines()[-1].get_data()
    f_x, f_y = f_kde
    f_peak_x = f_x[np.argmax(f_y)]
    f_peak_y = max(f_y)
    ax.axvline(f_peak_x, color='mediumseagreen', linestyle='--', linewidth=1)
    ax.text(f_peak_x, f_peak_y + 0.05, f'Peak: {f_peak_x:.3f}', color='mediumseagreen', fontsize=10, ha='center')

    # Final plot setup
    ax.set_title('BERTScore Distributions (Precision, Recall, F1)', fontsize=15)
    ax.set_xlabel('Score', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=12)
    fig.tight_layout()

    # Show the plot in Streamlit
    st.pyplot(fig)




def boxplot_metrics():
    """
    Display a boxplot of BERTScore Precision, Recall, and F1 distributions.
    """
    # Create DataFrame

    st.title("BERT Scores Box Plot")
    df = pd.DataFrame({
        'Precision': precision_scores,
        'Recall': recall_scores,
        'F1': f1_scores
    })

    # Melt to long format
    df_melted = df.melt(var_name='Metric', value_name='Score')

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='Metric', y='Score', data=df_melted,
                palette=['skyblue', 'goldenrod', 'mediumseagreen'], ax=ax)

    ax.set_title('BERTScore Metrics Distribution (Boxplot)', fontsize=15)
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()

    # Show in Streamlit
    st.pyplot(fig)




def plot_bertscore_all():

    # Ground Truth Answer Length
    st.title("BERT Score vs Ground Truth Answer Length")

    ref_lengths = [len(item['answer'].split()) for item in data]
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.scatter(ref_lengths, precision_scores, alpha=0.7, color='skyblue', edgecolors='black', label='Precision')
    ax2.scatter(ref_lengths, recall_scores, alpha=0.7, color='goldenrod', edgecolors='black', label='Recall')
    ax2.scatter(ref_lengths, f1_scores, alpha=0.7, color='red', edgecolors='black', label='F1 Score')
    ax2.set_title('BERTScore Metrics vs. Ground Truth Answer Length', fontsize=15)
    ax2.set_xlabel('Ground Truth Answer Length (in words)', fontsize=13)
    ax2.set_ylabel('BERTScore', fontsize=13)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=12)
    st.pyplot(fig2)


        # Generated Answer Length
    st.title("BERT Score vs Generated Answer Length")

    gen_lengths = [len(item['generated_answer'].split()) for item in data]
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(gen_lengths, precision_scores, alpha=0.7, color='skyblue', edgecolors='black', label='Precision')
    ax1.scatter(gen_lengths, recall_scores, alpha=0.7, color='goldenrod', edgecolors='black', label='Recall')
    ax1.scatter(gen_lengths, f1_scores, alpha=0.7, color='red', edgecolors='black', label='F1 Score')
    ax1.set_title('BERTScore Metrics vs. Generated Answer Length', fontsize=15)
    ax1.set_xlabel('Generated Answer Length (in words)', fontsize=13)
    ax1.set_ylabel('BERTScore', fontsize=13)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=12)
    st.pyplot(fig1)



# --------Rouge scores----------------

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

for item in data:
    reference = item['answer']
    generated = item['generated_answer']
    
    scores = scorer.score(reference, generated)
    
    rouge1_scores.append(scores['rouge1'].fmeasure)
    rouge2_scores.append(scores['rouge2'].fmeasure)
    rougeL_scores.append(scores['rougeL'].fmeasure)



def plot_rogue():

    st.title("The Rogue Score Density Plot")

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.kdeplot(rouge1_scores, fill=True, label='ROUGE-1', color='skyblue', linewidth=2, ax=ax)
    sns.kdeplot(rouge2_scores, fill=True, label='ROUGE-2', color='goldenrod', linewidth=2, ax=ax)
    sns.kdeplot(rougeL_scores, fill=True, label='ROUGE-L', color='mediumseagreen', linewidth=2, ax=ax)

    ax.set_title('ROUGE Score Distributions (KDE)', fontsize=15)
    ax.set_xlabel('Score', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    st.pyplot(fig)


def boxplot_rogue():
    """
    Plots a boxplot of ROUGE-1, ROUGE-2, and ROUGE-L scores using Streamlit.
    """
    st.title("The Rogue Scores Box Plot")

    # Create DataFrame
    df_rouge = pd.DataFrame({
        'ROUGE-1': rouge1_scores,
        'ROUGE-2': rouge2_scores,
        'ROUGE-L': rougeL_scores
    })

    # Convert to long format for seaborn boxplot
    df_rouge_melted = df_rouge.melt(var_name='Metric', value_name='Score')

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='Metric', y='Score', data=df_rouge_melted,
                palette=['skyblue', 'goldenrod', 'mediumseagreen'], ax=ax)

    ax.set_title('ROUGE Score Distributions (Boxplot)', fontsize=15)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    st.pyplot(fig)



def plot_rouge_vs_length():

    # Ground Truth Answer Length
    st.title("Rouge Score vs Ground Truth Answer Length")
    ref_lengths = [len(item['answer'].split()) for item in data]
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.scatter(ref_lengths, rouge1_scores, alpha=0.7, color='skyblue', edgecolors='black', label='ROUGE-1')
    ax2.scatter(ref_lengths, rouge2_scores, alpha=0.7, color='orange', edgecolors='black', label='ROUGE-2')
    ax2.scatter(ref_lengths, rougeL_scores, alpha=0.7, color='green', edgecolors='black', label='ROUGE-L')
    ax2.set_title('ROUGE Scores vs. Ground Truth Answer Length', fontsize=15)
    ax2.set_xlabel('Ground Truth Answer Length (in words)', fontsize=13)
    ax2.set_ylabel('ROUGE Score', fontsize=13)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=12)
    st.pyplot(fig2)


    # Generated Answer Length
    st.title("Rouge Score vs Generated Answer Length")
    gen_lengths = [len(item['generated_answer'].split()) for item in data]
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(gen_lengths, rouge1_scores, alpha=0.7, color='skyblue', edgecolors='black', label='ROUGE-1')
    ax1.scatter(gen_lengths, rouge2_scores, alpha=0.7, color='orange', edgecolors='black', label='ROUGE-2')
    ax1.scatter(gen_lengths, rougeL_scores, alpha=0.7, color='green', edgecolors='black', label='ROUGE-L')
    ax1.set_title('ROUGE Scores vs. Generated Answer Length', fontsize=15)
    ax1.set_xlabel('Generated Answer Length (in words)', fontsize=13)
    ax1.set_ylabel('ROUGE Score', fontsize=13)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=12)
    st.pyplot(fig1)


# -----------Blue Scores-------------------


bleu_scores = []
smoothing = SmoothingFunction().method1

for item in data:
    reference = [nltk.word_tokenize(item['answer'])]
    candidate = nltk.word_tokenize(item['generated_answer'])
    
    score = sentence_bleu(reference, candidate, smoothing_function=smoothing)
    bleu_scores.append(score)


def plot_bleu():
    # Ground Truth Answer Lengths
    st.title("BLEU Score vs Ground Truth Answer Length")
    ref_lengths = [len(item['answer'].split()) for item in data]
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(ref_lengths, bleu_scores, alpha=0.7, color='purple', edgecolors='black')
    ax1.set_title('BLEU Score vs. Ground Truth Answer Length', fontsize=15)
    ax1.set_xlabel('Ground Truth Answer Length (in words)', fontsize=13)
    ax1.set_ylabel('BLEU Score', fontsize=13)
    ax1.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig1)

    # Generated Answer Lengths
    st.title("BLEU Score vs Generated Answer Length")
    gen_lengths = [len(item['generated_answer'].split()) for item in data]
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.scatter(gen_lengths, bleu_scores, alpha=0.7, color='purple', edgecolors='black')
    ax2.set_title('BLEU Score vs. Generated Answer Length', fontsize=15)
    ax2.set_xlabel('Generated Answer Length (in words)', fontsize=13)
    ax2.set_ylabel('BLEU Score', fontsize=13)
    ax2.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig2)
