import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import matplotlib.pyplot as plt
import ssl
import os
import certifi

# Ensure NLTK data is downloaded
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Global variables to store processed data
original_sentences = []
processed_sentences = []
tfidf_matrix = None
similarity_matrix = None

# Preprocess text
def preprocess_text(text):
    global original_sentences, processed_sentences
    original_sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    processed_sentences = []
    for sentence in original_sentences:
        words = word_tokenize(sentence)
        processed_sentence = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words and word not in string.punctuation]
        processed_sentences.append(' '.join(processed_sentence))
    return original_sentences, processed_sentences

# Vectorize text
def vectorize_text(processed_sentences):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(processed_sentences)
    return X, vectorizer

# Build similarity matrix
def build_similarity_matrix(sentences, tfidf_matrix):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0, 0]
    return similarity_matrix

# TextRank summarization
def textrank_summary(num_sentences=5):
    global original_sentences, processed_sentences, tfidf_matrix, similarity_matrix
    if len(processed_sentences) == 0:
        return "No content to summarize."
    similarity_matrix = build_similarity_matrix(processed_sentences, tfidf_matrix)
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(original_sentences)), reverse=True)
    summary = ' '.join([ranked_sentences[i][1] for i in range(min(num_sentences, len(ranked_sentences)))])
    return summary

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Get BERT sentence embedding
def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Calculate semantic similarity
def calculate_semantic_similarity(text1, text2):
    embedding1 = get_sentence_embedding(text1)
    embedding2 = get_sentence_embedding(text2)
    return cosine_similarity(embedding1, embedding2)[0, 0]

# Evaluate semantic summary
def evaluate_semantic_summary(generated_summary, reference_summary):
    return calculate_semantic_similarity(generated_summary, reference_summary)

# Calculate semantic metrics
def calculate_semantic_metrics(generated_summary, reference_summary):
    similarity_score = evaluate_semantic_summary(generated_summary, reference_summary)
    accuracy = similarity_score
    precision = similarity_score
    return accuracy, precision

# Load text file
def load_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            input_text.delete("1.0", tk.END)
            input_text.insert(tk.END, text)

# Preprocess text
def preprocess_action():
    global original_sentences, processed_sentences
    document = input_text.get("1.0", tk.END).strip()
    original_sentences, processed_sentences = preprocess_text(document)
    messagebox.showinfo("Info", "Text preprocessing completed.")

# Vectorize text
def vectorize_action():
    global tfidf_matrix
    tfidf_matrix, _ = vectorize_text(processed_sentences)
    messagebox.showinfo("Info", "Text vectorization completed.")

# Build similarity matrix
def build_similarity_matrix_action():
    global similarity_matrix
    similarity_matrix = build_similarity_matrix(processed_sentences, tfidf_matrix)
    messagebox.showinfo("Info", "Similarity matrix built.")

# Generate summary
def generate_summary_action():
    num_sentences = int(num_sentences_var.get())
    summary = textrank_summary(num_sentences)
    summary_text.delete("1.0", tk.END)
    summary_text.insert(tk.END, summary)

# Compare summaries
def compare_summaries_action():
    generated_summary = summary_text.get("1.0", tk.END).strip()
    reference_summary = reference_text.get("1.0", tk.END).strip()
    if not generated_summary or not reference_summary:
        messagebox.showerror("Error", "Both generated summary and reference summary are required for evaluation.")
        return
    accuracy, precision = calculate_semantic_metrics(generated_summary, reference_summary)
    messagebox.showinfo("Evaluation Scores", f"Accuracy: {accuracy}, Precision: {precision}")

    # Plotting
    metrics = ['Accuracy', 'Precision']
    values = [accuracy, precision]

    plt.bar(metrics, values, color=['blue', 'green'])
    plt.ylim(0, 1)
    plt.title('Performance Metrics')
    plt.show()
    
# Save summary to file
def save_summary():
    summary = summary_text.get("1.0", tk.END).strip()
    if not summary:
        messagebox.showerror("Error", "No summary to save.")
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
    if file_path:
        with open(file_path, 'w') as file:
            file.write(summary)
        messagebox.showinfo("Success", "Summary saved successfully.")

# Clear inputs
def clear_inputs():
    input_text.delete("1.0", tk.END)
    summary_text.delete("1.0", tk.END)
    reference_text.delete("1.0", tk.END)

# Exit application
def exit_application():
    root.quit()

# Create the main window
root = tk.Tk()
root.title("Snap Text")
root.geometry("1200x800")
root.configure(bg='orange')

# Configure styles
style = ttk.Style()
style.configure('TFrame', font=('Arial', 12), background='orange') 
style.configure('TLabel', font=('Arial', 12), background='orange')
style.configure('TButton', font=('Arial', 12), background='orange') 
style.configure('TEntry', font=('Arial', 12))
style.configure('TScrollbar', background='#ADD8E6')

# Create horizontal frames
button_frame = ttk.Frame(root, style='TFrame')
content_frame = ttk.Frame(root, style='TFrame')

button_frame.pack(side="left", fill="y", padx=10, pady=10)
content_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

# Buttons on the side
# Load file button
load_button = ttk.Button(button_frame, text="Load Text File", command=load_file)
load_button.pack(pady=5, fill='x')

# Preprocess text button
preprocess_button = ttk.Button(button_frame, text="Preprocess Text", command=preprocess_action)
preprocess_button.pack(pady=5, fill='x')

# Vectorize text button
vectorize_button = ttk.Button(button_frame, text="Vectorize Text", command=vectorize_action)
vectorize_button.pack(pady=5, fill='x')

# Build similarity matrix button
build_similarity_button = ttk.Button(button_frame, text="Build Similarity Matrix", command=build_similarity_matrix_action)
build_similarity_button.pack(pady=5, fill='x')

# Number of sentences
num_sentences_label = ttk.Label(button_frame, text="Number of Sentences in Summary:")
num_sentences_label.pack(pady=5, fill='x')
num_sentences_var = tk.StringVar(value="5")
num_sentences_entry = ttk.Entry(button_frame, textvariable=num_sentences_var, width=5)
num_sentences_entry.pack(pady=5, fill='x')

# Generate summary button
generate_button = ttk.Button(button_frame, text="Generate Summary", command=generate_summary_action)
generate_button.pack(pady=5, fill='x')

# Compare summaries button
compare_button = ttk.Button(button_frame, text="Compare Summaries", command=compare_summaries_action)
compare_button.pack(pady=5, fill='x')

# Save summary button
save_button = ttk.Button(button_frame, text="Save Summary", command=save_summary)
save_button.pack(pady=10, fill='x')

# Clear inputs button
clear_button = ttk.Button(button_frame, text="Clear", command=clear_inputs)
clear_button.pack(pady=10, fill='x')

# Exit button
exit_button = ttk.Button(button_frame, text="Exit", command=exit_application)
exit_button.pack(pady=10, fill='x')

# Input text
input_label = ttk.Label(content_frame, text="Input Text:")
input_label.pack(pady=5, fill='x')
input_text = scrolledtext.ScrolledText(content_frame, wrap=tk.WORD, width=80, height=10, font=('Arial', 12))
input_text.pack(pady=5, fill='both', expand=True)

# Summary text
summary_label = ttk.Label(content_frame, text="Generated Summary:")
summary_label.pack(pady=5, fill='x')
summary_text = scrolledtext.ScrolledText(content_frame, wrap=tk.WORD, width=80, height=10, font=('Arial', 12))
summary_text.pack(pady=5, fill='both', expand=True)

# Reference summary text
reference_label = ttk.Label(content_frame, text="Reference Summary (Optional for Evaluation):")
reference_label.pack(pady=5, fill='x')
reference_text = scrolledtext.ScrolledText(content_frame, wrap=tk.WORD, width=80, height=10, font=('Arial', 12))
reference_text.pack(pady=5, fill='both', expand=True)

# Run the application
root.mainloop()
