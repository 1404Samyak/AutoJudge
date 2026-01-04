import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import os
import nltk
import torch
import json
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from pykeen.models.unimodal.trans_e import TransE  
from torch.serialization import safe_globals  # for safe PyTorch load

nltk.download('punkt_tab')
nltk.download('punkt')

st.set_page_config(page_title="Problem Difficulty Predictor", layout="centered")
st.title("📘 Problem Difficulty Predictor-AutoJudge")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/
WORD2VEC_PATH = os.path.join(BASE_DIR, "word2vec_problem_solver.model")
TRANS_E_PATH = os.path.join(BASE_DIR, "transe_model.pt")
ENTITY_ID_PATH = os.path.join(BASE_DIR, "entity_to_id.json")

@st.cache_resource
def load_word2vec_model():
    return Word2Vec.load(WORD2VEC_PATH)
w2v_model = load_word2vec_model()
VECTOR_SIZE = w2v_model.vector_size

@st.cache_resource
def load_transE_model():
    with safe_globals([TransE]):  
        model = torch.load(TRANS_E_PATH, map_location='cpu', weights_only=False)
    with open(ENTITY_ID_PATH, 'r') as f:
        entity_to_id = json.load(f)
    return model, entity_to_id

transE_model, entity_to_id = load_transE_model()

def get_kg_embedding(entity_name):
    if entity_name in entity_to_id:
        idx = entity_to_id[entity_name]
        return transE_model.entity_representations[0](indices=torch.tensor([idx])).detach().cpu().numpy()[0]
    else:
        return [0] * 128

LETTER_SCORE = {ch: (i + 1) * 10 for i, ch in enumerate(string.ascii_uppercase)}
LETTER_SCORE["Z"] = 0

def parse_problem_index(idx):
    idx = str(idx).strip()
    match = re.search(r'([A-Z]+)(\d+)?$', idx)
    if match:
        letter = match.group(1)
        number = int(match.group(2)) if match.group(2) else 0
    else:
        letter = "Z"
        number = 0
    return letter, number

def index_to_score(idx):
    letter, number = parse_problem_index(idx)
    return LETTER_SCORE.get(letter, 0) + number

def extract_problem_id(url: str):
    pattern = r"/problem/(\d+)/([A-Z][A-Z0-9]*)"
    match = re.search(pattern, url)
    if match:
        return match.group(1) + match.group(2)
    return None

def get_mean_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

def text_length(text):
    return len(text.split())

def avg_sentence_length(text):
    sentences = re.split(r'[.!?]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) == 0:
        return 0
    sentence_lengths = [len(s.split()) for s in sentences]
    return np.mean(sentence_lengths)

FORMULA_SYMBOLS = r'[=<>+\-*/%^()]'
def formula_symbol_count(text):
    return len(re.findall(FORMULA_SYMBOLS, text))

def extract_text_features(text):
    return {
        "text_length": text_length(text),
        "avg_sentence_length": avg_sentence_length(text),
        "formula_symbol_count": formula_symbol_count(text)
    }

st.subheader("🔗 Problem Link")
problem_url = st.text_input(
    "Paste Codeforces problem link",
    placeholder="https://codeforces.com/problemset/problem/2181/D"
)
problem_id = None
index_score = None
if problem_url:
    problem_id = extract_problem_id(problem_url)
    if problem_id:
        st.success(f"Extracted Problem ID: **{problem_id}**")
        index_score = index_to_score(problem_id)
    else:
        st.error("Invalid Codeforces problem link")
st.divider()

st.subheader("⏱ Constraints")
time_limit = st.text_input("Time Limit (e.g. 1s, 2s)")
memory_limit = st.text_input("Memory Limit (e.g. 256MB)")
st.divider()

st.subheader("📝 Problem Statement")
description = st.text_area("Problem Description", height=150)
input_desc = st.text_area("Input Description", height=120)
output_desc = st.text_area("Output Description", height=120)
combined_text = " ".join([description.strip(), input_desc.strip(), output_desc.strip()])
st.divider()

st.subheader("🏷 Tags")
problem_tags = [
    "brute force", "implementation", "math", "dp", "matrices", "greedy",
    "sortings", "binary search", "flows", "graph matchings", "shortest paths",
    "combinatorics", "number theory", "geometry", "dfs and similar", "graphs",
    "trees", "constructive algorithms", "strings", "data structures",
    "divide and conquer", "games", "*special", "two pointers", "hashing",
    "dsu", "bitmasks", "expression parsing", "probabilities",
    "string suffix structures", "meet-in-the-middle", "fft", "2-sat",
    "interactive", "ternary search", "chinese remainder theorem", "schedules"
]

tags_input = st.text_input("Enter tags (comma separated)", placeholder="dp, greedy, graphs")
tag_vector = [0] * len(problem_tags)
tags = []
if tags_input:
    tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]
    tag_vector = [1 if tag in tags else 0 for tag in problem_tags]

if tags:
    st.success("0/1 Tag table successfully created")
st.divider()

# st.subheader("🧠 Word2Vec Embeddings")
tokens = word_tokenize(combined_text.lower())
embedding_vector = get_mean_vector(tokens, w2v_model)
embeddings_df = pd.DataFrame(
    [embedding_vector],
    columns=[f"emb_{i}" for i in range(len(embedding_vector))]
)

if combined_text.strip():
    st.success("Word2Vec embeddings successfully generated")
st.divider()

# st.subheader("🧬 KG Embeddings for Tags")
if tags:
    kg_embeddings = np.mean([get_kg_embedding(tag.replace(" ", "").lower()) for tag in tags], axis=0)
    st.success("KG embeddings successfully generated")
else:
    kg_embeddings = [0] * 128
kg_embeddings_df = pd.DataFrame(
    [kg_embeddings],
    columns=[f"kg_{i}" for i in range(128)]
)

other_features = {
    "id": problem_id,
    "index_score": index_score,
    "time_limit": time_limit,
    "memory_limit": memory_limit
}
other_df = pd.DataFrame([other_features])
for tag, value in zip(problem_tags, tag_vector):
    other_df[tag] = value

text_features_df = pd.DataFrame([extract_text_features(combined_text)])

st.subheader("🧪 Add Input/Output TestCases")
if "examples" not in st.session_state:
    st.session_state.examples = []

with st.form(key="example_form", clear_on_submit=True):
    example_input = st.text_area("Input", height=100, placeholder="Enter input for this example")
    example_output = st.text_area("Output", height=50, placeholder="Enter expected output")
    add_button = st.form_submit_button("➕ Add Example")

if add_button:
    if example_input.strip() == "" or example_output.strip() == "":
        st.warning("Please fill both Input and Output before adding.")
    else:
        st.session_state.examples.append({
            "input": example_input,
            "output": example_output
        })
        st.success(f"Example #{len(st.session_state.examples)} added!")

def extract_structural_features(example_list):
    all_ex_in_char_per_line = []
    all_ex_out_char_per_line = []
    all_ex_line_counts = []
    for ex in example_list:
        inp_str = str(ex.get('input', '')).strip()
        in_lines = inp_str.split('\n')
        in_chars_in_this_ex = sum(sum(len(c) for c in line.split()) for line in in_lines)
        out_str = str(ex.get('output', '')).strip()
        out_lines = out_str.split('\n')
        out_chars_in_this_ex = sum(sum(len(c) for c in line.split()) for line in out_lines)
        all_ex_in_char_per_line.append(in_chars_in_this_ex / len(in_lines) if in_lines else 0)
        all_ex_out_char_per_line.append(out_chars_in_this_ex / len(out_lines) if out_lines else 0)
        all_ex_line_counts.append(len(in_lines))
    return {
        "avg_input_chars_per_line": np.mean(all_ex_in_char_per_line),
        "avg_output_chars_per_line": np.mean(all_ex_out_char_per_line),
        "avg_input_lines_per_example": np.mean(all_ex_line_counts),
        "num_examples": len(example_list)
    }

if st.session_state.examples:
    structural_features = extract_structural_features(st.session_state.examples)
    # Rename keys to match expected X_test column names
    structural_features = {
        "avg_in_char_per_line": structural_features.get("avg_input_chars_per_line", 0),
        "avg_out_char_per_line": structural_features.get("avg_output_chars_per_line", 0),
        "avg_line_count": structural_features.get("avg_input_lines_per_example", 0),
        "sample_count": structural_features.get("num_examples", 0)
    }
else:
    structural_features = {
        "avg_in_char_per_line": 0,
        "avg_out_char_per_line": 0,
        "avg_line_count": 0,
        "sample_count": 0
    }

structural_df = pd.DataFrame([structural_features])

final_df = pd.concat(
    [
        other_df.reset_index(drop=True),
        text_features_df.reset_index(drop=True),
        embeddings_df.reset_index(drop=True),
        kg_embeddings_df.reset_index(drop=True),
        structural_df.reset_index(drop=True)
    ],
    axis=1
)

X_test = final_df.drop(columns=["id"])
X_test["time_limit"] = pd.to_numeric(X_test["time_limit"], errors="coerce")
X_test["memory_limit"] = pd.to_numeric(X_test["memory_limit"], errors="coerce")

column_order = ['time_limit', 'memory_limit', 'index_score', 'text_length', 'avg_sentence_length', 'formula_symbol_count', '*special', '2-sat', 'binary search', 'bitmasks', 'brute force', 'chinese remainder theorem', 'combinatorics', 'constructive algorithms', 'data structures', 'dfs and similar', 'divide and conquer', 'dp', 'dsu', 'expression parsing', 'fft', 'flows', 'games', 'geometry', 'graph matchings', 'graphs', 'greedy', 'hashing', 'implementation', 'interactive', 'math', 'matrices', 'meet-in-the-middle', 'number theory', 'probabilities', 'schedules', 'shortest paths', 'sortings', 'string suffix structures', 'strings', 'ternary search', 'trees', 'two pointers', 'avg_in_char_per_line', 'avg_out_char_per_line', 'avg_line_count', 'sample_count'] + \
               [f"emb_{i}" for i in range(300)] + \
               [f"kg_{i}" for i in range(128)]
final_column_order = [col for col in column_order if col in X_test.columns]
X_test = X_test[final_column_order]

st.subheader("📦 Final X_test Table")
st.dataframe(X_test)
st.info(f"Final X_test shape: {X_test.shape}")

import joblib
import lightgbm as lgb

CLASSIFICATION_MODEL_PATH = os.path.join(BASE_DIR, "classification_lgbm_model.pkl")
REGRESSION_MODEL_PATH = os.path.join(BASE_DIR, "regression_lgbm_model.pkl")  # regression LGBM
st.divider()
st.subheader("🎯 Difficulty Prediction")

@st.cache_resource
def load_lgbm_model(path):
    return joblib.load(path)

lgbm_class_model = load_lgbm_model(CLASSIFICATION_MODEL_PATH)
lgbm_reg_model = load_lgbm_model(REGRESSION_MODEL_PATH)

if st.button("Predict Difficulty"):
    # Classification prediction
    pred_class = lgbm_class_model.predict(X_test)[0]
    # Regression prediction
    pred_scaled = lgbm_reg_model.predict(X_test)[0]
    pred_scaled = np.clip(pred_scaled, 0.0, 1.0)
    pred_actual = pred_scaled * 10  # inverse min-max scaling
    st.success(f"Predicted Difficulty Class: **{pred_class}**")
    st.success(f"Predicted Difficulty Score: **{pred_actual:.2f} / 10**")


