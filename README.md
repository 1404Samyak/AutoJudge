---
title: AutoJudge
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: Streamlit template space
---

# 🚀 AutoJudge: Programming Problems Difficulty Prediction Model 

## 📌 Project Overview
**AutoJudge** is an advanced **Machine Learning–driven evaluation framework** built to automatically analyze and assess complex technical problems. It leverages multi-task learning to provide both qualitative and quantitative difficulty insights from a single input.

- 🧩 **Categorical Classification** (Easy / Medium / Hard) : Automatically assigns each problem to a difficulty class — **Easy, Medium, or Hard** — enabling quick and intuitive understanding of problem complexity.
- 📈 **Continuous Difficulty Regression** (Score from 0–10) : Predicts a fine-grained numerical difficulty score ranging from **0 to 10**, offering a more precise and scalable measure of problem difficulty beyond fixed labels.

- By combining these two perspectives, AutoJudge delivers a comprehensive difficulty assessment, making it suitable for **competitive programming** analysis, problem curation etc 

---
### 🎬 Demo

- 🎥 Demo Video: https://drive.google.com/file/d/1y4fTTMJjjG8LWANeRBlDzk7Bkbvf-R9r/view?usp=sharing

- 🌐 Live App (Hugging Face Space):
https://huggingface.co/spaces/1404Samyak/AutoJudge


### 📚 Resources

- 📓 Google Colab Notebook: https://colab.research.google.com/drive/1mirwFBvUqevVA2tQdQL_MZQ0ybJEvThT?usp=sharing
- Kindly click on the this link to view the Colab notebook directly.
- Some cell outputs and execution logs are not visible in the GitHub source code preview, so please use the link to access the full notebook on Google Colab, where all cell outputs are properly displayed.
- The streamlit code is among the files with the name streamlit_app.py
- 📊 Dataset Link: https://huggingface.co/datasets/open-r1/codeforces

---

## 👨‍🔬 Developer Details
- **Name:** Samyak Mahapatra  
- **Degree:** B.Tech in Electronics and Communication Engineering (ECE)  
- **Institute:** Indian Institute of Technology (IIT), Roorkee  
- **Enrollment No:** 23116087  

---

## 🛠️ System Architecture & Methodology

AutoJudge follows a **four-stage modular pipeline**, ensuring robustness and scalability.The four stages are as follows 

### 1️⃣ Feature Extraction & Dense Features
Structural and statistical metadata are extracted directly from the problem content to capture intrinsic complexity.

**Key Dense Features:**
- ***Statistical Metrics:*** `sample_count`, `time_limit`, `memory_limit`, `text_length`,`avg_in_char`,`avg_out_char`,`avg_line`,`sample_count`
  
- ***Complexity Indicators:*** `index_score`, `avg_line_count`, `avg_sentence_length`, `formula_symbol_count`
  
- ***Algorithmic Tags:*** One-hot encoded tags such as `dp`, `greedy`, `math`, `graphs`, `geometry`, and `number theory`


### 2️⃣ Textual Embeddings (Word2Vec)
To capture semantic meaning from problem statements:

- The ***NLTK*** library was used to split each problem statement into smaller units called tokens (words). This step helps the model understand the text word by word instead of treating the whole sentence as a single string.
  
- Instead of using a pre-trained model, a **Word2Vec model** was trained on the dataset. This allows the embeddings to better capture competitive programming–specific terms, problem language, and patterns present in the dataset. Embeddings generated **300-dimensional embeddings** (`emb_0` → `emb_299`) to capture semantic menaing of problem,input and output description.
  
- Since each problem contains many words, the embeddings of all words are averaged using mean pooling. This creates **one fixed-size vector** that represents the overall meaning of the entire problem statement, making it easy to use as input for machine learning models.


### 3️⃣ Knowledge Graph Modeling (TransE)
To encode relationships between algorithmic concepts:

- ***Knowlege Graph Construction*** : A **Knowledge Graph (KG)** was built to represent relationships between **algorithmic tags and technical concepts** (such as DP, graphs, binary search, constraints, etc.). Each concept is treated as a **node**, and meaningful relationships between them are stored as **edges**.
  
- ***Graph Relationship Pipeline*** : Problem metadata (tags, categories, limits, and concepts) is first processed and converted into **(head, relation, tail)** triplets. These triplets define how different concepts are connected and form the input training data for the knowledge graph model.
  
- ***TransE Model for Graph Learning*** : The TransE (Translation Embedding) model is used to learn vector representations of graph entities. It works by learning embeddings such that **head + relation ≈ tail**,allowing the model to capture how different algorithmic concepts are related to each other.The model is trained using **Margin Ranking Loss**, which helps the model distinguish correct relationships from incorrect (negative) ones. This ensures that valid concept relationships are placed closer together in the embedding space.
  
- ***KG Embedding Generation*** : After training, each concept is represented as a **128-dimensional vector**. These vectors (kg_0 → kg_127) act as structured knowledge features and are appended to the problem’s feature set before being passed to the final prediction models.


### 4️⃣ Model Selection & Training
Multiple ***Machine Learning*** models were evaluated before final selection such as LightGBM, CatBoost, Logistic/Linear Regression, SVM, RandomForest, XGBoost, GradientBoost, AdaBoost

- **Classification Model:** LightGBM came out to have best overall metrics
- **Regression Model:** CatBoost had best overall metrics but very close to overall metrics of LGBMRegressor

These models were chosen based on performance, stability, and generalization on the test data 

---

## 📈 Model Performance

### 🔹 Classification Results (LightGBM)
- **Overall Accuracy:** **77.09%**

| Difficulty | Precision | Recall | F1-Score | Support |
|----------|----------|--------|---------|---------|
| Easy     | 0.7856   | 0.8201 | 0.8025  | 478     |
| Medium   | 0.6075   | 0.5629 | 0.5843  | 517     |
| Hard     | 0.8480   | 0.8639 | 0.8559  | 904     |


### 🔹 Regression Results

| Metric | CatBoost (Best) | LightGBM |
|------|-----------------|----------|
| **MAE** | **1.1086** | 1.1124 |
| **RMSE** | **1.4523** | 1.4615 |
| **R² Score** | **0.7092** | 0.7055 |

---

## ▶️ Run Locally

1. **Clone the Repository**
   ```bash
   git clone https://huggingface.co/spaces/1404Samyak/AutoJudge

2. **Navigate to the Project Directory**
    ```bash
    cd AutoJudge

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

4. **Run the Application**
   ```bash
   python streamlit_app.py
    ```
---
### Web Interface
Click on https://huggingface.co/spaces/1404Samyak/AutoJudge to view the live app
- Pick any problem from codeforces and paste its link shown at address bar and paste it in the link .The code will extract the problem id to get information
- Then paste the constraints (time and memory limit per test) displayed for each problem in the constraints section
- Then paste the problem description ,input description and output scription as it is the word2vec model will take care of it
- Then provide the problem tags given for all problems and finally give some sample input /output examples which are available as test cases for each problem in codeforces.You can give 2-3 or all of test cases  just for the model to get rough idea of input and output structure.

---
### 🧠 Technologies Used

- **Programming Language**: Python
- **Web Framework**: Streamlit 
- **Machine Learning Models**: LightGBM, CatBoost, Logistic/Linear Regression, SVM, RandomForest, XGBoost, GradientBoost, AdaBoost
- **NLP & Embeddings**: Word2Vec (Gensim), NLTK
- **Knowledge Graphs**: PyKEEN (TransE)
- **Data Processing**: Pandas, NumPy
- **Deep Learning Backend**: PyTorch
- **Deployment Platform**: Hugging Face Spaces

















