# 🚀 Comprehensive Multi-Label Toxic Comment Classification Using Advanced NLP Techniques

## 📌 Project Overview
This project focuses on detecting **toxic comments** using **advanced NLP techniques** and **machine learning models**. By leveraging **deep learning models, transformers (DistilBERT), and traditional classifiers (Logistic Regression, XGBoost, LightGBM)**, we aim to **classify comments into six toxicity categories**:
- **Toxic**
- **Severe Toxic**
- **Obscene**
- **Threat**
- **Insult**
- **Identity Hate**

The dataset used in this project comes from the **Toxic Comment Classification Challenge** on Kaggle, which contains Wikipedia talk page comments labeled for multiple types of toxicity.

---

## 📂 Dataset
- **Source:** [Kaggle - Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data)
- **Size:** Over **1.5 million comments**
- **Challenge:** Extreme **class imbalance** (most comments are non-toxic, while severe toxic and identity hate are rare)

### 🛠️ **Preprocessing Steps**
- **Text Cleaning:** Removal of special characters, emojis, and stopwords.
- **Tokenization & Vectorization:** Using **TF-IDF** for traditional models and **Word Embeddings** for deep learning.
- **Handling Class Imbalance:** **Oversampling**, **undersampling**, and **class weighting** to improve minority class detection.

---

## 🔍 Exploratory Data Analysis (EDA)
Key insights from **EDA**:
1. **Toxicity by Race/Ethnicity** – Certain identity-based comments have a **higher toxicity percentage**.
2. **Temporal Trends** – Toxicity increased from **2015 to 2017**.
3. **Gender-Based Toxicity** – Higher toxicity is observed in comments mentioning **female identities**.
4. **Word Clouds** – Common **insulting words** appear frequently in **toxic comments**.

📊 **Visualization Samples**:
- **Toxicity Trends Over Time**  
  ![Toxicity Trends](images/toxicity_trends.png)

- **Word Cloud for Toxic Comments**  
  ![Word Cloud](images/toxic_wordcloud.png)

---

## ⚙️ Machine Learning Models Used
This project implements multiple models, evaluating their effectiveness in **toxic comment classification**.

| Model | ROC-AUC Score | Strengths | Weaknesses |
|--------|------------|-----------|------------|
| **Logistic Regression (TF-IDF)** | 0.94 | Fast, interpretable | Fails on rare labels |
| **XGBoost** | 0.92 | Handles non-linearity well | Needs feature engineering |
| **LSTM & GRU** | 0.93 | Captures sequence information | High training cost |
| **DistilBERT (Fine-tuned)** | 0.95 | Best performance, understands context | Requires more data |
| **LightGBM** | 0.87 | Efficient on large data | Not as strong in capturing text nuances |
| **CatBoost** | 0.87 | Good balance between precision & recall | Computationally intensive |

🔍 **Best Performing Model**: **DistilBERT** achieves the highest **ROC-AUC (0.95)** with **balanced performance** across all toxicity labels.

---

## 📊 Model Performance
### **1️⃣ Logistic Regression with TF-IDF**
- **Strengths**: Fast, easy to interpret.
- **Weaknesses**: Struggles with **rare labels** like severe toxicity.
- **Validation Performance:**

Precision: 0.95 Recall: 0.94 F1-Score: 0.93

### **2️⃣ XGBoost Classifier**
- **Improves F1-score** on **minority classes** but lacks deep contextual understanding.
- **Macro F1-Score:** 0.68

### **3️⃣ LSTM & GRU Models**
- **Captures long-range dependencies** but **expensive computationally**.
- **Validation Accuracy:** 0.93

### **4️⃣ DistilBERT Fine-Tuning (Best Model)**
- **Understands context deeply using Transformers**
- **ROC-AUC:** 0.95
- **Validation F1-Score:** 0.96
- **Confusion Matrix:**
![Confusion Matrix](images/confusion_matrix.png)

### **5️⃣ LightGBM & CatBoost Models**
- **LightGBM Accuracy:** 76.72%
- **CatBoost Accuracy:** 80.65%
- **Tradeoff:** **Faster inference but slightly less effective for complex toxicity cases**.

---

📌 Future Enhancements
Fine-tuning larger Transformer models (RoBERTa, GPT-4)
Deploying the model as a REST API for real-time moderation
Developing a browser extension to filter toxic comments in real-time
