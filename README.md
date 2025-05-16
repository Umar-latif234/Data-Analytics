## **📌 Task 1: Employee Attrition Prediction**

### **Objective:**

Predict whether an employee will leave the company (attrition) based on HR dataset features.

### **Dataset:**

IBM HR Analytics Employee Attrition dataset.

### **Key Models & Tools:**

* **Random Forest Classifier**
* **Preprocessing:** OneHotEncoder, StandardScaler
* **Pipeline** for data transformation and modeling

### **Evaluation Metrics:**

* Confusion Matrix
* Precision, Recall, F1-Score
* Overall Accuracy: **79%**
* Observations: Better performance on class 0 (non-attrition) compared to class 1 (attrition), indicating class imbalance.

---

## **📌 Task 2: Text Summarization System**

### **Objective:**

Summarize news articles using extractive and abstractive summarization.

### **Dataset:**

CNN/Daily Mail Dataset

### **Key Models & Tools:**

* **Extractive:** spaCy + TF-IDF-based sentence scoring
* **Abstractive:** BART model (from HuggingFace Transformers)
* **Evaluation:** ROUGE metrics

### **Pipeline:**

* Load and preprocess news articles
* Generate summaries using two different methods
* Compare summaries with reference highlights using ROUGE

---

## **📌 Task 3: SVM Classification on Tabular Data**

### **Objective:**

Build an SVM classifier to predict classes based on numeric and categorical features.

### **Key Models & Tools:**

* **Support Vector Machine (SVC)**
* **Preprocessing:** StandardScaler
* **Metrics:** F1-Score, ROC AUC, Confusion Matrix, Classification Report

### **Observations:**

* Good use of visualization (`matplotlib`, `seaborn`)
* Comprehensive model evaluation setup with emphasis on performance metrics

---

 Tools & Libraries Used
Python, Jupyter Notebook
Libraries:** scikit-learn, pandas, numpy, matplotlib, seaborn, spaCy, HuggingFace Transformers, datasets
Model Evaluation:** Confusion Matrix, Classification Report, ROUGE

