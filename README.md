# 🛍️ Product Success Prediction System

This project aims to predict whether a retail product will be a **"top"** or a **"flop"** before it is launched, using machine learning models trained on historical data.

---

## 📌 Project Overview

Retailers often struggle to predict the success of new products. This system uses historical product data to forecast product performance based on attributes like:

- **Category**
- **Color**
- **Promotion Type**
- **Other features**

The model classifies products as either:
- ✅ `top`: Likely to succeed
- ❌ `flop`: Likely to fail

---

## 🧠 Models Used

### ✅ Random Forest Classifier
- Handles both numerical and categorical features well
- Provides feature importance for interpretability
- Robust against overfitting

### ✅ Artificial Neural Network (Optional)
- Captures complex non-linear relationships
- Useful for experimentation

---

## 📁 Project Structure
Main Project/
│
├── historic.csv # (Train_data)
├── prediction_input.csv # (Test_data)
│
├── EDA.py 
├── Model_RandomForest.py
├── ANN_Model.py # (PyTorch)
├── Model_Selection.py 
├── Predict_Products.py 
│
├── output/
│ └── product_predictions.csv # Final predictions
│
└── requirements.txt # Libraries to Install


---

## ✅ Features

- Preprocessing: Handles missing values, encodes categorical data, scales features.
- Model Training: Uses Random Forest Classifier.
- Prediction: Classifies new product entries as `top` or `flop`.
- Evaluation: Classification report and feature importance metrics.


---
## 🔍 Model Evaluation

Evaluate your model's performance using:

- **Classification Report:** Precision, recall, F1-score, and support for each class.
- **Accuracy Score:** Overall percentage of correct predictions.
- **Feature Importance:** (Random Forest) Understand which features influence predictions most.

---

## 📈 Future Improvements

- Integrate advanced models like **XGBoost** or **LightGBM** for enhanced accuracy.
- Apply **GridSearch** for hyperparameter optimization.
- Deploy as a web application using **Flask** or **Streamlit**.
- Save predictions and logs to a **database** for tracking.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/product-success-predictor.git
cd product-success-predictor

python -m venv env
env\Scripts\activate  # On Windows

pip install -r requirements.txt

----- **🚀 How to Run** ------

Run Prediction Script:
---------------------------
python Predict_Products.py
----------------------------

Output:
------------------------------------------------------
Predictions will be printed in terminal and saved to:
------------------------------------------------------


---

## 👨‍💻 Author

**Krushna Mane**
----
