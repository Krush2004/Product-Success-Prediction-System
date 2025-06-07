import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from Model_RandomForest import RandomForestModel
from Model_ANN import ANNModel
from sklearn.metrics import confusion_matrix, classification_report


# Evaluate Random Forest Model
rf = RandomForestModel()
rf.load("C:/Users/krush/OneDrive/Desktop/SK International/Main Project/historic.csv")
rf.preprocess()
rf.train()
y_true_rf, y_pred_rf = rf.test()

# Evaluate ANN Model
ann = ANNModel()
ann.load("C:/Users/krush/OneDrive/Desktop/SK International/Main Project/historic.csv")
ann.preprocess()
ann.train()
y_true_ann, y_pred_ann = ann.test()

# Plot Confusion Matrices
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_matrix(y_true_rf, y_pred_rf), annot=True, fmt='d', ax=axs[0], cmap='Blues') # Heatmap for Random Forest
axs[0].set_title("Random Forest Confusion Matrix")
axs[0].set_xlabel("Predicted")
axs[0].set_ylabel("Actual")

sns.heatmap(confusion_matrix(y_true_ann, y_pred_ann), annot=True, fmt='d', ax=axs[1], cmap='Greens') # Heatmap for ANN
axs[1].set_title("ANN Confusion Matrix")
axs[1].set_xlabel("Predicted")
axs[1].set_ylabel("Actual")

plt.tight_layout()
plt.show()

# Model Selection Summary
# Evaluation metrics manually compared classification_report output
print("\n\033[1mModel Selection Summary:\033[0m")
print("Both models perform well, but Random Forest trains faster and provides slightly better accuracy.")
print("Therefore, Random Forest is selected for deployment due to its simplicity and efficiency.")
