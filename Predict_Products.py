import pandas as pd
import os
from Model_RandomForest import RandomForestModel

#  File paths 
BASE_DIR = "C:/Users/krush/OneDrive/Desktop/SK International/Main Project"
historic_data_path = os.path.join(BASE_DIR, "historic.csv")
prediction_input_path = os.path.join(BASE_DIR, "prediction_input.csv")
output_dir = os.path.join(BASE_DIR, "output")
output_file_path = os.path.join(output_dir, "product_predictions.csv")

# Output directory exists 
os.makedirs(output_dir, exist_ok=True)

# Load and Train the Model 
model = RandomForestModel()
model.load(historic_data_path)
model.preprocess()
model.train()

# Predict using the file path (model handles preprocessing) 
results = model.predict(new_data_path=prediction_input_path)

# Save to CSV 
results.to_csv(output_file_path, index=False)
print("\n Prediction results saved to:", output_file_path)

#  Preview results
print(results.head())
