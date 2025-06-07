import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader

#  Neural network model
class ANN(nn.Module):
    def __init__(self, input_size):
        super(ANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# Wrapper for the ANN model
class ANNModel:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Funcation Load() to load the training data
    def load(self, file_path="C:/Users/krush/OneDrive/Desktop/SK International/Main Project/historic.csv"):
        """Load historical product dataset."""
        self.data = pd.read_csv(file_path)

# Function for train the model
    def preprocess(self):
        """Preprocess data: handle missing values, encode categories, scale features."""
        df = self.data.copy()
        df.dropna(inplace=True)

        for col in ['category', 'main_promotion', 'color']:
            df[col] = LabelEncoder().fit_transform(df[col])

        df['success_indicator'] = df['success_indicator'].map({'top': 1, 'flop': 0})
        X = df.drop(columns=['item_no', 'success_indicator'])
        y = df['success_indicator']

        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(X)
        self.y = y.values

# Funcation Load() to load the training data
    def train(self):
        """Train the PyTorch ANN model."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)

        self.X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        self.X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(self.device)
        self.y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32).to(self.device)

        train_dataset = TensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        input_size = self.X_train.shape[1]
        self.model = ANN(input_size).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train()
        for epoch in range(10):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

# Testing the New data for prediction
    def test(self):
        """Evaluate model performance on test data."""
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(self.X_test)
            y_pred_labels = (y_pred > 0.5).int().cpu().numpy()
            y_true = self.y_test.cpu().numpy()

        print("Classification Report for ANN Model (PyTorch):\n")
        print(classification_report(y_true, y_pred_labels))
        return y_true, y_pred_labels

# Calling the Run parameters to show path
    def predict(self, new_data_path="C:/Users/krush/OneDrive/Desktop/SK International/Main Project/prediction_input.csv"):
        """Predict on new product data."""
        pred_df = pd.read_csv(new_data_path)
        pred_df.dropna(inplace=True)

        for col in ['category', 'main_promotion', 'color']:
            pred_df[col] = LabelEncoder().fit_transform(pred_df[col])

        X_pred = pred_df.drop(columns=['item_no'])
        X_scaled = self.scaler.transform(X_pred)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_tensor)
            y_pred_labels = (y_pred > 0.5).int().cpu().numpy()

        pred_df['prediction'] = y_pred_labels
        pred_df['prediction_label'] = pred_df['prediction'].map({1: 'top', 0: 'flop'})

        print("\nSample ANN Predictions:")
        print(pred_df[['item_no', 'prediction_label']].head())

        return pred_df[['item_no', 'prediction_label']]


# Run script
if __name__ == "__main__":
    ann_model = ANNModel()
    ann_model.load()
    ann_model.preprocess()
    ann_model.train()
    ann_model.test()
    ann_model.predict()
