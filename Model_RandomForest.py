import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split # To split the train and test outcomes
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


# Funcation Load() to load the training data
class RandomForestModel:
    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load(self, file_path="C:/Users/krush/OneDrive/Desktop/SK International/Main Project/historic.csv"):
        """Load historical product dataset."""
        self.data = pd.read_csv(file_path)

# Function for train the model
    def preprocess(self):
        """Preprocess data: handle missing values, encode categories."""
        df = self.data.copy()
        df.dropna(inplace=True) # Drop missing values

        # Encode categorical features
        label_encoders = {}
        for col in ['category', 'main_promotion', 'color']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        df['success_indicator'] = df['success_indicator'].map({'top': 1, 'flop': 0}) # Map 'top' and 'flop'
        self.X = df.drop(columns=['item_no', 'success_indicator'])
        self.y = df['success_indicator']

# Funcation Load() to load the training data
    def train(self):
        """Train the Random Forest classifier."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)

# Testing the New data for prediction
    def test(self):
        """Evaluate model on test set."""
        y_pred = self.model.predict(self.X_test)
        print("Random Forest Accuracy:", accuracy_score(self.y_test, y_pred))
        print("Classification Report:\n", classification_report(self.y_test, y_pred))
        return self.y_test, y_pred

# Calling the Run parameters to show path
    def predict(self, new_data_path="C:/Users/krush/OneDrive/Desktop/SK International/Main Project/prediction_input.csv"):
        """Predict on new product data."""
        pred_df = pd.read_csv(new_data_path)

        pred_df.dropna(inplace=True) # Drop missing values

        # Encode same categorical
        for col in ['category', 'main_promotion', 'color']:
            pred_df[col] = LabelEncoder().fit_transform(pred_df[col])

        X_pred = pred_df.drop(columns=['item_no'])

        predictions = self.model.predict(X_pred) # Predict probabilities and labels
        pred_df['prediction'] = predictions
        pred_df['prediction_label'] = pred_df['prediction'].map({1: 'top', 0: 'flop'})

        print("\nSample Predictions:")
        print(pred_df[['item_no', 'prediction_label']].head())

        return pred_df[['item_no', 'prediction_label']]


# Run the script for testing
if __name__ == "__main__":
    model = RandomForestModel()
    model.load()
    model.preprocess()
    model.train()
    model.test()
    model.predict()
