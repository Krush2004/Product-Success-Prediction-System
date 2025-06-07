import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Load the historical product 
file_path = "C:/Users/krush/OneDrive/Desktop/SK International/Main Project/historic.csv"
historic = pd.read_csv(file_path)
print("First 5 rows of data:")
print(historic.head())

# Value counts of target variable
sns.countplot(data=historic, x="success_indicator")
plt.title("Success Distribution (Top vs Flop)")
plt.xlabel("Success Indicator")
plt.ylabel("Count")
plt.show()

# Success rate by category
historic['is_top'] = (historic.success_indicator == 'top').astype(int)

# Barplot of success rate 
sns.barplot(data=historic, x="category", y="is_top")
plt.title("Success Rate by Category")
plt.xlabel("Product Category")
plt.ylabel("Top Success Rate")
plt.xticks(rotation=45)
plt.show()

# Missing values
print("\nMissing values per column:")
missing_cols = historic.isnull().sum()
print(missing_cols[missing_cols > 0])

# Heatmap to visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(historic.isnull(), cbar=False, cmap='viridis')
plt.title("Heatmap of Missing Values")
plt.xlabel("Columns")
plt.ylabel("Records")
plt.show()
