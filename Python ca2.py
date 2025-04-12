import pandas as pd
import numpy as np
import dtale

import seaborn as sns
import matplotlib.pyplot as plt

de=pd.read_csv("C:/Users/hp/OneDrive/Desktop/pythonca/pythonProject.csv")
de.head()
de.info()
na = de.isna().sum()
print(na)
de=de.fillna({'Value': de["Value"].mean(),'LowCI':de['LowCI'].mean(), 'HighCI': de['HighCI'].mean()})

de['Confidence Interval'] = de['Confidence Interval'].astype(str)

# Function to fill missing confidence interval values
def fill_confidence_interval(row):
    if pd.isna(row['Confidence Interval']) or row['Confidence Interval'].lower() == 'nan':
        return f"{row['LowCI']} - {row['HighCI']}"
    return row['Confidence Interval']

de['Confidence Interval'] = de.apply(fill_confidence_interval, axis=1)

# Calculate quartiles
Q1 = de["Value"].quantile(0.25)
Q2 = de["Value"].median()
Q3 = de["Value"].quantile(0.75)

# Assign quartile range
de["Quartile Range"] = de["Value"].apply(lambda x: "Q1" if x <= Q1 else "Q2" if x <= Q2 else "Q3" if x <= Q3 else "Q4")

# Delete the "Suppression Flag" column
de.drop(columns=["Suppression Flag"], inplace=True)

# Convert both date columns to a consistent format (YYYY-MM-DD)
de["Time Period Start Date"] = pd.to_datetime(de["Time Period Start Date"]).dt.strftime("%Y-%m-%d")
de["Time Period End Date"] = pd.to_datetime(de["Time Period End Date"]).dt.strftime("%Y-%m-%d")


pd.set_option('display.max_rows', 10500)
pd.set_option('display.max_columns', 50)
print(de)

'''
dfd=pd.read_csv("C:/Users/hp/OneDrive/Desktop/pythonProject.csv")
dt=dtale.show(dfd)
dt.open_browser()
'''


#visualization 
#Bar Plot – Average Value by State
plt.figure(figsize=(14, 6))
sns.barplot(data=de, x='State', y='Value', estimator='mean', ci=None)
plt.xticks(rotation=90)
plt.title("Average Value by State")
plt.tight_layout()
plt.show()


# Box Plot – Value Distribution by Quartile
plt.figure(figsize=(8, 6))
sns.boxplot(data=de, x='Quartile Range', y='Value')
plt.title("Value Distribution Across Quartiles")
plt.show()


#Line Plot – Value Over Time
de['Time Period Start Date'] = pd.to_datetime(de['Time Period Start Date'])

plt.figure(figsize=(12, 6))
sns.lineplot(data=de.sort_values("Time Period Start Date"), x='Time Period Start Date', y='Value')
plt.title("Value Over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#Heatmap – Correlation Between Numeric Columns
plt.figure(figsize=(8, 6))
sns.heatmap(de.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Numeric Columns")
plt.show()


#Count Plot – Number of Records by Quartile Range
plt.figure(figsize=(6, 4))
sns.countplot(data=de, x='Quartile Range', order=['Q1', 'Q2', 'Q3', 'Q4'],palette='viridis')
plt.title("Record Count by Quartile")
plt.show()


# Bar Plot – Average Value by Indicator
plt.figure(figsize=(14, 6))
sns.barplot(data=de, x='Indicator', y='Value', estimator='count', ci=None,palette='tab20b')
plt.xticks(rotation=90)
plt.title("Average Value by Indicator and Sex")
plt.tight_layout()
plt.show()

#Scatter Plot of Value vs LowCI 
plt.figure(figsize=(8, 6))
sns.scatterplot(data=de, x='LowCI', y='Value', hue='Quartile Range', palette='viridis')
plt.title("Scatter Plot of Value vs LowCI")
plt.xlabel("Low Confidence Interval")
plt.ylabel("Value")
plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = de.copy()

le = LabelEncoder()
df['State'] = le.fit_transform(df['State'])
df['Indicator'] = le.fit_transform(df['Indicator'])

df['Time Period Start Date'] = pd.to_datetime(df['Time Period Start Date'])
df['Date'] = df['Time Period Start Date'].map(pd.Timestamp.toordinal)

# Define features and target
X = df[['State', 'Indicator', 'Date']]
y = df['Value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions and metrics
pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

print(" Mean Squared Error:", round(mse, 2))
print(" R^2 Score:", round(r2, 2))

# Visual comparison
plt.scatter(y_test, pred, alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.grid(True)
plt.tight_layout()
