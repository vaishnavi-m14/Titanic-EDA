# Titanic EDA Code for Beginners

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: show plots inline if in notebook (ignored in script)
# %matplotlib inline

# Load dataset
df = pd.read_csv("train.csv")

# Basic Information
print("Shape of dataset:", df.shape)
print("\nData types and non-null counts:")
print(df.info())
print("\nSummary statistics:")
print(df.describe(include='all'))

# Missing values
print("\nMissing values:\n", df.isnull().sum())

# Value counts for categorical columns
print("\nValue Counts:")
print("Sex:\n", df['Sex'].value_counts())
print("Pclass:\n", df['Pclass'].value_counts())
print("Embarked:\n", df['Embarked'].value_counts())

# Plot settings
sns.set(style='whitegrid')
plt.figure(figsize=(10, 6))

# Bar plots
sns.countplot(x='Sex', data=df)
plt.title("Gender Distribution")
plt.show()

sns.countplot(x='Pclass', data=df)
plt.title("Passenger Class Distribution")
plt.show()

sns.countplot(x='Embarked', data=df)
plt.title("Port of Embarkation")
plt.show()

# Histograms
df['Age'].hist(bins=30, edgecolor='black')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

df['Fare'].hist(bins=40, edgecolor='black')
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.show()

# Boxplots
sns.boxplot(y='Age', data=df)
plt.title("Boxplot of Age")
plt.show()

sns.boxplot(y='Fare', data=df)
plt.title("Boxplot of Fare")
plt.show()

# Survival vs Category
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title("Survival by Gender")
plt.show()

sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title("Survival by Class")
plt.show()

sns.countplot(x='Survived', hue='Embarked', data=df)
plt.title("Survival by Embarked Port")
plt.show()

# Convert 'Sex' to numeric for correlation
df['Sex_num'] = df['Sex'].map({'male': 0, 'female': 1})

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
