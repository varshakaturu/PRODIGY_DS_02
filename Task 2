import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic training dataset
data = pd.read_csv('/task 2 dataset.csv')

# Check for missing values
missing_values = data.isnull().sum()
print(missing_values)

# Fill missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Convert categorical variables using one-hot encoding
data = pd.get_dummies(data, columns=['Sex', 'Embarked'])

# Drop columns that won't be useful for analysis, but only if they exist
for col in ['Name', 'Ticket', 'Cabin']:
    if col in data.columns:
        data.drop(col, axis=1, inplace=True)

#Exploratory Data Analysis (EDA)
# Get summary statistics
print(data.describe())

# Histogram of 'Age'
plt.hist(data['Age'], bins=20, edgecolor='k')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age')
plt.show()

# Correlation matrix
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# Survival rate by gender
survival_rate_by_gender = data.groupby('Sex_female')['Survived'].mean()
print(survival_rate_by_gender)

# Survival rate by class
survival_rate_by_class = data.groupby('Pclass')['Survived'].mean()
print(survival_rate_by_class)

# Pair plot to visualize relationships
sns.pairplot(data[['Age', 'Fare', 'Survived']], hue='Survived')
plt.show()


# Scatter plot for 'Age' vs 'Fare'
plt.scatter(data['Age'], data['Fare'], c=data['Survived'], cmap='coolwarm', alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Age vs Fare colored by Survival')
plt.colorbar(label='Survived')
plt.show()
