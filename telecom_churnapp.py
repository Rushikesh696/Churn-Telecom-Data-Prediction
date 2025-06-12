import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

st.set_page_config(layout="wide")
st.title("📊Telecom Churn Analysis")
st.image("telecom_churn_image.jpg", width=500)

# load the data
@st.cache_data
def load_data():
    churn_data = pd.read_csv("churn_dataset.csv")
    # Clean TotalCharges column
    churn_data = churn_data[churn_data["TotalCharges"] != " "]  # Remove rows with blank TotalCharges
    churn_data["TotalCharges"] = churn_data["TotalCharges"].astype(float)
    return churn_data

churn_dataset = load_data()
st.write("## Displaying first five rows", churn_dataset.head())

# show basic information about the dataset
if st.checkbox("show data summary"):
    st.write(churn_dataset.describe())
    st.write(churn_dataset.info())

# show the distribution of the target variable
st.write("# Univariate Analysis")
st.write("### Target Variable Distribution")

fig1, ax1 = plt.subplots(figsize=(8,3))
sns.countplot(data=churn_dataset, x='Churn',ax=ax1)
ax1.set_title("Churn Distribution")
ax1.set_xlabel("Churn")
ax1.set_ylabel("Count")
st.pyplot(fig1)



# Add Insights
st.markdown("### 🔍 Insights")
churn_count = churn_dataset['Churn'].value_counts()
churn_percent = (churn_count / churn_count.sum()) * 100

st.write(f"- **Non-Churn Customers**: {churn_count[0]} ({churn_percent[0]:.2f}%)")
st.write(f"- **Churn Customers**: {churn_count[1]} ({churn_percent[1]:.2f}%)")

# Optional insight statement
if churn_percent[1] < 40:
    st.markdown("- The dataset shows that **most customers did not churn**, indicating class imbalance.")
else:
    st.markdown("- The churn rate is relatively high, which could impact business revenue.")




# show the distribution of numerical features
st.write("### Numerical Features Distribution")
def plot_numerical_distribution(data, feature):
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.histplot(data[feature], bins=30, kde=True, ax=ax)
    ax.set_title(f"Distribution of {feature}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Count")
    st.pyplot(fig)

numerical_features = churn_dataset.select_dtypes(include=[np.number]).columns.tolist()
print(numerical_features)
selected_feature = st.selectbox("Select a numerical feature to plot", numerical_features)
if selected_feature:
    plot_numerical_distribution(churn_dataset, selected_feature)

# show the distribution of categorical features
st.write("### Categorical Features Distribution")
def plot_categorical_distribution(data, feature):
    fig, ax = plt.subplots(figsize=(8,3))
    sns.countplot(data=data, x=feature, ax=ax)
    ax.set_title(f'Distribution of {feature}')
    ax.set_xlabel(feature)
    ax.set_ylabel('Count')
    st.pyplot(fig)

categorical_features = churn_dataset.select_dtypes(include=[object]).columns.tolist()
if 'customerID' in categorical_features:
    categorical_features.remove('customerID')
print(categorical_features)
selected_categorical_feature = st.selectbox("Select a categorical feature to plot", categorical_features)
if selected_categorical_feature:
    plot_categorical_distribution(churn_dataset, selected_categorical_feature)


# features related to churn
st.write("### Influence of features on churn")
def plot_churn_rate(data, feature):
    fig, ax = plt.subplots(figsize=(8,3))
    sns.countplot(data=data, x=feature, hue='Churn', ax=ax)
    ax.set_title(f'Churn Rate by {feature}')
    ax.set_xlabel(feature)
    ax.set_ylabel('Count')
    ax.legend(title='Churn', loc='upper right')
    st.pyplot(fig)


if 'Churn' in categorical_features:
    categorical_features.remove('Churn')
selected_churn_feature = st.selectbox("Select a feature to analyze churn rate", categorical_features)
if selected_churn_feature:
    plot_churn_rate(churn_dataset, selected_churn_feature)






