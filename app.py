import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
@st.cache_data
def load_data():
    column_names = [f'feature_{i}' for i in range(1558)] + ['label']
    df = pd.read_csv('ad.data', header=None, names=column_names, na_values=['?', '   ?'])
    df.dropna(inplace=True)
    df['label'] = df['label'].apply(lambda x: 1 if x == 'ad.' else 0)
    feature_cols = df.columns[:-1]
    df[feature_cols] = df[feature_cols].astype(float)
    return df

# Load and prepare data
df = load_data()
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Home", "Dataset", "Summary", "Graphs", "Predict"])

# --- Navigation Logic ---

# Home Page
if option == "Home":
    st.title("📊 Internet Advertisement Classifier")
    st.markdown("""
        This app uses a machine learning model to classify whether an input corresponds to an **Advertisement** or **Non-Advertisement**.
        \nExplore the options in the sidebar to learn more or try predictions.
    """)

# Dataset Page
elif option == "Dataset":
    st.title("📁 Dataset View")
    st.write("Here's a sample of the dataset after cleaning:")
    st.dataframe(df.head(20))

# Summary Page
elif option == "Summary":
    st.title("📈 Dataset Summary")
    st.write("### Data Description")
    st.write(df.describe())

    st.write("### Class Distribution")
    class_counts = df['label'].value_counts().rename({0: 'Not Ad', 1: 'Ad'})
    st.bar_chart(class_counts)

# Graphs Page
elif option == "Graphs":
    st.title("📊 Feature Graphs")

    st.write("Select a feature to visualize:")
    feature_index = st.slider("Feature Index", 0, 20, 0)
    feature_name = f"feature_{feature_index}"

    fig, ax = plt.subplots()
    ax.hist(df[feature_name], bins=30, color='skyblue')
    ax.set_title(f"Distribution of {feature_name}")
    st.pyplot(fig)

# Predict Page
elif option == "Predict":
    st.title("🤖 Make a Prediction")
    st.write("Input values for the first few features to predict whether it's an Advertisement:")

    user_input = []
    for i in range(5):  # Only first 5 features for input
        user_input.append(st.number_input(f"Feature {i}", value=0.0))

    if st.button("Predict"):
        input_data = user_input + [0.0]*(X.shape[1] - len(user_input))  # Padding
        prediction = model.predict([input_data])
        st.success("Prediction: **Advertisement**" if prediction[0] == 1 else "Not an Advertisement")

    st.subheader("📋 Model Performance")
    y_pred = model.predict(X_test)
    st.text(classification_report(y_test, y_pred))
