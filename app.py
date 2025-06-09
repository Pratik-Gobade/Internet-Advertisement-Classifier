import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np

# --- Load dataset ---
@st.cache_data
def load_data():
    column_names = [f'feature_{i}' for i in range(1558)] + ['label']
    df = pd.read_csv('ad.data', header=None, names=column_names, na_values=['?', '   ?'])
    df.dropna(inplace=True)
    df['label'] = df['label'].apply(lambda x: 1 if x == 'ad.' else 0)
    df[df.columns[:-1]] = df[df.columns[:-1]].astype(float)
    return df

# --- Load model ---
@st.cache_resource
def load_model():
    return joblib.load('ad_classifier_model.pkl')

# Load Data and Model
df = load_data()
model = load_model()

# Prepare Data
X = df.drop('label', axis=1)
y = df['label']

# Sidebar Navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Home", "Dataset", "Summary", "Graphs", "Predict"])

# Home Page
if option == "Home":
    st.title("📊 Internet Advertisement Classifier")
    st.markdown("""
        This app uses a **Random Forest Classifier** model to detect if the data input is related to an **Advertisement** or **Non-Advertisement**.
        
        Navigate using the sidebar to explore dataset, summary statistics, visualizations, and try your own predictions!
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
    
    feature_index = st.slider("Feature Index", 0, 1557, 0)
    feature_name = f"feature_{feature_index}"
    
    fig, ax = plt.subplots()
    ax.hist(df[feature_name], bins=30, color='skyblue', edgecolor='black')
    ax.set_title(f"Distribution of {feature_name}")
    st.pyplot(fig)

# Predict Page
elif option == "Predict":
    st.title("🤖 Make a Prediction")
    st.write("Enter values for the first few features:")

    user_input = []
    num_features = 10  # Let user input for 10 features
    for i in range(num_features):
        val = st.number_input(f"Feature {i}", value=0.0)
        user_input.append(val)

    if st.button("Predict"):
        input_data = user_input + [0.0] * (X.shape[1] - len(user_input))  # Pad remaining features with 0
        prediction = model.predict([input_data])[0]
        result = "Advertisement" if prediction == 1 else "Not an Advertisement"
        st.success(f"Prediction: **{result}**")

    st.subheader("📋 Model Performance")
    # Evaluate only when needed to save time
    if st.checkbox("Show classification report"):
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_pred = model.predict(X_test)
        st.text(classification_report(y_test, y_pred))
