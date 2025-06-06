import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load and clean data
column_names = [f'feature_{i}' for i in range(1558)] + ['label']
df = pd.read_csv('ad.data', header=None, names=column_names, na_values=['?', '   ?'])
df.dropna(inplace=True)
df['label'] = df['label'].apply(lambda x: 1 if x == 'ad.' else 0)
df[df.columns[:-1]] = df[df.columns[:-1]].astype(float)

# Prepare data
X = df.drop('label', axis=1)
y = df['label']
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'ad_classifier_model.pkl')
print("Model saved as ad_classifier_model.pkl")
