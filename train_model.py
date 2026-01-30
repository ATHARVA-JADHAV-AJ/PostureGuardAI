import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load Data
print("Loading dataset...")
try:
    df = pd.read_csv('posture_dataset.csv')
except FileNotFoundError:
    print("ERROR: 'posture_dataset.csv' not found. Run data_collector.py first.")
    exit()

# --- FIX: REMOVE BROKEN DATA ---
# This line deletes any row that has missing numbers (NaN)
original_count = len(df)
df = df.dropna()
new_count = len(df)

if original_count != new_count:
    print(f"⚠️ Cleaned dataset: Removed {original_count - new_count} empty/corrupt rows.")

if len(df) < 10:
    print("ERROR: Not enough data left after cleaning. Please record more data!")
    exit()

# 2. Prepare Features (X) and Labels (y)
X = df.drop('label', axis=1)
y = df['label']

# 3. Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Model
print("Training the Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 6. Save the Brain
joblib.dump(model, 'posture_model.pkl')
print("SUCCESS: Model saved as 'posture_model.pkl' 🧠")