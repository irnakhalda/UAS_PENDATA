import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Load dataset
df = pd.read_csv("Thyroid_Diff.csv")

# 2. Pisahkan fitur dan target
X = df.drop(columns=["Recurred"])
y = df["Recurred"]

# 3. Encode kolom kategorikal
label_encoders = {}
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

# 4. Encode target (Recurred)
le_target = LabelEncoder()
y = le_target.fit_transform(y)
label_encoders["Recurred"] = le_target

# 5. Split data training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Latih model Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 7. Simpan model dan label encoder
with open("model_rf.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

# 8. Cetak akurasi model pada data test
accuracy = model.score(X_test, y_test)
print(f"Akurasi Model: {accuracy * 100:.2f}%")
print("Model dan encoder berhasil disimpan.")
