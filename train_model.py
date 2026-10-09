"""
Train the Heart Disease SVC model and persist artifacts:
  - model.pkl     : trained SVC
  - scaler.pkl    : fitted StandardScaler
  - columns.pkl   : ordered list of encoded feature column names
"""
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("heart.csv")

# ── Encode categoricals (same as notebook) ────────────────────────────────────
cat_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
X_encoded = pd.get_dummies(df, columns=cat_cols)
X_encoded = X_encoded.drop("target", axis=1)
y = df["target"]

feature_columns = list(X_encoded.columns)

# ── Split ──────────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=1
)

# ── Scale ──────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ── Train SVC (best accuracy in notebook: ~80.3 %) ────────────────────────────
model = SVC(kernel="linear", random_state=0, probability=True)
model.fit(X_train_sc, y_train)

acc = model.score(X_test_sc, y_test)
print(f"Test accuracy: {acc:.3f}")

# ── Persist ───────────────────────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

print("Saved: model.pkl, scaler.pkl, columns.pkl")
