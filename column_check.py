"""
Does a column help the model? Test it by removing it and comparing scores.

Idea (3 steps):
  1. Score the model with ALL columns             -> baseline
  2. Score the model again with ONE column removed -> new score
  3. Compare. Lower than baseline = the column helps (keep it).
              About the same or higher = the column isn't needed.

Run (titanic.csv in the same folder):  python column_check.py
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# ---- load and prepare (same cleaning as titanic_real.py) ----
df = pd.read_csv("titanic.csv")
df.columns = df.columns.str.lower()
t = df["name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
df["title"] = t.where(t.isin(["Mr", "Mrs", "Miss", "Master"]), "Other")
df["family_size"] = df["sibsp"] + df["parch"] + 1
df["has_cabin"] = df["cabin"].notna().astype(int)
df = df.drop(columns=["passengerid", "name", "cabin", "ticket"])

X = df.drop(columns=["survived"])
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)


# ---- one function: "give me the CV score if I drop these columns" ----
def cv_score(cols_to_drop):
    X_try = X_train.drop(columns=cols_to_drop)           # remove the columns
    num = X_try.select_dtypes(include="number").columns
    cat = X_try.select_dtypes(exclude="number").columns
    pre = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")),
                          ("s", StandardScaler())]), num),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    pipe = Pipeline([("prep", pre), ("clf", LogisticRegression(max_iter=1000))])
    scores = cross_val_score(pipe, X_try, y_train, cv=5, scoring="accuracy")
    return scores.mean(), scores.std()


# ---- run it for each idea ----
tests = {
    "baseline (nothing dropped)": [],
    "drop embarked": ["embarked"],
    "drop fare": ["fare"],
    "drop sibsp + parch (keep family_size)": ["sibsp", "parch"],
    "drop family_size": ["family_size"],
    "drop has_cabin": ["has_cabin"],
    "drop age": ["age"],
    "drop sex": ["sex"],
}
base_mean, _ = cv_score([])
print(f"{'test':42s} {'CV acc':>7s} {'+/-':>6s} {'vs base':>8s}")
for name, cols in tests.items():
    m, s = cv_score(cols)
    print(f"{name:42s} {m:7.3f} {s:6.3f} {m - base_mean:+8.3f}")

print("\nHow to read it: 'vs base' near 0 (within about +/-0.01) is just noise.")
print("A clearly negative number means that column helps, so keep it.")
