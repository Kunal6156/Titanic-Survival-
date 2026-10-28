"""
REAL DATA PROJECT: Kaggle Titanic

What to do
  1. Put titanic.csv (the Kaggle train file) in the same folder as this script.
  2. Run:  python eda.py titanic.csv --target Survived   and read the output.
  3. Implement run_project(df) below. If you already passed survival_project.py,
     copy your solution in - the data is loaded and cleaned up for you so the
     same code works. Then do requirement 10, which is new.
  4. Run:  python titanic_real.py     -> should print PASSED
  5. Copy the numbers into github_readme_template.md and push to GitHub.

Already done for you in the loader (see __main__):
  - column names lowercased, 'passengerid' renamed to 'passenger_id'
  - 'ticket' dropped

Implement run_project(df) so it returns:
    {
      "cv_scores": {"logreg": float, "rf": float},
      "best_model": "logreg" or "rf",
      "test_accuracy": float,
      "test_f1": float,
    }

Requirements
1. Remove duplicate rows.
2. Feature engineering:
   a. title      : text between ", " and "." in 'name'.
                   Keep Mr, Mrs, Miss, Master; replace all others with "Other".
   b. family_size: sibsp + parch + 1
   c. has_cabin  : 1 if 'cabin' is not missing, else 0
3. Drop 'passenger_id', 'name' and 'cabin'.
4. Target is 'survived'. Split: test_size=0.2, random_state=42, stratify=y.
5. Preprocessing (fit on train only): numeric -> median impute + scale;
   categorical -> most frequent impute + one-hot (handle_unknown="ignore").
6. Two pipelines using that preprocessing:
      logreg: LogisticRegression(max_iter=1000)
      rf    : RandomForestClassifier(n_estimators=200, random_state=42)
7. Score each with cross_val_score (cv=5, scoring="accuracy") on the TRAIN split.
8. best_model = the one with the higher mean CV score.
   Fit it on the train split; report accuracy and F1 on the test split.
9. Print a small comparison table (model, CV accuracy).
10. NEW (printed, not checked): print the 5 most important features of the
    fitted best model.
      - rf     : model.named_steps["clf"].feature_importances_
      - logreg : abs(model.named_steps["clf"].coef_[0])
      - names  : model.named_steps["prep"].get_feature_names_out()
    (name your pipeline steps "prep" and "clf" so these lines work)

Pass condition: test_accuracy >= ACC_MIN, best_model is the higher CV score.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


ACC_MIN = 0.75


# ---------------- YOUR CODE BELOW ----------------
def run_project(df: pd.DataFrame) -> dict:
    # TODO: implement the steps above
    df=df.drop_duplicates()
    x=df.drop(columns=['passenger_id','name','sibsp','parch','fare','cabin','embarked','survived'])
    x['sex']=x['sex'].map({'male':1,'female':0})
    y=df['survived']
    x_col=Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())])
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=42,stratify=y)
    logreg = Pipeline([('prep', x_col), ('clf', LogisticRegression(max_iter=1000))])
    rf = Pipeline([('prep', x_col), ('clf', RandomForestClassifier(n_estimators=200, random_state=42))])
    cv_logreg = cross_val_score(logreg, x_train, y_train, cv=5, scoring='accuracy').mean()
    cv_rf = cross_val_score(rf, x_train, y_train, cv=5, scoring='accuracy').mean()
    cv_scores = {"logreg": float(cv_logreg), "rf": float(cv_rf)}
    best_name = max(cv_scores, key=cv_scores.get)
    best_model = logreg if best_name == "logreg" else rf

    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)
    test_acc = float(accuracy_score(y_test, y_pred))
    test_f1 = float(f1_score(y_test, y_pred))

    print("\nModel   CV accuracy")
    for name, score in cv_scores.items():
        print(f"{name:7s} {score:.3f}")

    clf = best_model.named_steps["clf"]
    values = clf.feature_importances_ if best_name == "rf" else np.abs(clf.coef_[0])
    print("\nTop features:")
    print(pd.Series(values, index=x_train.columns).sort_values(ascending=False).head(5))

    return {
        "cv_scores": cv_scores,
        "best_model": best_name,
        "test_accuracy": test_acc,
        "test_f1": test_f1,
    }


# ---------------- LOADER + CHECKER (don't edit) ----------------
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "titanic.csv"
    data = pd.read_csv(path)
    data.columns = data.columns.str.lower()
    data = data.rename(columns={"passengerid": "passenger_id"})
    data = data.drop(columns=["ticket"], errors="ignore")
    print("Raw shape:", data.shape, "| columns:", list(data.columns))

    res = run_project(data.copy())
    print(res)
    assert set(res) == {"cv_scores", "best_model", "test_accuracy", "test_f1"}, "Wrong keys"
    assert set(res["cv_scores"]) == {"logreg", "rf"}, "cv_scores needs logreg and rf"
    assert all(isinstance(v, float) for v in res["cv_scores"].values()), "CV scores must be floats"
    assert isinstance(res["test_accuracy"], float) and isinstance(res["test_f1"], float), "Use floats"
    assert res["best_model"] == max(res["cv_scores"], key=res["cv_scores"].get), \
        "best_model must be the higher CV score"
    assert res["test_accuracy"] >= ACC_MIN, "Test accuracy too low"
    print("PASSED")
