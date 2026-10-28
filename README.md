# Titanic Survival Prediction

A small, leakage-free scikit-learn workflow that predicts whether a passenger survived, comparing two models with cross-validation.

This is a learning project. It practises the full workflow: explore, clean, split, preprocess, compare models, evaluate once on held-out data.

## Data

- Source: Kaggle Titanic `train.csv` (not included in this repo; download it from Kaggle and save it as `titanic.csv`)
- 891 passengers, 12 columns; 342 survived and 549 did not
- Issues found with `eda.py`: 177 missing ages, 687 missing cabins (77%), 2 missing embarkation ports, and ID-like columns (`PassengerId`, `Name`, `Ticket`)

## Approach

1. **Explore** with `eda.py` (missing values, target balance, outliers, ID-like columns)
2. **Clean:** drop duplicates; drop `passenger_id`, `name`, `cabin`, `ticket`; also drop `sibsp`, `parch`, `fare` and `embarked` to keep the model simple
3. **Encode:** `sex` mapped to 0/1; remaining features are `pclass`, `sex`, `age`
4. **Split:** 80/20 train/test, `random_state=42`, stratified on `survived`
5. **Preprocess in a pipeline:** median imputation for `age`, then standard scaling
6. **Compare two models** with 5-fold cross-validation on the training set only: Logistic Regression and Random Forest (200 trees)
7. **Evaluate once** on the untouched test set using the model with the higher CV score

## Results

| Model | 5-fold CV accuracy |
|-------|--------------------|
| Logistic Regression | 0.788 |
| Random Forest | 0.806 |

Best model: Random Forest.
Test accuracy: **0.816** | Test F1: **0.744**

Feature importance (Random Forest): `age` 0.44, `sex` 0.40, `pclass` 0.17.

## Limitations and next steps

- Only three features are used. Next I would add engineered features (passenger title from the name, family size, a has-cabin flag) and measure their effect with cross-validation.
- No hyperparameter tuning yet. A `GridSearchCV` over tree depth and number of trees is the obvious next step.
- With a dataset this small, CV scores differ by about 0.01 between runs, so small differences between models should not be over-read.

## How to run

```
pip install numpy pandas scikit-learn
python eda.py titanic.csv --target Survived
python titanic_real.py
```

## Files

- `titanic_real.py`: the model pipeline and checker
- `eda.py`: reusable exploratory data analysis script
- `column_check.py`: tests whether a column helps by comparing CV scores with and without it
