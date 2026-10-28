"""
Reusable EDA (exploratory data analysis) script.

Usage:
    python eda.py data.csv
    python eda.py data.csv --target approved
    python eda.py data.csv --target approved --plots     (saves PNGs to eda_plots/)

Prints: structure, missing values, duplicates, target balance, outliers,
categorical issues (messy text, ID-like, high cardinality), correlations,
possible leaks, and a short list of suggested next steps.
"""
import argparse
import os
import numpy as np
import pandas as pd

pd.set_option("display.width", 150)
pd.set_option("display.max_columns", 30)


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--target", default=None, help="target column name")
    ap.add_argument("--plots", action="store_true", help="save plots")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    todo = []   # suggested next steps

    # 1. Structure --------------------------------------------------------
    section("1. STRUCTURE")
    print("shape:", df.shape)
    print(df.dtypes.to_string())
    print("\nfirst rows:")
    print(df.head().to_string())

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    if args.target:
        if args.target not in df.columns:
            raise SystemExit(f"Target '{args.target}' not in columns: {list(df.columns)}")
        num_cols = [c for c in num_cols if c != args.target]
        cat_cols = [c for c in cat_cols if c != args.target]

    # 2. Missing + duplicates --------------------------------------------
    section("2. MISSING VALUES AND DUPLICATES")
    miss = df.isna().sum()
    miss = pd.DataFrame({"missing": miss, "percent": (miss / len(df) * 100).round(1)})
    miss = miss[miss.missing > 0].sort_values("missing", ascending=False)
    print(miss.to_string() if len(miss) else "no missing values")
    dups = int(df.duplicated().sum())
    print("\nduplicate rows:", dups)
    if dups:
        todo.append("drop_duplicates()")
    if len(miss):
        todo.append("impute missing values (median for numeric, most frequent for categorical)")
    for c, row in miss.iterrows():
        if row.percent > 40:
            todo.append(f"column '{c}' is {row.percent}% missing - consider dropping it")

    # 3. Target -----------------------------------------------------------
    is_classification = None
    if args.target:
        section(f"3. TARGET: {args.target}")
        y = df[args.target]
        is_classification = (not pd.api.types.is_numeric_dtype(y)) or y.nunique() <= 10
        if y.isna().any():
            print("missing target rows:", int(y.isna().sum()))
            todo.append("drop rows where the target is missing")
        if is_classification:
            vc = y.value_counts(normalize=True).round(3)
            print("classification target; class shares:")
            print(vc.to_string())
            if vc.min() < 0.15:
                print("\n-> IMBALANCED: rare class is only", f"{vc.min():.1%}")
                todo.append("imbalanced target: use stratify, class_weight='balanced', "
                            "and judge by recall / precision / F1, not accuracy")
            else:
                todo.append("target is reasonably balanced: use stratify=y and accuracy/F1")
        else:
            print("regression target:")
            print(y.describe().round(2).to_string())
            sk = y.skew()
            print("skew:", round(sk, 2))
            if abs(sk) > 1:
                todo.append("target is skewed: consider log-transforming it")
            todo.append("regression: evaluate with R2 / RMSE, no stratify")

    # 4. Numeric features -------------------------------------------------
    section("4. NUMERIC FEATURES")
    if num_cols:
        print(df[num_cols].describe().round(2).T.to_string())
        out_rows = []
        for c in num_cols:
            s = df[c].dropna()
            q1, q3 = s.quantile(.25), s.quantile(.75)
            iqr = q3 - q1
            n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
            neg = int((s < 0).sum())
            out_rows.append((c, n_out, neg, round(s.skew(), 2)))
        info = pd.DataFrame(out_rows, columns=["column", "IQR_outliers", "negatives", "skew"])
        print("\noutliers / negatives / skew:")
        print(info.to_string(index=False))
        for c in num_cols:
            lc = c.lower()
            if (lc == "id" or lc.endswith("_id") or lc.endswith("id")) \
                    and df[c].nunique() > 0.9 * len(df):
                todo.append(f"'{c}' looks like an ID - drop it")
        for c, n_out, neg, sk in out_rows:
            if n_out > 0.05 * len(df):
                todo.append(f"'{c}' has many outliers ({n_out}) - check or cap them")
            if abs(sk) > 2:
                todo.append(f"'{c}' is highly skewed - consider a log transform")
        todo.append("scale numeric columns if you use KNN / SVC / KMeans / logistic regression")
    else:
        print("no numeric features")

    # 5. Categorical features --------------------------------------------
    section("5. CATEGORICAL FEATURES")
    if cat_cols:
        for c in cat_cols:
            s = df[c].dropna().astype(str)
            n = s.nunique()
            print(f"\n{c}: {n} unique values; top:")
            print(s.value_counts().head(5).to_string())
            if n == len(df):
                todo.append(f"'{c}' is unique per row (ID-like) - drop it")
            elif n > 50:
                todo.append(f"'{c}' has high cardinality ({n}) - one-hot will explode; "
                            "group rare values or use another encoding")
            norm = s.str.strip().str.lower().nunique()
            if norm < n:
                print(f"-> messy text: {n} raw values collapse to {norm} "
                      "after strip + lowercase")
                todo.append(f"clean text in '{c}' (strip spaces, fix case)")
        todo.append("one-hot encode categorical columns (handle_unknown='ignore')")
    else:
        print("no categorical features")

    # 6. Correlations -----------------------------------------------------
    section("6. CORRELATIONS")
    corr_cols = num_cols + ([args.target] if args.target and
                            pd.api.types.is_numeric_dtype(df[args.target]) else [])
    if len(corr_cols) >= 2:
        corr = df[corr_cols].corr()
        pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(num_cols)
                 for b in num_cols[i + 1:] if abs(corr.loc[a, b]) > 0.8]
        if pairs:
            print("highly correlated feature pairs (|r| > 0.8):")
            for a, b, r in pairs:
                print(f"  {a} ~ {b}: {r:.2f}")
            todo.append("highly correlated features are redundant - consider dropping one")
        else:
            print("no feature pairs with |r| > 0.8")
        if args.target and args.target in corr.columns:
            tc = corr[args.target].drop(args.target).abs().sort_values(ascending=False)
            print(f"\ncorrelation with target ({args.target}), absolute:")
            print(tc.round(3).to_string())
            for c, r in tc.items():
                if r > 0.95:
                    print(f"-> '{c}' is almost perfectly correlated with the target: possible LEAK")
                    todo.append(f"check '{c}' for target leakage")
    else:
        print("not enough numeric columns")

    # 7. Plots ------------------------------------------------------------
    if args.plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            os.makedirs("eda_plots", exist_ok=True)
            if num_cols:
                df[num_cols].hist(figsize=(10, 7), bins=30)
                plt.tight_layout()
                plt.savefig("eda_plots/histograms.png")
                plt.close()
            if len(corr_cols) >= 2:
                fig, ax = plt.subplots(figsize=(7, 6))
                im = ax.imshow(df[corr_cols].corr(), cmap="coolwarm", vmin=-1, vmax=1)
                ax.set_xticks(range(len(corr_cols)), corr_cols, rotation=90)
                ax.set_yticks(range(len(corr_cols)), corr_cols)
                fig.colorbar(im)
                plt.tight_layout()
                plt.savefig("eda_plots/correlation.png")
                plt.close()
            print("\nplots saved to eda_plots/")
        except ImportError:
            print("\nmatplotlib not installed: pip install matplotlib")

    # 8. Summary ----------------------------------------------------------
    section("SUGGESTED NEXT STEPS")
    seen = set()
    for t in todo:
        if t not in seen:
            print(" -", t)
            seen.add(t)
    print(" - split into train/test BEFORE fitting any preprocessing")


if __name__ == "__main__":
    main()
