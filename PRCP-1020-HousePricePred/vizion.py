# smart_recommender.py
"""
Smart Recommender - a lightweight recommendation engine for EDA & model suggestions.

Usage:
    from smart_recommender import analyze_df, data_summary, next_steps, suggest_models, plot_columns

Author: Milind's assistant (example)
"""

import pandas as pd
import numpy as np
import io
import math
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import compute_class_weight

sns.set_style("whitegrid")


def _safe_head(df: pd.DataFrame, n=5):
    try:
        return df.head(n)
    except Exception:
        return None


def analyze_df(df: pd.DataFrame) -> Dict[str, Any]:
    """Return meta-features and quick statistics about the dataframe."""
    n_rows, n_cols = df.shape
    dtypes = df.dtypes.to_dict()
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / n_rows).to_dict()
    duplicated = int(df.duplicated().sum())
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime', 'datetime64[ns]']).columns.tolist()

    n_numeric = len(numeric_cols)
    n_categorical = len(categorical_cols)
    n_datetime = len(datetime_cols)
    n_missing_cols = int((df.isnull().sum() > 0).sum())

    # basic distribution checks
    skewness = {}
    for col in numeric_cols:
        try:
            skewness[col] = float(df[col].skew())
        except Exception:
            skewness[col] = None

    # cardinality for categorical
    cardinality = {}
    for col in categorical_cols:
        try:
            cardinality[col] = int(df[col].nunique(dropna=False))
        except Exception:
            cardinality[col] = None

    meta = {
        'n_rows': n_rows,
        'n_cols': n_cols,
        'dtypes': dtypes,
        'n_numeric': n_numeric,
        'n_categorical': n_categorical,
        'n_datetime': n_datetime,
        'n_missing_cols': n_missing_cols,
        'missing_count': missing_count.to_dict(),
        'missing_pct': missing_pct,
        'duplicated_rows': duplicated,
        'numeric_columns': numeric_cols,
        'categorical_columns': categorical_cols,
        'datetime_columns': datetime_cols,
        'skewness': skewness,
        'cardinality': cardinality,
    }
    return meta


def data_summary(df: pd.DataFrame, show: bool = True) -> Dict[str, Any]:
    """Print and return a concise summary for quick notebook use."""
    meta = analyze_df(df)
    summary = {
        'shape': (meta['n_rows'], meta['n_cols']),
        'num_numeric': meta['n_numeric'],
        'num_categorical': meta['n_categorical'],
        'num_datetime': meta['n_datetime'],
        'num_missing_columns': meta['n_missing_cols'],
        'num_duplicated_rows': meta['duplicated_rows'],
        'top_missing_cols': sorted(meta['missing_pct'].items(), key=lambda x: -x[1])[:10],
        'top_cardinality': sorted(meta['cardinality'].items(), key=lambda x: -x[1])[:10] if meta['cardinality'] else [],
    }
    if show:
        print("DATA SUMMARY")
        print("-----------")
        print("Shape:", summary['shape'])
        print("Numeric columns:", summary['num_numeric'])
        print("Categorical columns:", summary['num_categorical'])
        print("Datetime columns:", summary['num_datetime'])
        print("Columns with missing values:", summary['num_missing_columns'])
        print("Duplicated rows:", summary['num_duplicated_rows'])
        print("Top missing columns (col, %):")
        for c, p in summary['top_missing_cols']:
            print(f"  - {c}: {p:.2%}")
        if summary['top_cardinality']:
            print("High cardinality categorical columns (sample):")
            for c, card in summary['top_cardinality']:
                print(f"  - {c}: {card}")
        print("\nSample rows:")
        display = _safe_head(df, 5)
        try:
            from IPython.display import display as _display
            _display(display)
        except Exception:
            print(display)
    return summary


def plot_columns(df: pd.DataFrame, cols: Optional[List[str]] = None, max_plots: int = 8) -> None:
    """
    Smart plotting: numeric -> hist + box, categorical -> countplot, datetime -> line counts by time unit.
    Shows up to max_plots columns (prioritizes columns with issues: missing, high skew, high cardinality).
    """
    if cols is None:
        cols = list(df.columns)

    # score columns by need-to-inspect
    scores = []
    for c in cols:
        score = 0
        if df[c].isnull().sum() > 0:
            score += 2
        if c in df.select_dtypes(include=[np.number]).columns:
            # skew
            try:
                s = abs(df[c].skew())
                if not math.isnan(s):
                    score += min(2, s)
            except Exception:
                pass
        if c in df.select_dtypes(include=['object', 'category']).columns:
            score += min(2, df[c].nunique() / max(1, len(df)))
        scores.append((c, score))
    scores.sort(key=lambda x: -x[1])
    selected = [c for c, _ in scores][:max_plots]

    for c in selected:
        plt.figure(figsize=(8, 4))
        if c in df.select_dtypes(include=[np.number]).columns:
            ax = plt.subplot(1, 2, 1)
            sns.histplot(df[c].dropna(), kde=True)
            ax.set_title(f"Distribution: {c}")
            ax2 = plt.subplot(1, 2, 2)
            sns.boxplot(x=df[c].dropna())
            ax2.set_title(f"Boxplot: {c}")
            plt.tight_layout()
            plt.show()
        elif c in df.select_dtypes(include=['object', 'category', 'bool']).columns:
            plt.figure(figsize=(6, 4))
            vc = df[c].value_counts(dropna=False).iloc[:30]
            sns.barplot(x=vc.values, y=vc.index)
            plt.title(f"Value counts: {c} (top {len(vc)})")
            plt.xlabel("count")
            plt.show()
        elif np.issubdtype(df[c].dtype, np.datetime64) or c in df.select_dtypes(include=['datetime']).columns:
            tmp = df.set_index(pd.to_datetime(df[c], errors='coerce'))[c].resample('D').count()
            if tmp.dropna().shape[0] > 0:
                tmp.plot(figsize=(10, 3))
                plt.title(f"Time-series frequency: {c}")
                plt.ylabel("count")
                plt.show()
            else:
                print(f"Column {c}: not enough datelike values to plot.")
        else:
            print(f"Column {c}: unsupported dtype for automatic plotting.")


def detect_outliers(df: pd.DataFrame, cols: Optional[List[str]] = None, method: str = 'iqr') -> Dict[str, int]:
    """Simple outlier detection. Returns count of outliers per numeric column for chosen method."""
    numeric = cols if cols is not None else df.select_dtypes(include=[np.number]).columns.tolist()
    outlier_counts = {}
    for c in numeric:
        s = df[c].dropna()
        if s.empty:
            outlier_counts[c] = 0
            continue
        if method == 'iqr':
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_counts[c] = int(((s < lower) | (s > upper)).sum())
        elif method == 'zscore':
            mean = s.mean()
            std = s.std()
            if std == 0 or pd.isna(std):
                outlier_counts[c] = 0
            else:
                z = (s - mean) / std
                outlier_counts[c] = int((abs(z) > 3).sum())
        else:
            raise ValueError("method must be 'iqr' or 'zscore'")
    return outlier_counts


def fix_missing(df: pd.DataFrame, strategy: str = 'auto', fill_value: Optional[Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Returns a COPY of df where missing values are handled according to the strategy.

    strategy:
      - 'auto'    : numeric -> median, categorical -> mode, datetime -> forward-fill
      - 'drop_rows': drop rows with any missing
      - 'drop_cols' : drop columns with >50% missing (threshold adjustable later)
      - 'fill_mean'/'fill_median'/'fill_mode' : global fill
      - 'custom' : use fill_value param (single scalar for all)
    """
    df2 = df.copy()
    report = {}
    n_rows_before = df2.shape[0]

    if strategy == 'drop_rows':
        df2 = df2.dropna(axis=0)
        report['dropped_rows'] = n_rows_before - df2.shape[0]
        return df2, report
    if strategy == 'drop_cols':
        thresh = df2.shape[0] * 0.5
        cols_before = df2.shape[1]
        df2 = df2.dropna(axis=1, thresh=thresh)
        report['dropped_cols'] = cols_before - df2.shape[1]
        return df2, report

    if strategy == 'auto':
        for c in df2.columns:
            if df2[c].isnull().sum() == 0:
                continue
            if c in df2.select_dtypes(include=[np.number]).columns:
                val = df2[c].median()
                df2[c] = df2[c].fillna(val)
                report[c] = {'method': 'median', 'filled': int(df2[c].isnull().sum() == 0)}
            elif c in df2.select_dtypes(include=['datetime', 'datetime64[ns]']).columns:
                df2[c] = df2[c].fillna(method='ffill').fillna(method='bfill')
                report[c] = {'method': 'ffill/bfill'}
            else:
                try:
                    mode = df2[c].mode(dropna=True)
                    if not mode.empty:
                        df2[c] = df2[c].fillna(mode[0])
                        report[c] = {'method': 'mode'}
                    else:
                        df2[c] = df2[c].fillna('missing')
                        report[c] = {'method': 'placeholder_missing'}
                except Exception:
                    df2[c] = df2[c].fillna('missing')
                    report[c] = {'method': 'placeholder_missing'}
        return df2, report

    if strategy in {'fill_mean', 'fill_median', 'fill_mode'}:
        for c in df2.columns:
            if df2[c].isnull().sum() == 0:
                continue
            if strategy == 'fill_mean' and c in df2.select_dtypes(include=[np.number]).columns:
                df2[c] = df2[c].fillna(df2[c].mean())
            elif strategy == 'fill_median' and c in df2.select_dtypes(include=[np.number]).columns:
                df2[c] = df2[c].fillna(df2[c].median())
            else:
                mode = df2[c].mode(dropna=True)
                df2[c] = df2[c].fillna(mode[0] if not mode.empty else 'missing')
        return df2, {'strategy': strategy}

    if strategy == 'custom':
        if fill_value is None:
            raise ValueError("fill_value must be provided for custom strategy")
        df2 = df2.fillna(fill_value)
        return df2, {'strategy': 'custom', 'fill_value': fill_value}

    raise ValueError(f"Unknown strategy: {strategy}")


def _infer_task_type(df: pd.DataFrame, target: Optional[str] = None) -> Tuple[str, Optional[List[str]]]:
    """Infer recommended ML task type from dataset & optional target name."""
    if target is None:
        # no target: recommend unsupervised if many rows, otherwise EDA
        return 'unsupervised', None
    if target not in df.columns:
        return 'unknown', None
    y = df[target]
    n_unique = int(y.nunique(dropna=False))
    if y.dtype in [np.float64, np.float32, np.int64, np.int32] and n_unique > max(20, 0.02 * len(y)):
        return 'regression', None
    else:
        # classification candidate
        labels = y.dropna().unique().tolist()
        return 'classification', labels


def suggest_models(meta: Dict[str, Any], target: Optional[str] = None, df: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
    """
    Suggest a small list of candidate models with reasons (rule-based).
    meta: output of analyze_df
    target: column name if known
    """
    suggestions = []
    # If no target given -> clustering / unsupervised suggestion
    if target is None:
        suggestions.append({'model': 'KMeans/DBSCAN', 'reason': 'No target provided. Consider clustering or dimensionality reduction (PCA/UMAP).'})
        suggestions.append({'model': 'PCA/UMAP', 'reason': 'Good for high dimensional numeric data and visualization.'})
        return suggestions

    # If df provided, try to get target details
    if df is not None and target in df.columns:
        task, labels = _infer_task_type(df, target)
    else:
        # fallback using meta only
        task = 'classification' if meta.get('n_categorical', 0) > 0 else 'regression'

    n_rows = meta.get('n_rows', 0)
    n_numeric = meta.get('n_numeric', 0)
    n_categorical = meta.get('n_categorical', 0)

    if task == 'regression':
        suggestions.append({'model': 'LinearRegression', 'reason': 'Simple baseline for regression; low variance if features are not many.'})
        suggestions.append({'model': 'RandomForestRegressor', 'reason': 'Robust to outliers and works with mixed features.'})
        suggestions.append({'model': 'XGBoost/LightGBM', 'reason': 'Powerful gradient boosters — good next step.'})
    elif task == 'classification':
        # try to detect if binary / imbalanced
        if df is not None:
            y = df[target]
            n_classes = int(y.nunique(dropna=False))
            class_counts = y.value_counts(dropna=False)
            imbalance_ratio = None
            if len(class_counts) > 0:
                imbalance_ratio = class_counts.max() / max(1, class_counts.min()) if class_counts.min() > 0 else None
            if n_classes == 2:
                suggestions.append({'model': 'LogisticRegression', 'reason': 'Simple, interpretable baseline for binary tasks.'})
                suggestions.append({'model': 'RandomForestClassifier', 'reason': 'Non-linear baseline that handles mixed types.'})
                if n_rows > 5000:
                    suggestions.append({'model': 'LightGBM/XGBoost', 'reason': 'Gradient boosters scale well and are often strong performers.'})
                if imbalance_ratio is not None and imbalance_ratio > 5:
                    suggestions.append({'model': 'RandomForest/LightGBM with class_weight or resampling', 'reason': f'Imbalanced classes detected (ratio ≈ {imbalance_ratio:.1f}). Consider class_weights, SMOTE or resampling.'})
            else:
                suggestions.append({'model': 'RandomForest/LightGBM', 'reason': 'Multiclass capable ensemble models; good baseline.'})
                suggestions.append({'model': 'One-vs-Rest with Linear Models', 'reason': 'If many classes with sparse features.'})
        else:
            suggestions.append({'model': 'RandomForestClassifier', 'reason': 'General-purpose baseline for classification.'})
            suggestions.append({'model': 'LightGBM/XGBoost', 'reason': 'Strong ensemble options.'})
    else:
        suggestions.append({'model': 'Explore dataset', 'reason': 'Could not infer task; please specify target column.'})

    return suggestions


def next_steps(df: pd.DataFrame, target: Optional[str] = None) -> List[str]:
    """
    Core recommendation engine: returns an ordered list of actions to take next.
    Designed to be human-readable so you can paste it into your notebook or checklist.
    """
    meta = analyze_df(df)
    steps = []

    # High level suggestions
    steps.append("1. Problem definition: Confirm objective and target variable (if any).")

    # Missing values
    if meta['n_missing_cols'] > 0:
        # list top missing columns
        missing_sorted = sorted(meta['missing_pct'].items(), key=lambda x: -x[1])
        top = missing_sorted[0]
        steps.append(f"2. Missing values detected in {meta['n_missing_cols']} columns. Check top: {top[0]} ({top[1]:.2%} missing). Consider fix_missing(df, strategy='auto') or decide domain-specific imputation.")
    else:
        steps.append("2. No missing values detected (nice).")

    # Duplicates
    if meta['duplicated_rows'] > 0:
        steps.append(f"3. Found {meta['duplicated_rows']} duplicated rows. Consider df.drop_duplicates() after verifying duplicates are true duplicates.")
    else:
        steps.append("3. No duplicate rows detected.")

    # Cardinality warnings for categorical columns
    high_card_cols = [c for c, card in meta.get('cardinality', {}).items() if card is not None and card > 100]
    if high_card_cols:
        steps.append(f"4. High-cardinality categorical columns detected: {high_card_cols[:5]}. Consider encoding strategies (target encoding, hashing) or grouping rare values.")
    else:
        steps.append("4. No high-cardinality categorical columns (or manageable).")

    # Numeric vs categorical
    if meta['n_numeric'] == 0:
        steps.append("5. No numeric columns detected; check if numeric columns are stored as objects (strings). Try df.apply(pd.to_numeric) on suspect columns.")
    else:
        steps.append(f"5. {meta['n_numeric']} numeric columns detected. Check distributions (skewness) and outliers: use detect_outliers().")

    # If target present: task inference and evaluation hints
    if target:
        if target not in df.columns:
            steps.append(f"6. Target column '{target}' not found in dataframe — make sure you passed the correct name.")
        else:
            task, labels = _infer_task_type(df, target)
            if task == 'classification':
                class_counts = df[target].value_counts(dropna=False)
                steps.append(f"6. Task: Classification. Target '{target}' has {len(class_counts)} classes. Check class balance: top counts -> {class_counts.iloc[:5].to_dict()}. Consider stratified split.")
                if class_counts.min() == 0:
                    steps.append("   - Warning: some classes might have zero instances after filtering/cleaning.")
                if class_counts.max() / max(1, class_counts.min()) > 5:
                    steps.append("   - High class imbalance detected. Add resampling or use class_weight (in RandomForest/Logistic) or specialized algorithms.")
            elif task == 'regression':
                steps.append(f"6. Task: Regression. Target '{target}' looks continuous. Check distribution and outliers; consider log transform if very skewed.")
            else:
                steps.append("6. Task: Could not determine; inspect target manually.")

    else:
        steps.append("6. No target provided: consider unsupervised exploration (PCA, clustering), or decide on a target and labeling strategy.")

    # Modeling advice
    model_suggestions = suggest_models(meta, target=target, df=df)
    steps.append("7. Modeling suggestions (baselines):")
    for m in model_suggestions[:5]:
        steps.append(f"   - {m['model']}: {m['reason']}")

    # Final recommended order
    steps.append("8. Recommended workflow order:")
    steps.append("   A) Quick summary (data_summary).")
    steps.append("   B) Fix missing & duplicates (fix_missing).")
    steps.append("   C) EDA & visual checks (plot_columns).")
    steps.append("   D) Feature engineering (encoding, scaling).")
    steps.append("   E) Train simple baseline models with cross-validation.")
    steps.append("   F) Improve (tuning/features), then package/report results.")

    return steps


def generate_basic_report(df: pd.DataFrame, filename: Optional[str] = None) -> str:
    """
    Produce a small HTML report (string). If filename provided, saves to that file.
    The report contains summary metrics and small tables.
    """
    meta = analyze_df(df)
    buf = io.StringIO()
    buf.write("<html><head><title>Basic EDA Report</title></head><body>")
    buf.write("<h1>Basic EDA Report</h1>")
    buf.write(f"<p>Rows: {meta['n_rows']}, Columns: {meta['n_cols']}</p>")
    buf.write("<h2>Missing Values (top 20)</h2><ul>")
    for col, pct in sorted(meta['missing_pct'].items(), key=lambda x: -x[1])[:20]:
        buf.write(f"<li>{col}: {pct:.2%}</li>")
    buf.write("</ul>")

    buf.write("<h2>Columns by type</h2><ul>")
    buf.write(f"<li>Numeric: {len(meta['numeric_columns'])} — {meta['numeric_columns'][:20]}</li>")
    buf.write(f"<li>Categorical: {len(meta['categorical_columns'])} — {meta['categorical_columns'][:20]}</li>")
    buf.write("</ul>")

    buf.write("<h2>Top 10 rows</h2>")
    buf.write(df.head(10).to_html(index=False))
    buf.write("</body></html>")

    html = buf.getvalue()
    if filename:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
        return filename
    return html


# Example CLI / Notebook usage as module-level snippet (for quick copy-paste)
if __name__ == "__main__":
    # Demo usage - replace with your own CSV
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="Path to CSV file for demo", required=False)
    parser.add_argument("--target", help="Optional target column name", required=False)
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        print("=== DATA SUMMARY ===")
        data_summary(df)
        print("=== NEXT STEPS ===")
        for s in next_steps(df, target=args.target):
            print(s)
        print("\n=== MODEL SUGGESTIONS ===")
        meta = analyze_df(df)
        for r in suggest_models(meta, target=args.target, df=df):
            print("-", r)
    else:
        print("Run this module with --csv yourfile.csv to see recommendations.")
