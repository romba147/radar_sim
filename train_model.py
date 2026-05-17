"""
train_model.py — XGBoost training pipeline for intercept probability prediction.

Loads Monte Carlo engagement data, trains an XGBoost binary classifier,
evaluates performance (AUC-ROC, calibration curve), and saves the model.
"""

import numpy as np
import os

from mc_simulator import generate_dataset, save_dataset, load_dataset
from feature_extraction import dataset_to_Xy, FEATURE_NAMES


def train(data_path: str = "mc_engagement_data.npz",
          model_path: str = "xgb_intercept_model.json",
          n_samples: int = 50000,
          seed: int = 42,
          generate_if_missing: bool = True,
          verbose: bool = True):
    """Full training pipeline: data → train → evaluate → save.

    Parameters
    ----------
    data_path : str
        Path to MC dataset (NPZ). Generated if missing and generate_if_missing=True.
    model_path : str
        Where to save the trained XGBoost model.
    n_samples : int
        Number of MC samples to generate (if generating).
    seed : int
        Random seed.
    generate_if_missing : bool
        If True and data_path doesn't exist, generate the dataset.
    verbose : bool
        Print progress and metrics.
    """
    # Lazy imports so the module can be imported without these installed
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (roc_auc_score, accuracy_score,
                                 precision_score, recall_score,
                                 f1_score, classification_report)
    from sklearn.calibration import calibration_curve

    # ── Step 1: Load or generate data ────────────────────────────────
    if not os.path.exists(data_path):
        if generate_if_missing:
            print(f"  Dataset not found at {data_path}, generating {n_samples} samples...")
            data = generate_dataset(n_samples=n_samples, seed=seed, verbose=verbose)
            save_dataset(data, data_path)
        else:
            raise FileNotFoundError(f"Dataset not found: {data_path}")
    else:
        print(f"  Loading dataset from {data_path}")

    data = load_dataset(data_path)
    X, y = dataset_to_Xy(data)

    if verbose:
        print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"  Hit rate: {y.mean()*100:.1f}%")

    # ── Step 2: Train/Val/Test split ─────────────────────────────────
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=seed, stratify=y_temp
        # 0.176 of 0.85 ≈ 0.15 of total
    )

    if verbose:
        print(f"  Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
        print(f"  Train hit rate: {y_train.mean()*100:.1f}%")

    # ── Step 3: Train XGBoost ────────────────────────────────────────
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
    )

    if verbose:
        print("\n  Training XGBoost...")

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=verbose,
    )

    # ── Step 4: Evaluate on test set ─────────────────────────────────
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n  ══ Test Set Results ══")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    # ── Step 5: Feature importance ───────────────────────────────────
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    print(f"\n  ══ Feature Importance (top 10) ══")
    for rank, idx in enumerate(sorted_idx[:10]):
        print(f"  {rank+1:>2d}. {FEATURE_NAMES[idx]:<25s} {importances[idx]:.4f}")

    # ── Step 6: Calibration analysis ─────────────────────────────────
    try:
        prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
        cal_error = np.mean(np.abs(prob_true - prob_pred))
        print(f"\n  ══ Calibration ══")
        print(f"  Mean calibration error: {cal_error:.4f}")
        print(f"  {'Bin center':>12s}  {'Actual':>8s}  {'Predicted':>10s}")
        for pt, pp in zip(prob_true, prob_pred):
            print(f"  {pp:>12.3f}  {pt:>8.3f}  {pp:>10.3f}")
    except Exception as e:
        print(f"  Calibration analysis skipped: {e}")

    # ── Step 7: Save model ───────────────────────────────────────────
    model.save_model(model_path)
    print(f"\n  Model saved → {model_path}")

    # ── Step 8: Plot diagnostics ─────────────────────────────────────
    _plot_diagnostics(model, X_test, y_test, y_pred_proba, importances)

    return model


def _plot_diagnostics(model, X_test, y_test, y_pred_proba, importances):
    """Generate diagnostic plots: ROC curve, calibration, feature importance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    from sklearn.calibration import calibration_curve

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── ROC Curve ────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = np.trapz(tpr, fpr)
    axes[0].plot(fpr, tpr, "b-", linewidth=2, label=f"AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Calibration Curve ────────────────────────────────────────────
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
    axes[1].plot(prob_pred, prob_true, "s-", color="green", linewidth=2, label="XGBoost")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
    axes[1].set_xlabel("Mean Predicted Probability")
    axes[1].set_ylabel("Fraction of Positives")
    axes[1].set_title("Calibration Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # ── Feature Importance ───────────────────────────────────────────
    sorted_idx = np.argsort(importances)
    top_n = min(len(FEATURE_NAMES), 17)
    axes[2].barh(range(top_n),
                 importances[sorted_idx[-top_n:]],
                 color="steelblue")
    axes[2].set_yticks(range(top_n))
    axes[2].set_yticklabels([FEATURE_NAMES[i] for i in sorted_idx[-top_n:]])
    axes[2].set_xlabel("Importance")
    axes[2].set_title("Feature Importance")

    plt.tight_layout()
    plt.savefig("xgb_diagnostics.png", dpi=150)
    plt.close()
    print(f"  Diagnostics plot saved → xgb_diagnostics.png")


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("    XGBOOST INTERCEPT MODEL — TRAINING PIPELINE")
    print("=" * 62)
    train()
