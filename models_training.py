import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from config import EPOCHS, OUTPUT_DIR, SEED


def evaluate(y_true, proba, labels=None):
    pred = np.argmax(proba, axis=1)
    proba_clipped = np.clip(proba, 1e-9, 1 - 1e-9)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "loss": float(log_loss(y_true, proba_clipped, labels=labels)),
        "precision_macro": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }


class OvRLinearRegression:
    def __init__(self, epochs=EPOCHS, seed=SEED):
        self.epochs = epochs
        self.seed = seed
        self.models = []
        self.scaler = StandardScaler()

    def _new_model(self, seed):
        return SGDRegressor(
            penalty=None, learning_rate="constant", eta0=0.01,
            max_iter=self.epochs, random_state=seed,
        )

    def fit_with_history(self, X_tr, y_tr, X_te, y_te, n_classes, class_weights=None):
        Xs_tr = self.scaler.fit_transform(X_tr)
        Xs_te = self.scaler.transform(X_te)
        y_onehot = np.eye(n_classes)[y_tr]
        history = []
        for c in range(n_classes):
            self.models.append(self._new_model(self.seed + c))
        for epoch in range(1, self.epochs + 1):
            scores_tr = np.column_stack([m.partial_fit(Xs_tr, y_onehot[:, c]).predict(Xs_tr) for c, m in enumerate(self.models)])
            scores_te = np.column_stack([m.predict(Xs_te) for m in self.models])
            proba_te = expit(scores_te)
            proba_te = proba_te / proba_te.sum(axis=1, keepdims=True)
            metrics = evaluate(y_te, proba_te)
            metrics["epoch"] = epoch
            history.append(metrics)
        self.classes_ = np.arange(n_classes)
        return pd.DataFrame(history)

    def predict_proba(self, X):
        Xs = self.scaler.transform(X)
        scores = np.column_stack([m.predict(Xs) for m in self.models])
        proba = expit(scores)
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba


class RandomForestEpochs:
    def __init__(self, epochs=EPOCHS, trees_per_epoch=5, seed=SEED):
        self.epochs = epochs
        self.trees_per_epoch = trees_per_epoch
        self.seed = seed
        self.clf = RandomForestClassifier(
            n_estimators=trees_per_epoch, warm_start=True,
            random_state=seed, n_jobs=-1,
        )

    def fit_with_history(self, X_tr, y_tr, X_te, y_te):
        classes = np.unique(y_tr)
        weights = compute_class_weight("balanced", classes=classes, y=y_tr)
        self.clf.class_weight = dict(zip(classes.tolist(), weights.tolist()))
        history = []
        for epoch in range(1, self.epochs + 1):
            if epoch > 1:
                self.clf.n_estimators += self.trees_per_epoch
            self.clf.fit(X_tr, y_tr)
            metrics = evaluate(y_te, self.clf.predict_proba(X_te))
            metrics["epoch"] = epoch
            history.append(metrics)
        return pd.DataFrame(history)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


class XGBoostEpochs:
    def __init__(self, epochs=EPOCHS, rounds_per_epoch=5, seed=SEED, num_class=2):
        self.epochs = epochs
        self.rounds_per_epoch = rounds_per_epoch
        self.num_class = num_class
        objective = "binary:logistic" if num_class == 2 else "multi:softprob"
        self.params = {
            "objective": objective,
            "eval_metric": "logloss",
            "max_depth": 5,
            "eta": 0.1,
            "tree_method": "hist",
            "seed": seed,
        }
        if num_class > 2:
            self.params["num_class"] = num_class
        self.booster = None

    def fit_with_history(self, X_tr, y_tr, X_te, y_te, sample_weights=None):
        dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=sample_weights)
        dtest = xgb.DMatrix(X_te)
        history = []
        for epoch in range(1, self.epochs + 1):
            self.booster = xgb.train(
                self.params, dtrain,
                num_boost_round=self.rounds_per_epoch,
                xgb_model=self.booster,
            )
            raw = self.booster.predict(dtest)
            if self.num_class == 2:
                p1 = raw.reshape(-1, 1)
                proba = np.hstack([1 - p1, p1])
            else:
                proba = raw.reshape(-1, self.num_class)
            metrics = evaluate(np.asarray(y_te), proba)
            metrics["epoch"] = epoch
            history.append(metrics)
        return pd.DataFrame(history)

    def predict_proba(self, X):
        dmatrix = xgb.DMatrix(X)
        raw = self.booster.predict(dmatrix)
        if self.num_class == 2:
            p1 = raw.reshape(-1, 1)
            return np.hstack([1 - p1, p1])
        return raw.reshape(-1, self.num_class)


def balanced_weights(y):
    counts = np.bincount(y)
    weights = np.array([len(y) / (len(counts) * counts[c]) for c in y], dtype=float)
    return weights


def save_curves(history_df, tag):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for model_name, df in history_df.groupby("model"):
        axes[0].plot(df["epoch"], df["accuracy"], label=model_name)
        axes[1].plot(df["epoch"], df["loss"], label=model_name)
        axes[2].plot(df["epoch"], df["precision_macro"], label=model_name)
    axes[0].set_title(f"{tag} - Accuracy vs Epoch")
    axes[1].set_title(f"{tag} - Log Loss vs Epoch")
    axes[2].set_title(f"{tag} - Precision (macro) vs Epoch")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("Score")
    fig.tight_layout()
    path = OUTPUT_DIR / f"{tag}_training_curves.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_confusion(y_true, proba, class_names, tag, model_name):
    pred = np.argmax(proba, axis=1)
    cm = confusion_matrix(y_true, pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    short = [c[:16] for c in class_names]
    ax.set_xticklabels(short, rotation=45, ha="right")
    ax.set_yticklabels(short)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{tag}: {model_name}")
    fig.colorbar(im)
    fig.tight_layout()
    path = OUTPUT_DIR / f"{tag}_confusion_{model_name.replace(' ', '_').lower()}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def train_all_models(tag, X_tr, X_te, y_tr, y_te, feature_names, multiclass=False):
    OUTPUT_DIR.mkdir(exist_ok=True)
    classes = sorted(set(list(y_tr)) | set(list(y_te)))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_tr_idx = np.array([class_to_idx[c] for c in y_tr])
    y_te_idx = np.array([class_to_idx[c] for c in y_te])

    X_tr_np = np.asarray(X_tr, dtype=float)
    X_te_np = np.asarray(X_te, dtype=float)

    models = {}
    histories = []
    final_rows = []

    lin = OvRLinearRegression()
    hist = lin.fit_with_history(X_tr_np, y_tr_idx, X_te_np, y_te_idx, len(classes))
    hist["model"] = "Linear Regression"
    histories.append(hist)
    models["linear_regression"] = lin
    final_rows.append({"model": "Linear Regression", **hist.iloc[-1].to_dict()})
    print(f"[{tag}] Linear Regression done: acc={final_rows[-1]['accuracy']:.4f} loss={final_rows[-1]['loss']:.4f}")

    rf = RandomForestEpochs()
    hist = rf.fit_with_history(X_tr_np, y_tr_idx, X_te_np, y_te_idx)
    hist["model"] = "Random Forest"
    histories.append(hist)
    models["random_forest"] = rf
    final_rows.append({"model": "Random Forest", **hist.iloc[-1].to_dict()})
    print(f"[{tag}] Random Forest done: acc={final_rows[-1]['accuracy']:.4f} loss={final_rows[-1]['loss']:.4f}")

    sw = balanced_weights(y_tr_idx) if len(classes) > 2 else None
    xgbm = XGBoostEpochs(num_class=len(classes))
    hist = xgbm.fit_with_history(X_tr_np, y_tr_idx, X_te_np, y_te_idx, sample_weights=sw)
    hist["model"] = "XGBoost"
    histories.append(hist)
    models["xgboost"] = xgbm
    final_rows.append({"model": "XGBoost", **hist.iloc[-1].to_dict()})
    print(f"[{tag}] XGBoost done: acc={final_rows[-1]['accuracy']:.4f} loss={final_rows[-1]['loss']:.4f}")

    all_hist = pd.concat(histories, ignore_index=True)
    all_hist.to_csv(OUTPUT_DIR / f"{tag}_history.csv", index=False)
    save_curves(all_hist, tag)

    final_df = pd.DataFrame(final_rows)[[
        "model", "accuracy", "loss", "precision_macro", "precision_weighted",
        "recall_macro", "f1_macro",
    ]]
    final_df.to_csv(OUTPUT_DIR / f"{tag}_final_metrics.csv", index=False)

    best_row = final_df.sort_values("accuracy", ascending=False).iloc[0]
    best_key = {
        "Linear Regression": "linear_regression",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
    }[best_row["model"]]
    best_proba = models[best_key].predict_proba(X_te_np)

    report_txt = classification_report(
        y_te_idx, np.argmax(best_proba, axis=1),
        target_names=[str(c) for c in classes], zero_division=0,
    )
    (OUTPUT_DIR / f"{tag}_classification_report.txt").write_text(report_txt)
    save_confusion(y_te_idx, best_proba, [str(c) for c in classes], tag, str(best_row["model"]))

    joblib.dump(models["linear_regression"], OUTPUT_DIR / f"{tag}_linear.pkl")
    joblib.dump(models["random_forest"], OUTPUT_DIR / f"{tag}_rf.pkl")
    models["xgboost"].booster.save_model(str(OUTPUT_DIR / f"{tag}_xgboost.json"))

    summary = {
        "dataset": tag,
        "classes": [str(c) for c in classes],
        "features": [str(f) for f in feature_names],
        "feature_count": int(X_tr_np.shape[1]),
        "train_samples": int(X_tr_np.shape[0]),
        "test_samples": int(X_te_np.shape[0]),
        "epochs": EPOCHS,
        "final_metrics": final_df.to_dict(orient="records"),
        "best_model": str(best_row["model"]),
    }
    with open(OUTPUT_DIR / f"{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return models, final_df, best_proba, class_to_idx
