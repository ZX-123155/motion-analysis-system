"""
运动识别模型训练模块
- 支持 SVM 和 Random Forest
- 交叉验证 + 超参数调优
- 模型评估与保存
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings("ignore")


class MotionClassifier:
    """运动识别分类器"""

    def __init__(self, model_type="random_forest", random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_names = []

        if model_type == "random_forest":
            self.model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        elif model_type == "svm":
            self.model = SVC(random_state=random_state, probability=True)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def load_data(self, filepath):
        """加载特征数据"""
        df = pd.read_csv(filepath)
        # 排除非特征列
        exclude_cols = ["label", "window_start", "window_end", "sample_id"]
        self.feature_names = [c for c in df.columns if c not in exclude_cols]

        X = df[self.feature_names].values
        y = df["label"].values

        # 编码标签
        y_encoded = self.label_encoder.fit_transform(y)

        # 处理NaN和无穷值
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        return X, y_encoded

    def preprocess(self, X_train, X_test):
        """标准化"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def train(self, X_train, y_train, X_test, y_test, tune=True):
        """训练模型，可选超参数调优"""
        if tune and self.model_type == "random_forest":
            print("  Hyperparameter tuning for RandomForest...")
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            }
            grid = GridSearchCV(
                self.model, param_grid, cv=5, scoring="f1_weighted",
                n_jobs=-1, verbose=0
            )
            grid.fit(X_train, y_train)
            self.model = grid.best_estimator_
            print(f"  Best params: {grid.best_params_}")

        elif tune and self.model_type == "svm":
            print("  Hyperparameter tuning for SVM...")
            param_grid = {
                "C": [0.1, 1, 10],
                "gamma": ["scale", "auto"],
                "kernel": ["rbf", "linear"],
            }
            grid = GridSearchCV(
                self.model, param_grid, cv=5, scoring="f1_weighted",
                n_jobs=-1, verbose=0
            )
            grid.fit(X_train, y_train)
            self.model = grid.best_estimator_
            print(f"  Best params: {grid.best_params_}")
        else:
            self.model.fit(X_train, y_train)

        # 评估
        y_pred = self.model.predict(X_test)
        metrics = self.evaluate(y_test, y_pred)

        return metrics

    def evaluate(self, y_true, y_pred):
        """评估模型性能"""
        target_names = self.label_encoder.classes_

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }
        return metrics

    def plot_confusion_matrix(self, y_true, y_pred, save_path):
        """绘制混淆矩阵"""
        target_names = self.label_encoder.classes_
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=target_names,
               yticklabels=target_names,
               title="Confusion Matrix",
               ylabel="True Label",
               xlabel="Predicted Label")

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    def plot_feature_importance(self, save_path, top_n=20):
        """绘制特征重要性（仅RandomForest）"""
        if self.model_type != "random_forest":
            return

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(top_n), importances[indices][::-1], align="center")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([self.feature_names[i] for i in indices][::-1], fontsize=8)
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Top {top_n} Feature Importances")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    def save(self, model_dir):
        """保存模型和预处理工具"""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, model_dir / "classifier.pkl")
        joblib.dump(self.scaler, model_dir / "scaler.pkl")
        joblib.dump(self.label_encoder, model_dir / "label_encoder.pkl")
        joblib.dump(self.feature_names, model_dir / "feature_names.pkl")

        print(f"Model saved to {model_dir}")

    def predict(self, X):
        """预测"""
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        return self.label_encoder.inverse_transform(y_pred)

    def predict_proba(self, X):
        """预测概率"""
        X_scaled = self.scaler.transform(X)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_scaled)
        return None


def main():
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    model_dir = Path(__file__).parent.parent / "models"
    figures_dir = Path(__file__).parent.parent / "static" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    features_path = processed_dir / "features.csv"
    print(f"Loading features from {features_path}")

    # 训练 RandomForest
    print("\n=== Training Random Forest ===")
    rf = MotionClassifier("random_forest")
    X, y = rf.load_data(features_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_s, X_test_s = rf.preprocess(X_train, X_test)
    metrics_rf = rf.train(X_train_s, y_train, X_test_s, y_test, tune=True)
    print(f"  Random Forest Results: {json.dumps(metrics_rf, indent=2)}")

    # 训练 SVM
    print("\n=== Training SVM ===")
    svm = MotionClassifier("svm")
    X2, y2 = svm.load_data(features_path)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        X2, y2, test_size=0.2, random_state=42, stratify=y2
    )
    X_train2_s, X_test2_s = svm.preprocess(X_train2, X_test2)
    metrics_svm = svm.train(X_train2_s, y_train2, X_test2_s, y_test2, tune=True)
    print(f"  SVM Results: {json.dumps(metrics_svm, indent=2)}")

    # 选择更好的模型
    if metrics_rf["f1_weighted"] >= metrics_svm["f1_weighted"]:
        best_model = rf
        best_metrics = metrics_rf
        best_name = "Random Forest"
    else:
        best_model = svm
        best_metrics = metrics_svm
        best_name = "SVM"

    print(f"\n=== Best Model: {best_name} ===")
    print(json.dumps(best_metrics, indent=2))

    # 评估并保存图表
    y_pred = best_model.model.predict(X_test_s if best_name == "Random Forest" else X_test2_s)
    y_true = y_test if best_name == "Random Forest" else y_test2

    best_model.plot_confusion_matrix(y_true, y_pred, figures_dir / "confusion_matrix.png")
    best_model.plot_feature_importance(figures_dir / "feature_importance.png")

    # 分类报告
    report = classification_report(
        y_true, y_pred,
        target_names=best_model.label_encoder.classes_
    )
    print(f"\nClassification Report:\n{report}")

    # 保存
    best_model.save(model_dir)

    # 保存评估指标
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=2)

    print(f"\nConfusion matrix: {figures_dir / 'confusion_matrix.png'}")
    print(f"Feature importance: {figures_dir / 'feature_importance.png'}")


if __name__ == "__main__":
    main()
