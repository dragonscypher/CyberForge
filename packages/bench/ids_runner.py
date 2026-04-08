"""IDS Structured Runner — classical IDS baselines with sklearn.

Runs RF / XGBoost on tabular IDS datasets (NSL-KDD, CICIDS2017, etc.)
and emits the plan.md normalized benchmark summary schema.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

log = logging.getLogger(__name__)


class IDSBenchmarkResult(BaseModel):
    """Normalized benchmark result matching plan.md Section 8 schema."""
    dataset: str = ""
    model_type: str = ""
    quality: dict[str, float] = {}
    reliability: dict[str, float] = {}
    efficiency: dict[str, float] = {}
    confusion_matrix: list[list[int]] = []
    class_report: dict[str, Any] = {}
    success: bool = True
    error: Optional[str] = None


class IDSStructuredRunner:
    """Run classical ML baselines on IDS datasets.

    Supports: RandomForest, XGBoost, hybrid ensemble.
    """

    def __init__(self, random_state: int = 42):
        self._rs = random_state

    def run_nsl_kdd(
        self,
        classifier: str = "rf",
        n_estimators: int = 100,
        test_ratio: float = 0.2,
        train_path: str | None = None,
    ) -> IDSBenchmarkResult:
        """Run IDS benchmark on NSL-KDD dataset."""
        from packages.cyber.datasets import load_nsl_kdd

        # Find the training data
        if train_path is None:
            from pathlib import Path
            candidates = [
                Path("KDDTrain+.txt"),
                Path("data/KDDTrain+.txt"),
                Path("KDDTrain+_20Percent.txt"),
            ]
            for c in candidates:
                if c.exists():
                    train_path = str(c)
                    break
            if train_path is None:
                return IDSBenchmarkResult(
                    dataset="nsl_kdd", model_type=classifier,
                    success=False, error="KDDTrain+.txt not found",
                )

        t0 = time.perf_counter()
        df = load_nsl_kdd(train_path)
        load_ms = (time.perf_counter() - t0) * 1000

        return self._run_tabular(
            df=df,
            label_col="label2",
            dataset_name="nsl_kdd",
            classifier=classifier,
            n_estimators=n_estimators,
            test_ratio=test_ratio,
            load_ms=load_ms,
        )

    def run_dataset(
        self,
        df: pd.DataFrame,
        label_col: str = "label2",
        dataset_name: str = "custom",
        classifier: str = "rf",
        n_estimators: int = 100,
        test_ratio: float = 0.2,
    ) -> IDSBenchmarkResult:
        """Run IDS benchmark on an arbitrary DataFrame."""
        return self._run_tabular(
            df=df,
            label_col=label_col,
            dataset_name=dataset_name,
            classifier=classifier,
            n_estimators=n_estimators,
            test_ratio=test_ratio,
        )

    def run_cross_dataset(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        label_col: str = "label2",
        dataset_name: str = "transfer",
        classifier: str = "rf",
        n_estimators: int = 100,
    ) -> IDSBenchmarkResult:
        """Train on one dataset, test on another (transfer evaluation)."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import (accuracy_score,
                                         average_precision_score,
                                         classification_report,
                                         confusion_matrix, f1_score,
                                         roc_auc_score)

            t0 = time.perf_counter()
            X_train, y_train = self._prepare_features(train_df, label_col)
            X_test, y_test = self._prepare_features(test_df, label_col)

            # Align columns
            common_cols = list(set(X_train.columns) & set(X_test.columns))
            if not common_cols:
                return IDSBenchmarkResult(
                    dataset=dataset_name, success=False,
                    error="No common features between train and test datasets",
                )
            X_train = X_train[common_cols]
            X_test = X_test[common_cols]

            clf = self._build_classifier(classifier, n_estimators)
            clf.fit(X_train, y_train)
            train_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            y_pred = clf.predict(X_test)
            y_prob = self._get_proba(clf, X_test)
            infer_ms = (time.perf_counter() - t0) * 1000

            return self._build_result(
                y_test, y_pred, y_prob, dataset_name, classifier,
                train_ms=train_ms, infer_ms=infer_ms, n_samples=len(y_test),
            )
        except Exception as e:
            log.exception("Cross-dataset IDS benchmark failed")
            return IDSBenchmarkResult(dataset=dataset_name, model_type=classifier, success=False, error=str(e))

    def _run_tabular(
        self,
        df: pd.DataFrame,
        label_col: str,
        dataset_name: str,
        classifier: str,
        n_estimators: int,
        test_ratio: float,
        load_ms: float = 0.0,
    ) -> IDSBenchmarkResult:
        try:
            from sklearn.model_selection import train_test_split

            t0 = time.perf_counter()
            X, y = self._prepare_features(df, label_col)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_ratio, random_state=self._rs, stratify=y,
            )

            clf = self._build_classifier(classifier, n_estimators)
            clf.fit(X_train, y_train)
            train_ms = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            y_pred = clf.predict(X_test)
            y_prob = self._get_proba(clf, X_test)
            infer_ms = (time.perf_counter() - t0) * 1000

            return self._build_result(
                y_test, y_pred, y_prob, dataset_name, classifier,
                train_ms=train_ms, infer_ms=infer_ms,
                n_samples=len(y_test), load_ms=load_ms,
            )
        except Exception as e:
            log.exception("IDS benchmark failed")
            return IDSBenchmarkResult(dataset=dataset_name, model_type=classifier, success=False, error=str(e))

    def _prepare_features(self, df: pd.DataFrame, label_col: str):
        """Prepare numeric features from a DataFrame."""
        if pd.api.types.is_string_dtype(df[label_col]):
            y = (df[label_col] != "normal").astype(int)
        else:
            y = df[label_col].astype(int)
        X = df.drop(columns=[c for c in df.columns if c.startswith("label") or c == "attack_cat" or c == "labels" or c == "difficulty"], errors="ignore")

        # Keep only numeric columns
        X = X.select_dtypes(include=[np.number])

        # Handle NaN/inf
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        return X, y

    def _build_classifier(self, classifier: str, n_estimators: int):
        if classifier == "xgb":
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=n_estimators, max_depth=8, learning_rate=0.1,
                use_label_encoder=False, eval_metric="logloss",
                random_state=self._rs, n_jobs=-1,
            )
        else:  # Default: rf
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=n_estimators, max_depth=None,
                random_state=self._rs, n_jobs=-1,
            )

    @staticmethod
    def _get_proba(clf, X):
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X)
            return proba[:, 1] if proba.shape[1] == 2 else None
        return None

    @staticmethod
    def _build_result(
        y_test, y_pred, y_prob,
        dataset_name: str, classifier: str,
        train_ms: float = 0, infer_ms: float = 0,
        n_samples: int = 0, load_ms: float = 0,
    ) -> IDSBenchmarkResult:
        from sklearn.metrics import (accuracy_score, average_precision_score,
                                     classification_report, confusion_matrix,
                                     f1_score, precision_score, recall_score,
                                     roc_auc_score)

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        quality: dict[str, float] = {
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "false_alarm_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            "detection_rate": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        }

        if y_prob is not None:
            try:
                quality["roc_auc"] = float(roc_auc_score(y_test, y_prob))
            except ValueError:
                quality["roc_auc"] = 0.0
            try:
                quality["pr_auc"] = float(average_precision_score(y_test, y_prob))
            except ValueError:
                quality["pr_auc"] = 0.0

        reliability: dict[str, float] = {
            "structured_validity_rate": 1.0,  # Classical ML always produces structured output
        }

        if y_prob is not None:
            from packages.bench.metrics import (brier_score,
                                                expected_calibration_error)
            actuals = [int(v) for v in y_test]
            probs = [float(p) for p in y_prob]
            reliability["brier"] = brier_score(probs, actuals)
            reliability["ece"] = expected_calibration_error(probs, actuals)

        efficiency: dict[str, float] = {
            "load_time_ms": load_ms,
            "train_time_ms": train_ms,
            "inference_time_ms": infer_ms,
            "samples_per_sec": float(n_samples / max(infer_ms / 1000, 1e-9)),
        }

        cr = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        return IDSBenchmarkResult(
            dataset=dataset_name,
            model_type=classifier,
            quality=quality,
            reliability=reliability,
            efficiency=efficiency,
            confusion_matrix=cm.tolist(),
            class_report=cr,
            success=True,
        )
