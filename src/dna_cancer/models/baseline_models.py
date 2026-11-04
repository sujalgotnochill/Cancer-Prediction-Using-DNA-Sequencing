"""Baseline machine learning models for DNA sequence classification benchmark.

Provides:
- Multinomial Naive Bayes (multinomial_nb)
- Logistic Regression (logistic_regression)
- Random Forest (random_forest)

All implementations adhere strictly to the BaseDNAClassifier interface and are
registered in the ModelFactory registry.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import joblib
except ImportError:
    joblib = None  # type: ignore

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    from sklearn.ensemble import (
        GradientBoostingClassifier as SklearnGradientBoosting,
        RandomForestClassifier as SklearnRandomForest,
    )
    from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
    from sklearn.naive_bayes import MultinomialNB as SklearnMultinomialNB
    from sklearn.svm import LinearSVC as SklearnLinearSVC
except ImportError:
    SklearnMultinomialNB = None  # type: ignore
    SklearnLogisticRegression = None  # type: ignore
    SklearnRandomForest = None  # type: ignore
    SklearnLinearSVC = None  # type: ignore
    SklearnGradientBoosting = None  # type: ignore

from dna_cancer.models.base_model import BaseDNAClassifier
from dna_cancer.models.registry import register_model
from dna_cancer.utils.exceptions import ModelError, TrainingError
from dna_cancer.utils.logger import get_logger

logger = get_logger(__name__)


def _ensure_sklearn() -> None:
    if SklearnMultinomialNB is None or joblib is None:
        raise TrainingError(
            "scikit-learn and joblib are required for baseline classifiers. "
            "Please ensure scikit-learn is installed in your Python environment."
        )


@register_model("multinomial_nb")
class MultinomialNBClassifier(BaseDNAClassifier):
    """Multinomial Naive Bayes classifier for discrete k-mer count features."""

    def __init__(
        self,
        alpha: float = 1.0,
        fit_prior: bool = True,
        class_prior: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        self.alpha = float(alpha)
        self.fit_prior = bool(fit_prior)
        self.class_prior = class_prior
        self.extra_kwargs = kwargs
        self.model = None

    def _init_estimator(self) -> None:
        _ensure_sklearn()
        if self.model is None:
            self.model = SklearnMultinomialNB(
                alpha=self.alpha,
                fit_prior=self.fit_prior,
                class_prior=self.class_prior,
                **self.extra_kwargs,
            )

    def fit(
        self,
        X: Any,
        y: Any,
        validation_data: Optional[tuple] = None,
        **kwargs: Any,
    ) -> "MultinomialNBClassifier":
        self._init_estimator()
        shape_desc = X.shape if hasattr(X, "shape") else len(X)
        logger.info("Fitting MultinomialNB (alpha=%.3f) on X with shape %s", self.alpha, shape_desc)
        self.model.fit(X, y)
        return self

    def predict(self, X: Any) -> Any:
        self._init_estimator()
        return self.model.predict(X)

    def predict_proba(self, X: Any) -> Any:
        self._init_estimator()
        return self.model.predict_proba(X)

    def get_params(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "fit_prior": self.fit_prior,
            "class_prior": self.class_prior,
            **self.extra_kwargs,
        }

    def save(self, path: Union[str, Path]) -> None:
        _ensure_sklearn()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)
        logger.info("Saved MultinomialNB model to %s", p)

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs: Any) -> "MultinomialNBClassifier":
        _ensure_sklearn()
        p = Path(path)
        loaded = joblib.load(p)
        if not isinstance(loaded, cls):
            raise TrainingError(f"Expected instance of {cls.__name__}, got {type(loaded)}")
        return loaded


@register_model("logistic_regression")
class LogisticRegressionClassifier(BaseDNAClassifier):
    """L2-regularized Logistic Regression classifier for k-mer features."""

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
        solver: str = "lbfgs",
        multi_class: str = "auto",
        **kwargs: Any,
    ) -> None:
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.solver = solver
        self.multi_class = multi_class
        self.extra_kwargs = kwargs
        self.model = None

    def _init_estimator(self) -> None:
        _ensure_sklearn()
        if self.model is None:
            self.model = SklearnLogisticRegression(
                C=self.C,
                max_iter=self.max_iter,
                random_state=self.random_state,
                solver=self.solver,
                multi_class=self.multi_class,
                **self.extra_kwargs,
            )

    def fit(
        self,
        X: Any,
        y: Any,
        validation_data: Optional[tuple] = None,
        **kwargs: Any,
    ) -> "LogisticRegressionClassifier":
        self._init_estimator()
        shape_desc = X.shape if hasattr(X, "shape") else len(X)
        logger.info(
            "Fitting LogisticRegression (C=%.3f, max_iter=%d, solver=%s) on X with shape %s",
            self.C,
            self.max_iter,
            self.solver,
            shape_desc,
        )
        self.model.fit(X, y)
        return self

    def predict(self, X: Any) -> Any:
        self._init_estimator()
        return self.model.predict(X)

    def predict_proba(self, X: Any) -> Any:
        self._init_estimator()
        return self.model.predict_proba(X)

    def get_params(self) -> Dict[str, Any]:
        return {
            "C": self.C,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "solver": self.solver,
            "multi_class": self.multi_class,
            **self.extra_kwargs,
        }

    def save(self, path: Union[str, Path]) -> None:
        _ensure_sklearn()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)
        logger.info("Saved LogisticRegression model to %s", p)

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs: Any) -> "LogisticRegressionClassifier":
        _ensure_sklearn()
        p = Path(path)
        loaded = joblib.load(p)
        if not isinstance(loaded, cls):
            raise TrainingError(f"Expected instance of {cls.__name__}, got {type(loaded)}")
        return loaded


@register_model("random_forest")
class RandomForestClassifierWrapper(BaseDNAClassifier):
    """Ensemble Random Forest classifier for DNA k-mer frequency features."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs: Any,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.extra_kwargs = kwargs
        self.model = None

    def _init_estimator(self) -> None:
        _ensure_sklearn()
        if self.model is None:
            self.model = SklearnRandomForest(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                **self.extra_kwargs,
            )

    def fit(
        self,
        X: Any,
        y: Any,
        validation_data: Optional[tuple] = None,
        **kwargs: Any,
    ) -> "RandomForestClassifierWrapper":
        self._init_estimator()
        shape_desc = X.shape if hasattr(X, "shape") else len(X)
        logger.info(
            "Fitting RandomForest (n_estimators=%d, max_depth=%s) on X with shape %s",
            self.n_estimators,
            self.max_depth,
            shape_desc,
        )
        self.model.fit(X, y)
        return self

    def predict(self, X: Any) -> Any:
        self._init_estimator()
        return self.model.predict(X)

    def predict_proba(self, X: Any) -> Any:
        self._init_estimator()
        return self.model.predict_proba(X)

    def get_params(self) -> Dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            **self.extra_kwargs,
        }

    def save(self, path: Union[str, Path]) -> None:
        _ensure_sklearn()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)
        logger.info("Saved RandomForest model to %s", p)

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs: Any) -> "RandomForestClassifierWrapper":
        _ensure_sklearn()
        p = Path(path)
        loaded = joblib.load(p)
        if not isinstance(loaded, cls):
            raise TrainingError(f"Expected instance of {cls.__name__}, got {type(loaded)}")
        return loaded


@register_model("linear_svc")
@register_model("linear_svm")
class LinearSVCClassifier(BaseDNAClassifier):
    """Linear Support Vector Classifier (LinearSVC) for high-dimensional k-mer features."""

    def __init__(
        self,
        C: float = 1.0,
        class_weight: Optional[Union[str, Dict[Any, float]]] = None,
        max_iter: int = 2000,
        random_state: int = 42,
        dual: Union[str, bool] = "auto",
        **kwargs: Any,
    ) -> None:
        self.C = float(C)
        self.class_weight = class_weight
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.dual = dual
        self.extra_kwargs = kwargs
        self.model = None

    def _init_estimator(self) -> None:
        _ensure_sklearn()
        if self.model is None:
            self.model = SklearnLinearSVC(
                C=self.C,
                class_weight=self.class_weight,
                max_iter=self.max_iter,
                random_state=self.random_state,
                dual=self.dual,
                **self.extra_kwargs,
            )

    def fit(
        self,
        X: Any,
        y: Any,
        validation_data: Optional[tuple] = None,
        **kwargs: Any,
    ) -> "LinearSVCClassifier":
        self._init_estimator()
        self.model.fit(X, y)
        return self

    def predict(self, X: Any) -> Any:
        self._init_estimator()
        return self.model.predict(X)

    def predict_proba(self, X: Any) -> Any:
        self._init_estimator()
        if np is None:
            raise TrainingError("numpy is required for LinearSVC probability calibration.")
        df = self.model.decision_function(X)
        if df.ndim == 1:
            prob_pos = 1.0 / (1.0 + np.exp(-df))
            return np.vstack([1.0 - prob_pos, prob_pos]).T
        exp = np.exp(df - np.max(df, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def get_params(self) -> Dict[str, Any]:
        return {
            "C": self.C,
            "class_weight": self.class_weight,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "dual": self.dual,
            **self.extra_kwargs,
        }

    def save(self, path: Union[str, Path]) -> None:
        _ensure_sklearn()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)
        logger.info("Saved LinearSVC model to %s", p)

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs: Any) -> "LinearSVCClassifier":
        _ensure_sklearn()
        p = Path(path)
        loaded = joblib.load(p)
        if not isinstance(loaded, cls):
            raise TrainingError(f"Expected instance of {cls.__name__}, got {type(loaded)}")
        return loaded


@register_model("gradient_boosting")
class GradientBoostingClassifierWrapper(BaseDNAClassifier):
    """Gradient Boosting classifier for sequence feature representations."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.random_state = int(random_state)
        self.extra_kwargs = kwargs
        self.model = None

    def _init_estimator(self) -> None:
        _ensure_sklearn()
        if self.model is None:
            self.model = SklearnGradientBoosting(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state,
                **self.extra_kwargs,
            )

    def fit(
        self,
        X: Any,
        y: Any,
        validation_data: Optional[tuple] = None,
        **kwargs: Any,
    ) -> "GradientBoostingClassifierWrapper":
        self._init_estimator()
        self.model.fit(X, y)
        return self

    def predict(self, X: Any) -> Any:
        self._init_estimator()
        return self.model.predict(X)

    def predict_proba(self, X: Any) -> Any:
        self._init_estimator()
        return self.model.predict_proba(X)

    def get_params(self) -> Dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
            **self.extra_kwargs,
        }

    def save(self, path: Union[str, Path]) -> None:
        _ensure_sklearn()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)
        logger.info("Saved GradientBoosting model to %s", p)

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs: Any) -> "GradientBoostingClassifierWrapper":
        _ensure_sklearn()
        p = Path(path)
        loaded = joblib.load(p)
        if not isinstance(loaded, cls):
            raise TrainingError(f"Expected instance of {cls.__name__}, got {type(loaded)}")
        return loaded
