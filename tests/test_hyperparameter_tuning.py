# -*- coding: utf-8 -*-
import pytest
import numpy as np
from forecasting.hyperparameter_tuning import EnsembleHyperparameterOptimizer

class DummyConfig:
    hp_tune_n_trials = 2
    hp_tune_timeout_seconds = 10

class DummyCVSplitter:
    def split(self, X, y):
        # 2 folds
        n = len(X)
        mid = n // 2
        yield np.arange(0, mid), np.arange(mid, n)
        yield np.arange(mid, n), np.arange(0, mid)

@pytest.fixture
def dummy_data():
    X = np.random.randn(50, 5)
    y = X[:, 0] * 2 + np.random.randn(50) * 0.1
    year_labels = np.array([2010 + (i // 10) for i in range(50)])
    return X, y, year_labels

def test_hyperparameter_optimizer_initialization():
    config = DummyConfig()
    cv_splitter = DummyCVSplitter()
    optimizer = EnsembleHyperparameterOptimizer(config, cv_splitter)
    assert optimizer.config == config
    assert optimizer.cv_splitter == cv_splitter

def test_optimize_kernel_ridge(dummy_data):
    X, y, year_labels = dummy_data
    config = DummyConfig()
    cv_splitter = DummyCVSplitter()
    optimizer = EnsembleHyperparameterOptimizer(config, cv_splitter, random_state=42)
    
    # Run optimization
    best_params = optimizer.optimize_kernel_ridge(X, y.reshape(-1, 1), year_labels)
    
    assert isinstance(best_params, dict)
    assert 'alpha' in best_params
    assert 'kernel' in best_params
