import pytest
from typing import Optional
import pandas as pd
from src.domain.protocols import WineFeatures, ExperimentResult, Predictor

@pytest.fixture
def sample_wine_features():
    return WineFeatures(
        fixed_acidity=7.4,
        volatile_acidity=0.7,
        citric_acid=0.0,
        residual_sugar=1.9,
        chlorides=0.076,
        free_sulfur_dioxide=11.0,
        total_sulfur_dioxide=34.0,
        density=0.9978,
        pH=3.51,
        sulphates=0.56,
        alcohol=9.4
    )

@pytest.fixture
def sample_training_data():
    # テスト用の小さなデータセット
    data = {
        'fixed_acidity': [7.4, 7.8, 7.2],
        'volatile_acidity': [0.7, 0.88, 0.65],
        'citric_acid': [0, 0, 0.1],
        'residual_sugar': [1.9, 2.0, 1.8],
        'chlorides': [0.076, 0.080, 0.070],
        'free_sulfur_dioxide': [11, 13, 10],
        'total_sulfur_dioxide': [34, 36, 32],
        'density': [0.9978, 0.9980, 0.9975],
        'pH': [3.51, 3.49, 3.52],
        'sulphates': [0.56, 0.58, 0.54],
        'alcohol': [9.4, 9.2, 9.6]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_labels():
    return pd.Series([5, 6, 5])

# モック用のPredictor実装
class MockPredictor:
    def predict(self, features: WineFeatures) -> float:
        return 5.0

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

@pytest.fixture
def mock_predictor():
    return MockPredictor()

# モック用のExperiment実装
class MockExperiment:
    def run(
        self,
        experiment_name: str,
        params: dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> ExperimentResult:
        return ExperimentResult(
            model_name="mock_model",
            metrics={"accuracy": 0.8},
            parameters=params,
            model_version="v0.1"
        )

@pytest.fixture
def mock_experiment():
    return MockExperiment()

# モック用のModelManager実装
class MockModelManager:
    def __init__(self):
        self.models = {}

    def save_predictor(self, predictor: Predictor, version: str) -> None:
        self.models[version] = predictor

    def load_predictor(self, version: str) -> Optional[Predictor]:
        return self.models.get(version)

    def list_versions(self) -> list[str]:
        return list(self.models.keys())

@pytest.fixture
def mock_model_manager():
    return MockModelManager()
