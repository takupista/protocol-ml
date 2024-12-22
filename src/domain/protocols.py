from typing import Protocol, Dict, Any, Optional
import pandas as pd
from dataclasses import dataclass

# データ構造の定義
@dataclass
class WineFeatures:
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@dataclass
class ExperimentResult:
    model_name: str
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    model_version: str

# Core Protocols
class Predictor(Protocol):
    def predict(self, features: WineFeatures) -> float:
        """
        与えられた特徴量から品質スコアを予測する
        """
        ...

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        モデルを訓練する
        """
        ...

class Experiment(Protocol):
    def run(
        self, 
        experiment_name: str, 
        params: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> ExperimentResult:
        """
        実験を実行し、結果を返す
        """
        ...

class ModelManager(Protocol):
    def save_predictor(self, predictor: Predictor, version: str) -> None:
        """
        予測モデルを保存する
        """
        ...
    
    def load_predictor(self, version: str) -> Optional[Predictor]:
        """
        指定されたバージョンの予測モデルを読み込む
        """
        ...

    def list_versions(self) -> list[str]:
        """
        利用可能なモデルバージョンの一覧を返す
        """
        ...
    