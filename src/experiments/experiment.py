from typing import Dict, Any, Type
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import json
from datetime import datetime

from src.domain.protocols import Experiment, ExperimentResult, Predictor, WineFeatures
from src.models.predictors import RandomForestWinePredictor, XGBoostWinePredictor

class WineQualityExperiment(Experiment):
    """ワイン品質予測の実験を管理するクラス"""

    # 利用可能なモデルの定義
    AVAILABLE_MODELS: Dict[str, Type[Predictor]] = {
        "random_forest": RandomForestWinePredictor,
        "xgboost": XGBoostWinePredictor
    }

    def __init__(self, base_log_dir: str = "./experiments"):
        """
        Args:
            base_log_dir: 実験結果を保存するベースディレクトリ
        """
        self.base_log_dir = base_log_dir

    def _validate_experiment_name(self, name: str) -> None:
        """実験名を検証"""
        if not name or not isinstance(name, str):
            raise ValueError("Experiment name must be a non-empty string")
        if len(name.strip()) == 0:
            raise ValueError("Experiment name cannot be empty or whitespace")

    def _validate_parameters(self, model_type: str, params: Dict[str, Any]) -> None:
        """実験パラメータを検証"""
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model type: {model_type}. Available models: {list(self.AVAILABLE_MODELS.keys())}")

        # モデル固有のパラメータ検証
        if model_type == "random_forest":
            if "n_estimators" in params and (not isinstance(params["n_estimators"], int) or params["n_estimators"] <= 0):
                raise ValueError("n_estimators must be a positive integer")

        elif model_type == "xgboost":
            if "learning_rate" in params and (not isinstance(params["learning_rate"], (int, float)) or params["learning_rate"] <= 0):
                raise ValueError("learning_rate must be a positive number")
            if "max_depth" in params and (not isinstance(params["max_depth"], int) or params["max_depth"] <= 0):
                raise ValueError("max_depth must be a positive integer")

    def _calculate_metrics(
        self,
        predictor: Predictor,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """評価指標を計算"""
        predictions = []
        for _, row in X_test.iterrows():
            features_dict = row.to_dict()
            # ここで features_dict を WineFeatures に変換する必要があります
            features = WineFeatures(**features_dict)  # 修正
            prediction = predictor.predict(features)
            predictions.append(prediction)
        
        metrics = {
            "mse": mean_squared_error(y_test, predictions),
            "mae": mean_absolute_error(y_test, predictions),
            "r2": r2_score(y_test, predictions)
        }

        return metrics

    def _generate_model_version(self, experiment_name: str) -> str:
        """モデルのバージョン文字列を生成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{experiment_name}_{timestamp}"

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
        実験を実行する

        Args:
            experiment_name: 実験名
            params: 実験パラメータ
                必須: "model_type" - 使用するモデルの種類
                オプション: モデル固有のパラメータ
            X_train: 訓練データの特徴量
            y_train: 訓練データのターゲット
            X_test: テストデータの特徴量
            y_test: テストデータのターゲット

        Returns:
            ExperimentResult: 実験結果
        """
        # バリデーション
        self._validate_experiment_name(experiment_name)
        model_type = params.get("model_type")
        if not model_type:
            raise ValueError("model_type must be specified in params")
        self._validate_parameters(model_type, params)

        # データの検証
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Training data cannot be empty")
        if len(X_test) == 0 or len(y_test) == 0:
            raise ValueError("Test data cannot be empty")

        # モデルの初期化
        model_class = self.AVAILABLE_MODELS[model_type]
        model_params = {k: v for k, v in params.items() if k != "model_type"}
        predictor = model_class(**model_params)

        # 実験の実行
        start_time = time.time()
        predictor.train(X_train, y_train)
        training_time = time.time() - start_time

        # 評価指標の計算
        metrics = self._calculate_metrics(predictor, X_test, y_test)
        metrics["training_time"] = training_time

        # バージョンの生成
        model_version = self._generate_model_version(experiment_name)

        # 実験結果の作成
        result = ExperimentResult(
            model_name=model_type,
            metrics=metrics,
            parameters=params,
            model_version=model_version
        )

        # 結果のログ保存（オプション）
        self._save_experiment_log(experiment_name, result)

        return result

    def _save_experiment_log(self, experiment_name: str, result: ExperimentResult) -> None:
        """実験結果をJSONファイルとして保存"""
        from pathlib import Path
        import os

        log_dir = Path(self.base_log_dir) / experiment_name
        os.makedirs(log_dir, exist_ok=True)

        log_file = log_dir / f"{result.model_version}.json"
        log_data = {
            "model_name": result.model_name,
            "metrics": result.metrics,
            "parameters": result.parameters,
            "model_version": result.model_version,
            "timestamp": datetime.now().isoformat()
        }

        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
