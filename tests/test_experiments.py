import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

from src.experiments.experiment import WineQualityExperiment
from src.domain.protocols import ExperimentResult

@pytest.fixture
def experiment_data():
    """
    実験用のサンプルデータセットを生成
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test) の形式でデータを返す
    """
    np.random.seed(42)
    n_samples = 50  # テスト用に小さめのサンプル数

    # ワインの特徴量を現実的な範囲で生成
    data = {
        'fixed_acidity': np.random.uniform(5, 15, n_samples),
        'volatile_acidity': np.random.uniform(0.2, 1.2, n_samples),
        'citric_acid': np.random.uniform(0, 1, n_samples),
        'residual_sugar': np.random.uniform(1, 15, n_samples),
        'chlorides': np.random.uniform(0.01, 0.5, n_samples),
        'free_sulfur_dioxide': np.random.uniform(1, 70, n_samples),
        'total_sulfur_dioxide': np.random.uniform(10, 200, n_samples),
        'density': np.random.uniform(0.9, 1.0, n_samples),
        'pH': np.random.uniform(2.9, 3.9, n_samples),
        'sulphates': np.random.uniform(0.3, 1.5, n_samples),
        'alcohol': np.random.uniform(8, 14, n_samples)
    }
    X = pd.DataFrame(data)
    y = pd.Series(np.random.uniform(3, 8, n_samples))  # 品質スコア
    
    # 訓練データとテストデータに分割
    train_size = int(0.8 * n_samples)
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]
    
    return X_train, X_test, y_train, y_test

@pytest.fixture
def temp_experiment_dir():
    """
    一時的な実験ディレクトリを作成
    
    テスト終了後に自動的に削除される一時ディレクトリを提供
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

class TestWineQualityExperiment:
    """ワイン品質予測実験のテストスイート"""

    def test_random_forest_experiment(self, experiment_data, temp_experiment_dir):
        """
        正常系: RandomForestモデルでの実験が正常に実行できることを確認
        
        - 実験結果の型が正しいこと
        - 必要なメトリクスが含まれていること
        - 訓練時間が記録されていること
        """
        X_train, X_test, y_train, y_test = experiment_data
        experiment = WineQualityExperiment(base_log_dir=temp_experiment_dir)
        
        params = {
            "model_type": "random_forest",
            "n_estimators": 10,  # テスト用に小さい値
            "random_state": 42
        }
        
        result = experiment.run(
            experiment_name="test_rf",
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )
        
        assert isinstance(result, ExperimentResult)
        assert result.model_name == "random_forest"
        assert all(metric in result.metrics for metric in ["mse", "mae", "r2", "training_time"])

    def test_xgboost_experiment(self, experiment_data, temp_experiment_dir):
        """
        正常系: XGBoostモデルでの実験が正常に実行できることを確認
        
        - 実験結果の型が正しいこと
        - 必要なメトリクスが含まれていること
        - パラメータが正しく渡されていること
        """
        X_train, X_test, y_train, y_test = experiment_data
        experiment = WineQualityExperiment(base_log_dir=temp_experiment_dir)
        
        params = {
            "model_type": "xgboost",
            "n_estimators": 10,  # テスト用に小さい値
            "learning_rate": 0.1,
            "random_state": 42
        }
        
        result = experiment.run(
            experiment_name="test_xgb",
            params=params,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )
        
        assert isinstance(result, ExperimentResult)
        assert result.model_name == "xgboost"
        assert all(metric in result.metrics for metric in ["mse", "mae", "r2", "training_time"])

    def test_experiment_log_saving(self, experiment_data, temp_experiment_dir):
        """
        正常系: 実験結果のログが正しく保存されることを確認
        
        - ログファイルが正しいディレクトリに作成されること
        - ログの内容が実験結果と一致すること
        - JSONフォーマットが正しいこと
        """
        X_train, X_test, y_train, y_test = experiment_data
        experiment = WineQualityExperiment(base_log_dir=temp_experiment_dir)
        
        experiment_name = "test_logging"
        result = experiment.run(
            experiment_name=experiment_name,
            params={"model_type": "random_forest", "n_estimators": 10},
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )
        
        # ログファイルの存在確認
        log_dir = Path(temp_experiment_dir) / experiment_name
        log_files = list(log_dir.glob("*.json"))
        assert len(log_files) == 1
        
        # ログ内容の確認
        with open(log_files[0]) as f:
            log_data = json.load(f)
        
        assert log_data["model_name"] == result.model_name
        assert log_data["metrics"].keys() == result.metrics.keys()
        assert log_data["parameters"] == result.parameters

    def test_invalid_model_type(self, experiment_data, temp_experiment_dir):
        """
        異常系: 無効なモデルタイプでの実験実行時にValueErrorが発生することを確認
        """
        X_train, X_test, y_train, y_test = experiment_data
        experiment = WineQualityExperiment(base_log_dir=temp_experiment_dir)
        
        with pytest.raises(ValueError) as excinfo:
            experiment.run(
                experiment_name="test_invalid",
                params={"model_type": "invalid_model"},
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test
            )
        
        assert "Unknown model type" in str(excinfo.value)

    def test_invalid_parameters(self, experiment_data, temp_experiment_dir):
        """
        異常系: 無効なパラメータでの実験実行時にValueErrorが発生することを確認
        
        - negative n_estimators
        - negative learning_rate
        """
        X_train, X_test, y_train, y_test = experiment_data
        experiment = WineQualityExperiment(base_log_dir=temp_experiment_dir)
        
        # negative n_estimators
        with pytest.raises(ValueError):
            experiment.run(
                experiment_name="test_invalid_params",
                params={"model_type": "random_forest", "n_estimators": -10},
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test
            )

        # negative learning_rate
        with pytest.raises(ValueError):
            experiment.run(
                experiment_name="test_invalid_params",
                params={"model_type": "xgboost", "learning_rate": -0.1},
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test
            )

    def test_empty_experiment_name(self, experiment_data, temp_experiment_dir):
        """
        異常系: 空の実験名でのエラー処理を確認
        """
        X_train, X_test, y_train, y_test = experiment_data
        experiment = WineQualityExperiment(base_log_dir=temp_experiment_dir)
        
        invalid_names = ["", " ", None]
        for invalid_name in invalid_names:
            with pytest.raises(ValueError) as excinfo:
                experiment.run(
                    experiment_name=invalid_name,
                    params={"model_type": "random_forest"},
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test
                )
            assert "name" in str(excinfo.value).lower()
