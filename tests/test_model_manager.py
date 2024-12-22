import pytest
from pathlib import Path
import json
import pandas as pd
import numpy as np
import tempfile
from datetime import datetime

from src.infrastructure.model_manager import WineModelManager
from src.models.predictors import RandomForestWinePredictor, XGBoostWinePredictor
from src.domain.protocols import WineFeatures

@pytest.fixture
def temp_model_dir():
    """
    一時的なモデル保存ディレクトリを作成
    
    テスト終了後に自動的に削除される一時ディレクトリを提供
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def trained_rf_predictor():
    """
    訓練済みのRandomForestPredictorを作成
    
    簡単なデータセットで訓練した予測モデルを提供
    予測の一貫性を確保するため、乱数シードを固定
    """
    np.random.seed(42)
    n_samples = 10
    
    # 訓練データの生成
    X = pd.DataFrame({
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
    })
    y = pd.Series(np.random.uniform(3, 8, n_samples))

    # モデルの訓練
    predictor = RandomForestWinePredictor(n_estimators=10, random_state=42)
    predictor.train(X, y)
    return predictor

class TestWineModelManager:
    """WineModelManagerの機能テスト"""

    def test_save_and_load_predictor(self, temp_model_dir, trained_rf_predictor):
        """
        正常系: モデルの保存と読み込みが正常に動作することを確認
        
        検証項目:
        - モデルが正しく保存されること
        - 保存したモデルが正しく読み込めること
        - 読み込んだモデルが元のモデルと同じ予測を行うこと
        """
        manager = WineModelManager(model_dir=temp_model_dir)
        version = "v1.0"

        # モデルを保存
        manager.save_predictor(trained_rf_predictor, version)

        # モデルを読み込み
        loaded_predictor = manager.load_predictor(version)

        # モデルの型を確認
        assert isinstance(loaded_predictor, RandomForestWinePredictor)

        # 予測結果の一致を確認
        test_features = WineFeatures(
            fixed_acidity=7.0,
            volatile_acidity=0.5,
            citric_acid=0.3,
            residual_sugar=2.0,
            chlorides=0.08,
            free_sulfur_dioxide=20.0,
            total_sulfur_dioxide=100.0,
            density=0.997,
            pH=3.2,
            sulphates=0.6,
            alcohol=10.0
        )
        original_prediction = trained_rf_predictor.predict(test_features)
        loaded_prediction = loaded_predictor.predict(test_features)
        assert original_prediction == loaded_prediction

    def test_version_listing(self, temp_model_dir, trained_rf_predictor):
        """
        正常系: バージョン一覧の取得が正常に動作することを確認
        
        検証項目:
        - 複数のバージョンが正しく保存されること
        - バージョン一覧が正しく取得できること
        - バージョンが順序付けされていること
        """
        manager = WineModelManager(model_dir=temp_model_dir)
        versions = ["v1.0", "v1.1", "v2.0"]

        # 複数のバージョンを保存
        for version in versions:
            manager.save_predictor(trained_rf_predictor, version)

        # バージョン一覧を取得
        saved_versions = manager.list_versions()
        assert set(saved_versions) == set(versions)
        assert len(saved_versions) == len(versions)
        
        # バージョンが順序付けされていることを確認
        assert saved_versions == sorted(saved_versions)

    def test_version_info(self, temp_model_dir, trained_rf_predictor):
        """
        正常系: バージョン情報の取得が正常に動作することを確認
        
        検証項目:
        - バージョン情報が正しく保存されること
        - 必要なメタデータが含まれていること
        - タイムスタンプが適切な形式であること
        """
        manager = WineModelManager(model_dir=temp_model_dir)
        version = "v1.0"
        
        manager.save_predictor(trained_rf_predictor, version)
        info = manager.get_version_info(version)
        
        # 基本情報の確認
        assert info is not None
        assert info["model_class"] == "RandomForestWinePredictor"
        assert "created_at" in info
        assert "file_path" in info

        # タイムスタンプの形式を確認
        created_at = datetime.fromisoformat(info["created_at"])
        assert isinstance(created_at, datetime)

    def test_delete_version(self, temp_model_dir, trained_rf_predictor):
        """
        正常系: バージョンの削除が正常に動作することを確認
        
        検証項目:
        - モデルファイルが削除されること
        - メタデータから削除されること
        - 削除後にバージョンが取得できないこと
        """
        manager = WineModelManager(model_dir=temp_model_dir)
        version = "v1.0"
        
        # モデルを保存して削除
        manager.save_predictor(trained_rf_predictor, version)
        assert manager.delete_version(version)
        
        # 削除後の確認
        assert version not in manager.list_versions()
        assert manager.load_predictor(version) is None
        assert manager.get_version_info(version) is None
        
        # モデルファイルが実際に削除されていることを確認
        model_file = Path(temp_model_dir) / f"{version}.pkl"
        assert not model_file.exists()

    def test_metadata_persistence(self, temp_model_dir, trained_rf_predictor):
        """
        正常系: メタデータが永続化されることを確認
        
        検証項目:
        - メタデータファイルが作成されること
        - メタデータが正しいJSON形式で保存されること
        - 必要な情報がすべて含まれていること
        """
        manager = WineModelManager(model_dir=temp_model_dir)
        version = "v1.0"
        
        manager.save_predictor(trained_rf_predictor, version)
        
        # メタデータファイルの確認
        metadata_file = Path(temp_model_dir) / "model_metadata.json"
        assert metadata_file.exists()
        
        # メタデータの内容を確認
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        assert version in metadata
        assert metadata[version]["model_class"] == "RandomForestWinePredictor"
        assert "created_at" in metadata[version]
        assert "file_path" in metadata[version]

    def test_invalid_version_format(self, temp_model_dir, trained_rf_predictor):
        """
        異常系: 無効なバージョン形式でのエラーハンドリングを確認
        
        検証項目:
        - 無効なバージョン形式でValueErrorが発生すること
        - エラーメッセージが適切であること
        """
        manager = WineModelManager(model_dir=temp_model_dir)
        invalid_versions = ["", " ", "version1", "v1..0", "1.0"]
        
        for invalid_version in invalid_versions:
            with pytest.raises(ValueError) as excinfo:
                manager.save_predictor(trained_rf_predictor, invalid_version)
            assert "version" in str(excinfo.value).lower()

    def test_duplicate_version(self, temp_model_dir, trained_rf_predictor):
        """
        異常系: 重複バージョンの保存時のエラーハンドリングを確認
        
        検証項目:
        - 同じバージョンの再保存でValueErrorが発生すること
        - 既存のモデルが上書きされないこと
        """
        manager = WineModelManager(model_dir=temp_model_dir)
        version = "v1.0"
        
        # 最初の保存は成功
        manager.save_predictor(trained_rf_predictor, version)
        
        # 同じバージョンでの保存は失敗
        with pytest.raises(ValueError) as excinfo:
            manager.save_predictor(trained_rf_predictor, version)
        assert "exists" in str(excinfo.value).lower()

    def test_nonexistent_version_load(self, temp_model_dir):
        """
        正常系: 存在しないバージョンの読み込み時にNoneが返されることを確認
        
        検証項目:
        - 存在しないバージョンの読み込みでNoneが返されること
        - エラーが発生しないこと
        """
        manager = WineModelManager(model_dir=temp_model_dir)
        loaded_predictor = manager.load_predictor("v1.0")
        assert loaded_predictor is None
