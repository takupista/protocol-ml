import pytest
import pandas as pd
import numpy as np
from src.models.predictors import RandomForestWinePredictor, XGBoostWinePredictor
from src.domain.protocols import WineFeatures

@pytest.fixture
def real_training_data():
    """
    実際のトレーニングデータに近い形のテストデータを生成
    
    Returns:
        pd.DataFrame: ワインの特徴量を含むデータフレーム
    """
    np.random.seed(42)
    n_samples = 100
    
    # 実際のワインデータの特性を考慮した特徴量の生成
    data = {
        'fixed_acidity': np.random.uniform(5, 15, n_samples),      # pH調整に関連
        'volatile_acidity': np.random.uniform(0.2, 1.2, n_samples), # 酢酸関連
        'citric_acid': np.random.uniform(0, 1, n_samples),         # クエン酸
        'residual_sugar': np.random.uniform(1, 15, n_samples),     # 残糖
        'chlorides': np.random.uniform(0.01, 0.5, n_samples),      # 塩化物
        'free_sulfur_dioxide': np.random.uniform(1, 70, n_samples), # 遊離型SO2
        'total_sulfur_dioxide': np.random.uniform(10, 200, n_samples), # 総SO2
        'density': np.random.uniform(0.9, 1.0, n_samples),         # 密度
        'pH': np.random.uniform(2.9, 3.9, n_samples),              # pH値
        'sulphates': np.random.uniform(0.3, 1.5, n_samples),       # 硫酸塩
        'alcohol': np.random.uniform(8, 14, n_samples)             # アルコール度数
    }
    return pd.DataFrame(data)

@pytest.fixture
def real_target_data(real_training_data):
    """
    特徴量から現実的なターゲット変数を生成
    
    ワインの品質に影響を与える主要な特徴量に重みを付けて計算
    
    Args:
        real_training_data: 特徴量データ
    
    Returns:
        pd.Series: ワインの品質スコア（0-10の範囲）
    """
    np.random.seed(42)
    # ワインの品質に影響を与える主要な特徴量とその重み
    weights = {
        'alcohol': 0.4,              # アルコール度数は品質に強く影響
        'volatile_acidity': -0.3,    # 揮発性酸は負の影響
        'sulphates': 0.2,           # 硫酸塩は保存性に関連
        'pH': -0.1                  # pHも品質に影響
    }
    
    # 重み付けした特徴量の合計にノイズを加える
    target = sum(real_training_data[feat] * weight for feat, weight in weights.items())
    target = target + np.random.normal(0, 0.5, len(real_training_data))
    
    # 0-10の範囲に収める
    return pd.Series(np.clip(target, 0, 10))

class TestRandomForestWinePredictor:
    """RandomForestWinePredictorの実装テスト"""

    def test_training_and_prediction(self, real_training_data, real_target_data):
        """
        正常系: モデルが正しく訓練され、予測が行えることを確認
        
        - モデルの訓練が成功すること
        - 予測値が適切な範囲内であること
        - 予測値が浮動小数点数であること
        """
        predictor = RandomForestWinePredictor(n_estimators=10)  # テスト用に少ないツリー数
        predictor.train(real_training_data, real_target_data)
        
        features = WineFeatures(**real_training_data.iloc[0].to_dict())
        prediction = predictor.predict(features)
        
        assert isinstance(prediction, float)
        assert 0 <= prediction <= 10

    def test_prediction_consistency(self, real_training_data, real_target_data):
        """
        正常系: 同じ入力に対して一貫した予測を返すことを確認
        
        モデルの決定論的な性質を確認するテスト
        random_stateを固定することで、同じ入力に対して常に同じ出力を返すことを確認
        """
        predictor = RandomForestWinePredictor(random_state=42)
        predictor.train(real_training_data, real_target_data)
        
        features = WineFeatures(**real_training_data.iloc[0].to_dict())
        prediction1 = predictor.predict(features)
        prediction2 = predictor.predict(features)
        
        assert prediction1 == prediction2

    def test_feature_importance(self, real_training_data, real_target_data):
        """
        正常系: 特徴量の重要度が適切に計算されることを確認
        
        - 特徴量重要度の合計が1になることを確認
        - すべての特徴量の重要度が非負であることを確認
        """
        predictor = RandomForestWinePredictor()
        predictor.train(real_training_data, real_target_data)
        
        importances = predictor.model.feature_importances_
        assert len(importances) == len(real_training_data.columns)
        assert all(importance >= 0 for importance in importances)
        assert sum(importances) == pytest.approx(1.0)

class TestXGBoostWinePredictor:
    """XGBoostWinePredictorの実装テスト"""

    def test_training_and_prediction(self, real_training_data, real_target_data):
        """
        正常系: モデルが正しく訓練され、予測が行えることを確認
        
        - モデルの訓練が成功すること
        - 予測値が適切な範囲内であること
        - 予測値が浮動小数点数であること
        """
        predictor = XGBoostWinePredictor(n_estimators=10)
        predictor.train(real_training_data, real_target_data)
        
        features = WineFeatures(**real_training_data.iloc[0].to_dict())
        prediction = predictor.predict(features)
        
        assert isinstance(prediction, float)
        assert 0 <= prediction <= 10

    def test_prediction_consistency(self, real_training_data, real_target_data):
        """
        正常系: 同じ入力に対して一貫した予測を返すことを確認
        
        モデルの決定論的な性質を確認するテスト
        """
        predictor = XGBoostWinePredictor(random_state=42)
        predictor.train(real_training_data, real_target_data)
        
        features = WineFeatures(**real_training_data.iloc[0].to_dict())
        prediction1 = predictor.predict(features)
        prediction2 = predictor.predict(features)
        
        assert prediction1 == prediction2

    def test_learning_rate_effect(self, real_training_data, real_target_data):
        """
        正常系: 学習率が予測に影響を与えることを確認
        
        異なる学習率での予測値の違いを確認することで、
        ハイパーパラメータが実際にモデルの挙動に影響を与えることを確認
        """
        predictor_fast = XGBoostWinePredictor(learning_rate=0.3)
        predictor_slow = XGBoostWinePredictor(learning_rate=0.01)
        
        predictor_fast.train(real_training_data, real_target_data)
        predictor_slow.train(real_training_data, real_target_data)
        
        features = WineFeatures(**real_training_data.iloc[0].to_dict())
        prediction_fast = predictor_fast.predict(features)
        prediction_slow = predictor_slow.predict(features)
        
        # 学習率が異なれば、予測値も異なるはず
        assert prediction_fast != prediction_slow

@pytest.mark.parametrize("predictor_class", [
    RandomForestWinePredictor,
    XGBoostWinePredictor
])
class TestPredictorCommonBehavior:
    """予測モデルの共通動作テスト"""

    def test_untrained_model_raises_error(self, predictor_class, real_training_data):
        """
        異常系: 訓練前のモデルで予測を行うとエラーになることを確認
        
        すべての予測モデルで共通して確認すべき基本的な動作の確認
        """
        predictor = predictor_class()
        features = WineFeatures(**real_training_data.iloc[0].to_dict())
        
        with pytest.raises(RuntimeError) as excinfo:
            predictor.predict(features)
        assert "trained" in str(excinfo.value).lower()
