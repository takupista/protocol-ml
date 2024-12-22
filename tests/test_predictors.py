import pytest
import pandas as pd
from src.domain.protocols import WineFeatures

class TestPredictor:
    """Predictorクラスの基本的な機能テスト"""
    
    def test_predict_returns_float(self, mock_predictor, sample_wine_features):
        """
        正常系: 予測値が浮動小数点数として返され、適切な範囲内にあることを確認
        """
        result = mock_predictor.predict(sample_wine_features)
        assert isinstance(result, float)
        assert 0 <= result <= 10  # ワインの品質スコアは0-10の範囲

    def test_predict_accepts_wine_features(self, mock_predictor, sample_training_data):
        """
        正常系: WineFeaturesオブジェクトが正しく処理されることを確認
        """
        features_dict = sample_training_data.iloc[0].to_dict()
        features = WineFeatures(**features_dict)
        result = mock_predictor.predict(features)
        assert isinstance(result, float)

    def test_train_with_valid_data(self, mock_predictor, sample_training_data, sample_labels):
        """
        正常系: 有効なデータでの訓練が成功することを確認
        """
        try:
            mock_predictor.train(sample_training_data, sample_labels)
            features = WineFeatures(**sample_training_data.iloc[0].to_dict())
            result = mock_predictor.predict(features)
            assert isinstance(result, float)
        except Exception as e:
            pytest.fail(f"Training with valid data should not raise an exception: {e}")

    def test_train_with_mismatched_data_raises_error(self, mock_predictor):
        """
        異常系: 訓練データとラベルのサイズが一致しない場合にValueErrorが発生することを確認
        このテストは例外が発生することを期待している = テストは成功する
        """
        X = pd.DataFrame({'feature': [1, 2, 3]})
        y = pd.Series([1, 2])  # 意図的にサイズを変える

        with pytest.raises(ValueError) as excinfo:
            mock_predictor.train(X, y)
        
        assert "length" in str(excinfo.value).lower() or "match" in str(excinfo.value).lower()

    def test_predict_with_out_of_range_values(self, mock_predictor, sample_wine_features):
        """
        異常系: 異常な値でのValueErrorの発生を確認
        このテストは例外が発生することを期待している = テストは成功する
        """
        invalid_features = sample_wine_features
        invalid_features.pH = 15.0  # pH は通常0-14の範囲

        with pytest.raises(ValueError) as excinfo:
            mock_predictor.predict(invalid_features)
            
        assert "range" in str(excinfo.value).lower() or "invalid" in str(excinfo.value).lower()

    def test_predict_before_training(self, mock_predictor, sample_wine_features):
        """
        異常系: 訓練前の予測でRuntimeErrorが発生することを確認
        このテストは例外が発生することを期待している = テストは成功する
        """
        mock_predictor.is_trained = False  # 明示的に未訓練状態にする
        with pytest.raises(RuntimeError) as excinfo:
            mock_predictor.predict(sample_wine_features)
        assert "trained" in str(excinfo.value).lower()
