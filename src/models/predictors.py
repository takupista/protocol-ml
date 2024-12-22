from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from dataclasses import asdict

from src.domain.protocols import WineFeatures, Predictor

class BaseWinePredictor:
    """ワイン品質予測の基底クラス"""
    
    def _validate_features(self, features: WineFeatures) -> None:
        """特徴量の値を検証する"""
        feature_dict = asdict(features)
        
        # 必須フィールドの確認
        required_fields = {
            'fixed_acidity', 'volatile_acidity', 'citric_acid',
            'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
            'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol'
        }
        missing_fields = required_fields - set(feature_dict.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        # 値の範囲チェック
        validations = {
            'fixed_acidity': (0, 20),        # 一般的な範囲
            'volatile_acidity': (0, 2),      # 一般的な範囲
            'citric_acid': (0, 1),           # 一般的な範囲
            'residual_sugar': (0, 50),       # 一般的な範囲
            'chlorides': (0, 1),             # 一般的な範囲
            'free_sulfur_dioxide': (0, 100),  # 一般的な範囲
            'total_sulfur_dioxide': (0, 300), # 一般的な範囲
            'density': (0.5, 1.5),           # ワインの一般的な密度範囲
            'pH': (2.5, 4.5),                # ワインの一般的なpH範囲
            'sulphates': (0, 2),             # 一般的な範囲
            'alcohol': (8, 16)               # ワインの一般的なアルコール度数範囲
        }

        for field, (min_val, max_val) in validations.items():
            value = feature_dict[field]
            if not min_val <= value <= max_val:
                raise ValueError(
                    f"{field} value {value} is outside valid range [{min_val}, {max_val}]"
                )

    def _features_to_array(self, features: WineFeatures) -> np.ndarray:
        """WineFeaturesをモデル入力用の配列に変換する"""
        feature_dict = asdict(features)
        feature_array = np.array([
            feature_dict['fixed_acidity'],
            feature_dict['volatile_acidity'],
            feature_dict['citric_acid'],
            feature_dict['residual_sugar'],
            feature_dict['chlorides'],
            feature_dict['free_sulfur_dioxide'],
            feature_dict['total_sulfur_dioxide'],
            feature_dict['density'],
            feature_dict['pH'],
            feature_dict['sulphates'],
            feature_dict['alcohol']
        ]).reshape(1, -1)
        return feature_array


class RandomForestWinePredictor(BaseWinePredictor):
    """RandomForestを使用したワイン品質予測モデル"""
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """モデルを訓練する"""
        if len(X) != len(y):
            raise ValueError("Length of X and y must match")
        if len(X) == 0:
            raise ValueError("Cannot train model with empty dataset")
        
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, features: WineFeatures) -> float:
        """ワインの品質を予測する"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        self._validate_features(features)
        X = self._features_to_array(features)
        prediction = self.model.predict(X)[0]
        
        # 予測値を0-10の範囲に収める
        return float(np.clip(prediction, 0, 10))


class XGBoostWinePredictor(BaseWinePredictor):
    """XGBoostを使用したワイン品質予測モデル"""
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        random_state: int = 42
    ):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """モデルを訓練する"""
        if len(X) != len(y):
            raise ValueError("Length of X and y must match")
        if len(X) == 0:
            raise ValueError("Cannot train model with empty dataset")
        
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, features: WineFeatures) -> float:
        """ワインの品質を予測する"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        self._validate_features(features)
        X = self._features_to_array(features)
        prediction = self.model.predict(X)[0]
        
        # 予測値を0-10の範囲に収める
        return float(np.clip(prediction, 0, 10))
