import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path

from src.models.predictors import RandomForestWinePredictor, XGBoostWinePredictor
from src.experiments.experiment import WineQualityExperiment
from src.infrastructure.model_manager import WineModelManager
from src.domain.protocols import WineFeatures

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_wine_data() -> pd.DataFrame:
    """
    Wine Quality Datasetを読み込む
    
    Returns:
        pd.DataFrame: ワインデータセット
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    try:
        # CSVを読み込み
        wine_data = pd.read_csv(url, delimiter=";")
        
        # カラム名の変換マッピング
        column_mapping = {
            'fixed acidity': 'fixed_acidity',
            'volatile acidity': 'volatile_acidity',
            'citric acid': 'citric_acid',
            'residual sugar': 'residual_sugar',
            'free sulfur dioxide': 'free_sulfur_dioxide',
            'total sulfur dioxide': 'total_sulfur_dioxide',
        }
        
        # カラム名を変換
        wine_data = wine_data.rename(columns=column_mapping)
        
        logger.info(f"Successfully loaded wine data: {len(wine_data)} samples")
        return wine_data
    except Exception as e:
        logger.error(f"Error loading wine data: {e}")
        raise

def prepare_data(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    データを訓練用とテスト用に分割
    
    Args:
        data: 元のデータセット
        test_size: テストデータの割合
        random_state: 乱数シード
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = data.drop('quality', axis=1)
    y = data['quality']
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def run_basic_prediction():
    """基本的な予測の実行例"""
    logger.info("Starting basic prediction example")
    
    # データの読み込みと準備
    data = load_wine_data()
    X_train, X_test, y_train, y_test = prepare_data(data)
    
    # モデルのインスタンス化と訓練
    predictor = RandomForestWinePredictor(n_estimators=100)
    predictor.train(X_train, y_train)
    
    # サンプルデータでの予測
    sample_features = WineFeatures(
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
    
    prediction = predictor.predict(sample_features)
    logger.info(f"Predicted wine quality: {prediction:.2f}")

def run_experiment_comparison():
    """異なるモデルでの実験比較例"""
    logger.info("Starting experiment comparison")
    
    # データの準備
    data = load_wine_data()
    X_train, X_test, y_train, y_test = prepare_data(data)
    
    # 実験の設定
    experiment = WineQualityExperiment(base_log_dir="./experiments")
    
    # RandomForestでの実験
    rf_result = experiment.run(
        experiment_name="rf_experiment",
        params={
            "model_type": "random_forest",
            "n_estimators": 100
        },
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )
    
    # XGBoostでの実験（パラメータ調整版）
    xgb_result = experiment.run(
        experiment_name="xgb_experiment",
        params={
            "model_type": "xgboost",
            "n_estimators": 200,        # 増やす
            "learning_rate": 0.01,      # 小さくする
            "max_depth": 4,             # 木の深さを制限
            "min_child_weight": 2,      # 過学習防止
            "subsample": 0.8,           # サンプリング率
            "colsample_bytree": 0.8     # 特徴量のサンプリング率
        },
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )
    
    # 結果の比較
    logger.info("\nExperiment Results Comparison:")
    logger.info(f"RandomForest - MSE: {rf_result.metrics['mse']:.4f}")
    logger.info(f"XGBoost     - MSE: {xgb_result.metrics['mse']:.4f}")

def run_model_management():
    """モデル管理の例"""
    logger.info("Starting model management example")
    
    # データの準備
    data = load_wine_data()
    X_train, X_test, y_train, y_test = prepare_data(data)
    
    # モデルマネージャーの初期化
    manager = WineModelManager(model_dir="./saved_models")
    
    # モデルの訓練と保存
    predictor = RandomForestWinePredictor(n_estimators=100)
    predictor.train(X_train, y_train)
    
    # モデルの保存
    version = "v1.0"
    manager.save_predictor(predictor, version)
    logger.info(f"Saved model version: {version}")
    
    # 利用可能なバージョンの確認
    versions = manager.list_versions()
    logger.info(f"Available model versions: {versions}")
    
    # モデルの読み込みと予測
    loaded_predictor = manager.load_predictor(version)
    if loaded_predictor:
        sample_features = WineFeatures(
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
        prediction = loaded_predictor.predict(sample_features)
        logger.info(f"Prediction using loaded model: {prediction:.2f}")

def main():
    """メイン処理"""
    # 必要なディレクトリの作成
    Path("./logs").mkdir(exist_ok=True)
    Path("./saved_models").mkdir(exist_ok=True)
    
    try:
        # 基本的な予測の実行
        logger.info("\n=== Basic Prediction Example ===")
        run_basic_prediction()
        
        # 実験の比較
        logger.info("\n=== Experiment Comparison Example ===")
        run_experiment_comparison()
        
        # モデル管理
        logger.info("\n=== Model Management Example ===")
        run_model_management()
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main()
