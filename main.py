import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import logging
from pathlib import Path

from src.models.predictors import (
    RandomForestWinePredictor,
    XGBoostWinePredictor,
    RandomBaselinePredictor
)
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
    
    CSV形式のデータを読み込み、カラム名を適切な形式に変換する
    
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

def analyze_data_distribution():
    """
    データセットの分布を分析
    
    - 品質スコアの全体分布
    - 訓練データとテストデータでの分布の比較
    - 基本的な統計情報の表示
    """
    logger.info("\n=== Data Distribution Analysis ===")
    
    # データの読み込み
    data = load_wine_data()
    
    # 品質スコアの分布
    quality_distribution = data['quality'].value_counts().sort_index()
    logger.info("\nQuality Score Distribution:")
    for score, count in quality_distribution.items():
        logger.info(f"Score {score}: {count} samples ({count/len(data)*100:.1f}%)")
    
    # 基本統計量
    logger.info("\nQuality Score Statistics:")
    logger.info(f"Mean: {data['quality'].mean():.2f}")
    logger.info(f"Median: {data['quality'].median()}")
    logger.info(f"Std Dev: {data['quality'].std():.2f}")
    
    # 訓練/テストデータの分布の確認
    X_train, X_test, y_train, y_test = prepare_data(data)
    
    logger.info("\nTrain/Test Split Distribution:")
    train_dist = y_train.value_counts().sort_index()
    test_dist = y_test.value_counts().sort_index()
    
    logger.info("\nTraining Data Distribution:")
    for score, count in train_dist.items():
        logger.info(f"Score {score}: {count} samples ({count/len(y_train)*100:.1f}%)")
    
    logger.info("\nTest Data Distribution:")
    for score, count in test_dist.items():
        logger.info(f"Score {score}: {count} samples ({count/len(y_test)*100:.1f}%)")

def compare_models_with_cv():
    """
    クロスバリデーションでモデルを比較
    
    5分割交差検証を使用して、モデルの性能をより厳密に評価する
    MSEスコアの平均と標準偏差を計算
    """
    logger.info("\n=== Cross-Validation Comparison ===")
    
    # データの読み込み
    data = load_wine_data()
    X = data.drop('quality', axis=1)
    y = data['quality']
    
    # 交差検証の設定
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 結果を格納するリスト
    rf_mse_scores = []
    xgb_mse_scores = []
    random_mse_scores = []
    
    # 各分割でモデルを訓練・評価
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        # データの分割
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # RandomForest
        rf_predictor = RandomForestWinePredictor(n_estimators=100)
        rf_predictor.train(X_train, y_train)
        rf_val_predictions = np.array([
            rf_predictor.predict(WineFeatures(**features)) 
            for features in X_val.to_dict('records')
        ])
        rf_mse = np.mean((rf_val_predictions - y_val) ** 2)
        rf_mse_scores.append(rf_mse)
        
        # XGBoost
        xgb_predictor = XGBoostWinePredictor(
            n_estimators=200,
            learning_rate=0.01,
            max_depth=4,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8
        )
        xgb_predictor.train(X_train, y_train)
        xgb_val_predictions = np.array([
            xgb_predictor.predict(WineFeatures(**features))
            for features in X_val.to_dict('records')
        ])
        xgb_mse = np.mean((xgb_val_predictions - y_val) ** 2)
        xgb_mse_scores.append(xgb_mse)

        # Random Baseline
        random_predictor = RandomBaselinePredictor(random_state=42 + fold)  # 各foldで異なるシード
        random_predictor.train(X_train, y_train)
        random_val_predictions = np.array([
            random_predictor.predict(WineFeatures(**features))
            for features in X_val.to_dict('records')
        ])
        random_mse = np.mean((random_val_predictions - y_val) ** 2)
        random_mse_scores.append(random_mse)
        
        logger.info(f"\nFold {fold} Results:")
        logger.info(f"RandomForest MSE: {rf_mse:.4f}")
        logger.info(f"XGBoost MSE: {xgb_mse:.4f}")
        logger.info(f"Random Baseline MSE: {random_mse:.4f}")
    
    # 全体の結果をまとめる
    rf_mse_scores = np.array(rf_mse_scores)
    xgb_mse_scores = np.array(xgb_mse_scores)
    random_mse_scores = np.array(random_mse_scores)
    
    logger.info("\nOverall Results:")
    logger.info(f"RandomForest     - Average MSE: {rf_mse_scores.mean():.4f} (+/- {rf_mse_scores.std() * 2:.4f})")
    logger.info(f"XGBoost         - Average MSE: {xgb_mse_scores.mean():.4f} (+/- {xgb_mse_scores.std() * 2:.4f})")
    logger.info(f"Random Baseline - Average MSE: {random_mse_scores.mean():.4f} (+/- {random_mse_scores.std() * 2:.4f})")

def analyze_feature_importance():
    """
    特徴量の重要度を分析
    
    各モデルが各特徴量をどの程度重視しているかを確認
    特定の特徴量への過度の依存がないかチェック
    異なるモデル間での特徴量の重要度の違いを比較
    """
    logger.info("\n=== Feature Importance Analysis ===")
    
    # データの読み込みと準備
    data = load_wine_data()
    X = data.drop('quality', axis=1)
    y = data['quality']
    
    # RandomForestの特徴量重要度
    rf_predictor = RandomForestWinePredictor(n_estimators=100)
    rf_predictor.train(X, y)
    rf_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_predictor.model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\nRandomForest Feature Importance:")
    for _, row in rf_importance.iterrows():
        logger.info(f"{row['feature']}: {row['importance']:.4f}")
    
    # XGBoostの特徴量重要度
    xgb_predictor = XGBoostWinePredictor(
        n_estimators=200,
        learning_rate=0.01,
        max_depth=4,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8
    )
    xgb_predictor.train(X, y)
    xgb_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': xgb_predictor.model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\nXGBoost Feature Importance:")
    for _, row in xgb_importance.iterrows():
        logger.info(f"{row['feature']}: {row['importance']:.4f}")
    
    # 特徴量の相関関係
    correlations = data.corr()['quality'].sort_values(ascending=False)
    logger.info("\nFeature Correlations with Quality:")
    for feature, corr in correlations.items():
        if feature != 'quality':
            logger.info(f"{feature}: {corr:.4f}")
            
    # モデル間の特徴量重要度の比較
    importance_comparison = pd.DataFrame({
        'feature': X.columns,
        'rf_importance': rf_predictor.model.feature_importances_,
        'xgb_importance': xgb_predictor.model.feature_importances_
    }).sort_values('rf_importance', ascending=False)
    
    logger.info("\nFeature Importance Comparison (RF vs XGBoost):")
    for _, row in importance_comparison.iterrows():
        logger.info(f"{row['feature']}: RF={row['rf_importance']:.4f}, XGB={row['xgb_importance']:.4f}")

def run_basic_prediction():
    """
    基本的な予測の実行例
    
    単一のモデルでの予測を実行し、結果を表示
    """
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
    """
    異なるモデルでの実験比較
    
    RandomForestとXGBoostの性能を比較
    """
    logger.info("Starting experiment comparison")
    
    # データの準備
    data = load_wine_data()
    X_train, X_test, y_train, y_test = prepare_data(data)
    
    # 実験の設定
    experiment = WineQualityExperiment(base_log_dir="./logs")
    
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
    """
    モデル管理機能のデモンストレーション
    
    モデルの保存、読み込み、バージョン管理の機能を実演
    """
    logger.info("Starting model management example")
    
    # データの準備
    data = load_wine_data()
    X_train, X_test, y_train, y_test = prepare_data(data)
    
    # モデルの訓練
    predictor = RandomForestWinePredictor(n_estimators=100)
    predictor.train(X_train, y_train)
    
    # モデル管理の初期化
    manager = WineModelManager(model_dir="./saved_models")
    
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
    """メイン実行関数"""
    # 必要なディレクトリの作成
    Path("./logs").mkdir(exist_ok=True)
    Path("./saved_models").mkdir(exist_ok=True)
    
    try:
        # データ分析
        analyze_data_distribution()
        
        # クロスバリデーションによる比較
        compare_models_with_cv()
        
        # 特徴量の重要度分析
        analyze_feature_importance()
        
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
