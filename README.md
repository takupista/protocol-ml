# Protocol-ML

## 概要
このプロジェクトは、ワインの品質を予測する機械学習システムを実装したものです。Protocolベースの設計を採用し、拡張性と保守性を重視しています。

## 特徴
- Protocol（インターフェース）ベースの設計
- 複数の機械学習モデル（RandomForest, XGBoost）のサポート
- 実験管理システム
- モデルのバージョン管理
- 包括的なテストカバレッジ

## プロジェクト構造
```
wine_ml/
├── src/
│   ├── domain/
│   │   ├── __init__.py
│   │   └── protocols.py      # コアとなるProtocol定義
│   ├── models/
│   │   ├── __init__.py
│   │   └── predictors.py     # 予測モデルの実装
│   ├── experiments/
│   │   ├── __init__.py
│   │   └── experiment.py     # 実験管理システム
│   └── infrastructure/
│       ├── __init__.py
│       └── model_manager.py  # モデル管理システム
└── tests/
    ├── __init__.py
    ├── conftest.py               # テスト用の共通フィクスチャ
    ├── test_protocols.py         # 基本プロトコルのテスト
    ├── test_predictors.py        # Predictor実装のテスト
    ├── test_real_predictors.py   # 実際のモデル実装のテスト
    ├── test_experiments.py       # 実験システムのテスト
    └── test_model_manager.py     # モデル管理のテスト
```

## 主要コンポーネント

### Protocols (`src/domain/protocols.py`)
システムの核となるインターフェース定義：
- `Predictor`: モデルの訓練と予測を行うインターフェース
- `Experiment`: 実験の実行を管理するインターフェース
- `ModelManager`: モデルの保存と管理を行うインターフェース

### 予測モデル (`src/models/predictors.py`)
- `RandomForestWinePredictor`: RandomForestを使用した実装
- `XGBoostWinePredictor`: XGBoostを使用した実装
- 共通の特徴量バリデーションと変換処理

### 実験管理 (`src/experiments/experiment.py`)
- 異なるモデルとパラメータでの実験実行
- 実験結果の記録と保存
- 評価指標の計算（MSE, MAE, R2スコア）

### モデル管理 (`src/infrastructure/model_manager.py`)
- モデルの保存と読み込み
- バージョン管理システム
- メタデータの管理

## セットアップ方法

### 環境要件
- Python 3.8以上
- 必要なパッケージ:
  ```
  pandas
  numpy
  scikit-learn
  xgboost
  pytest（テスト用）
  ```

### インストール
```bash
# 仮想環境の作成（推奨）
python -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

## 使用方法

### 基本的な予測の実行
```python
from src.models.predictors import RandomForestWinePredictor
from src.domain.protocols import WineFeatures

# モデルのインスタンス化
predictor = RandomForestWinePredictor()

# データの準備（例）
X_train = ...  # 訓練データ
y_train = ...  # 訓練ラベル

# モデルの訓練
predictor.train(X_train, y_train)

# 予測
features = WineFeatures(
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
quality_score = predictor.predict(features)
```

### 実験の実行
```python
from src.experiments.experiment import WineQualityExperiment

# 実験の設定
experiment = WineQualityExperiment()

# 実験の実行
result = experiment.run(
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

# 結果の確認
print(result.metrics)
```

### モデルの保存と読み込み
```python
from src.infrastructure.model_manager import WineModelManager

# モデル管理の初期化
manager = WineModelManager()

# モデルの保存
manager.save_predictor(predictor, "v1.0")

# モデルの読み込み
loaded_predictor = manager.load_predictor("v1.0")

# 利用可能なバージョンの確認
versions = manager.list_versions()
```

## テストの構成と実行

### テストファイルの構成
- `test_protocols.py`: プロトコルの基本的な契約が守られているかをテスト
  - Predictor, Experiment, ModelManagerインターフェースの検証
  - モックオブジェクトを使用した基本的な動作確認

- `test_predictors.py`: Predictorの基本実装のテスト
  - 予測機能の基本的な動作
  - エラー処理
  - 入力値の検証

- `test_real_predictors.py`: 実際の機械学習モデルの実装テスト
  - RandomForestWinePredictorの動作確認
  - XGBoostWinePredictorの動作確認
  - モデルの訓練と予測の正確性
  - 特徴量の重要度や学習率の影響

- `test_experiments.py`: 実験管理システムのテスト
  - 実験の実行と結果の検証
  - メトリクスの計算
  - 実験ログの管理

- `test_model_manager.py`: モデル管理システムのテスト
  - モデルの保存と読み込み
  - バージョン管理
  - メタデータの管理

### テストの実行方法
```bash
# 全てのテストを実行
pytest

# 特定のテストファイルを実行
pytest tests/test_protocols.py
pytest tests/test_real_predictors.py

# 特定のテストクラスを実行
pytest tests/test_predictors.py::TestPredictor

# カバレッジレポートの生成
pytest --cov=src tests/
```

## プロジェクトの拡張
新しい予測モデルを追加する場合：

1. `src/models/predictors.py`に新しいPredictor実装を追加
2. 必要に応じて`src/domain/protocols.py`のインターフェースを拡張
3. `src/experiments/experiment.py`の`AVAILABLE_MODELS`に新しいモデルを追加
4. 新しいモデル用のテストを追加

## ライセンス
MIT License

## 貢献
- Issue報告や機能要望は大歓迎です
- プルリクエストを送る前に、テストが全て通ることを確認してください
- コードスタイルはプロジェクトの規約に従ってください

## 注意事項

### データ分割について
本プロジェクトの実装では、簡略化のためデータを学習用とテスト用の2分割としていますが、実際の機械学習プロジェクトでは以下の3分割が推奨されます：

1. **学習データ（Training Data）**
   - モデルの学習に使用
   - 全データの約60%を使用することが一般的

2. **検証データ（Validation Data）**
   - ハイパーパラメータの調整に使用
   - 全データの約20%を使用することが一般的
   - 学習済みモデルの性能を評価し、最適なハイパーパラメータを選択
   - モデル選択やアーキテクチャの決定にも使用

3. **テストデータ（Test Data）**
   - 最終的な性能評価のみに使用
   - 全データの約20%を使用することが一般的
   - ハイパーパラメータ調整後、一度だけ使用
   - モデルの真の汎化性能を評価

この3分割方式を採用することで：
- より信頼性の高い性能評価が可能
- オーバーフィッティングのリスクを適切に評価可能
- テストデータが「真の未知データ」として機能

### その他の注意点
- 本プロジェクトは教育・研究目的で作成されています
- 実運用での使用は、上記のデータ分割方式の採用を含め、十分なテストと検証を行ってください
- Cross Validationなど、より堅牢な評価手法の採用も検討してください
