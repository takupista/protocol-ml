# Protocol-ML

## 概要
このプロジェクトは、ワインの品質を予測する機械学習システムを実装したものです。Protocol（インターフェース）ベースの設計を採用し、拡張性と保守性を重視した実装となっています。UCI Machine Learning RepositoryのWine Quality Datasetを使用し、ワインの化学的特性から品質スコアを予測します。

## 特徴
- Protocol（インターフェース）ベースの設計による高い拡張性
- 複数の機械学習モデル（RandomForest, XGBoost）のサポート
- 実験管理システムによる実験結果の追跡
- モデルのバージョン管理機能
- 包括的なテストカバレッジ
- 特徴量のバリデーションと型安全性

## プロジェクト構造
```
protocol-ml/
├── main.py              # メインの実行スクリプト
├── requirements.txt     # 依存パッケージリスト
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
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # テスト用の共通フィクスチャ
│   ├── test_protocols.py     # プロトコルの基本的なテスト
│   ├── test_predictors.py    # Predictor実装の基本テスト
│   ├── test_real_predictors.py # 実際のモデル実装のテスト
│   ├── test_experiments.py   # 実験システムのテスト
│   └── test_model_manager.py # モデル管理のテスト
├── experiments/          # 実験結果の保存ディレクトリ
└── saved_models/        # 保存されたモデルのディレクトリ
```

## 主要コンポーネント

### Protocols (`src/domain/protocols.py`)
システムの核となるインターフェース定義を提供します：

```python
class Predictor(Protocol):
    def predict(self, features: WineFeatures) -> float:
        """与えられた特徴量から品質スコアを予測する"""
        ...

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """モデルを訓練する"""
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
        """実験を実行し、結果を返す"""
        ...

class ModelManager(Protocol):
    def save_predictor(self, predictor: Predictor, version: str) -> None:
        """予測モデルを保存する"""
        ...
    
    def load_predictor(self, version: str) -> Optional[Predictor]:
        """指定されたバージョンの予測モデルを読み込む"""
        ...
```

### 予測モデル (`src/models/predictors.py`)
具体的な予測モデルの実装を提供します：

- `RandomForestWinePredictor`: RandomForestを使用した実装
  - scikit-learnのRandomForestRegressorをベースに実装
  - ハイパーパラメータのカスタマイズが可能
  - 特徴量の重要度分析が可能

- `XGBoostWinePredictor`: XGBoostを使用した実装
  - より高度な予測性能を提供
  - 学習率や木の深さなどのパラメータ調整が可能

両モデルとも以下の機能を共有：
- 特徴量の範囲検証
- 訓練状態の管理
- 予測値の範囲制限（0-10）

### 実験管理 (`src/experiments/experiment.py`)
実験の実行と結果の管理を行います：

- 異なるモデルとパラメータでの実験実行
- 以下のメトリクスの計算：
  - MSE (Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R² スコア
  - 訓練時間
- 実験結果のJSON形式での保存
- 実験の再現性の確保

### モデル管理 (`src/infrastructure/model_manager.py`)
モデルのバージョン管理を提供：

- モデルの保存と読み込み
- バージョン管理（セマンティックバージョニング）
- メタデータの管理（モデル種類、作成日時など）
- 保存されたモデルの一覧取得

## セットアップ方法

### 環境要件
- Python 3.10以上

### インストール手順

1. リポジトリのクローン：
```bash
git clone https://github.com/takupista/protocol-ml.git
cd protocol-ml
```

2. 仮想環境の作成（推奨）：
```bash
# Unix/macOS
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

3. 依存パッケージのインストール：
```bash
pip install -r requirements.txt
```

4. 必要なディレクトリの作成：
```bash
mkdir -p logs saved_models
```

## 使用方法

### 基本的な使用例

1. データの準備と予測：
```python
from src.models.predictors import RandomForestWinePredictor
from src.domain.protocols import WineFeatures
import pandas as pd

# データの読み込み
data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", delimiter=";")
X = data.drop('quality', axis=1)
y = data['quality']

# モデルの訓練
predictor = RandomForestWinePredictor(n_estimators=100)
predictor.train(X, y)

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
print(f"Predicted quality: {quality_score}")
```

2. 実験の実行：
```python
from src.experiments.experiment import WineQualityExperiment
from sklearn.model_selection import train_test_split

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 実験の実行
experiment = WineQualityExperiment()
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

print("実験結果:")
print(f"MSE: {result.metrics['mse']:.4f}")
print(f"MAE: {result.metrics['mae']:.4f}")
print(f"R²: {result.metrics['r2']:.4f}")
```

3. モデルの保存と読み込み：
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
print(f"Available versions: {versions}")
```

### メインプログラムの実行

プロジェクトに含まれる`main.py`を使用して、一連の処理を実行できます：

```bash
python main.py
```

このスクリプトは以下の処理を行います：
- 基本的な予測の実行
- 異なるモデルでの実験比較
- モデル管理機能のデモンストレーション

## テストの実行

### すべてのテストの実行
```bash
pytest
```

### 特定のテストの実行
```bash
# 特定のテストファイル
pytest tests/test_predictors.py

# 特定のテストクラス
pytest tests/test_predictors.py::TestPredictor

# 特定のテストメソッド
pytest tests/test_predictors.py::TestPredictor::test_predict_returns_float
```

### カバレッジレポートの生成
```bash
pytest --cov=src tests/
```

## プロジェクトの拡張方法

### 新しい予測モデルの追加

1. `src/models/predictors.py`に新しいPredictor実装を追加：
```python
class NewModelPredictor(BaseWinePredictor):
    def __init__(self, **params):
        self.model = YourModel(**params)
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        # 実装

    def predict(self, features: WineFeatures) -> float:
        # 実装
```

2. `src/experiments/experiment.py`の`AVAILABLE_MODELS`に新しいモデルを追加：
```python
AVAILABLE_MODELS: Dict[str, Type[Predictor]] = {
    "random_forest": RandomForestWinePredictor,
    "xgboost": XGBoostWinePredictor,
    "new_model": NewModelPredictor  # 追加
}
```

3. 新しいモデル用のテストを追加

### プロトコルの拡張

必要に応じて`src/domain/protocols.py`のインターフェースを拡張できます：

```python
class Predictor(Protocol):
    def predict_proba(self, features: WineFeatures) -> Dict[int, float]:
        """確率的な予測を行う"""
        ...
```

## 注意事項

### データ分割について
本プロジェクトの実装例では、簡略化のためデータを学習用とテスト用の2分割としていますが、実際の機械学習プロジェクトでは以下の3分割が推奨されます：

1. **学習データ（Training Data）**
   - モデルの学習に使用
   - 全データの約60%を使用することが一般的

2. **検証データ（Validation Data）**
   - ハイパーパラメータの調整に使用
   - 全データの約20%を使用することが一般的
   - 学習済みモデルの性能を評価し、最適なハイパーパラメータを選択

3. **テストデータ（Test Data）**
   - 最終的な性能評価のみに使用
   - 全データの約20%を使用することが一般的
   - ハイパーパラメータ調整後、一度だけ使用

この3分割方式を採用することで：
- より信頼性の高い性能評価が可能
- オーバーフィッティングのリスクを適切に評価可能
- テストデータが「真の未知データ」として機能

### その他の推奨事項
- Cross Validationなど、より堅牢な評価手法の採用を検討してください
- 実運用での使用時は、十分なテストと検証を行ってください
- データの前処理（標準化、正規化など）の実装を検討してください

## ライセンス
MIT License
