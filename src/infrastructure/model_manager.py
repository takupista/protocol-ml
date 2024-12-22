import os
from pathlib import Path
import re
import pickle
import json
from typing import Optional, Dict, Any
from datetime import datetime

from src.domain.protocols import ModelManager, Predictor
from src.models.predictors import RandomForestWinePredictor, XGBoostWinePredictor

class WineModelManager(ModelManager):
    """ワイン品質予測モデルを管理するクラス"""

    # モデルクラスのマッピング
    MODEL_CLASSES = {
        "RandomForestWinePredictor": RandomForestWinePredictor,
        "XGBoostWinePredictor": XGBoostWinePredictor
    }

    def __init__(self, model_dir: str = "./models"):
        """
        Args:
            model_dir: モデルを保存するディレクトリ
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.model_dir / "model_metadata.json"
        self._initialize_metadata()

    def _initialize_metadata(self) -> None:
        """メタデータファイルの初期化"""
        if not self.metadata_file.exists():
            self._save_metadata({})

    def _load_metadata(self) -> Dict[str, Any]:
        """メタデータの読み込み"""
        if not self.metadata_file.exists():
            return {}
        with open(self.metadata_file, 'r') as f:
            return json.load(f)

    def _save_metadata(self, metadata: Dict[str, Any]) -> None:
        """メタデータの保存"""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _validate_version(self, version: str) -> None:
        """バージョン文字列の検証"""
        if not version or not isinstance(version, str):
            raise ValueError("Version must be a non-empty string")
        
        # バージョン形式の検証 (例: v1.0, v2.1.0)
        if not re.match(r'^v\d+\.\d+(\.\d+)?$', version):
            raise ValueError(
                "Invalid version format. Must be in format 'v1.0' or 'v1.0.0'"
            )

    def save_predictor(self, predictor: Predictor, version: str) -> None:
        """
        予測モデルを保存する

        Args:
            predictor: 保存するPredictor
            version: モデルのバージョン
        """
        if predictor is None:
            raise ValueError("Predictor cannot be None")
        
        self._validate_version(version)
        
        # 既存バージョンのチェック
        metadata = self._load_metadata()
        if version in metadata:
            raise ValueError(f"Version {version} already exists")

        # モデルファイルのパス
        model_path = self.model_dir / f"{version}.pkl"

        # メタデータの更新
        metadata[version] = {
            "model_class": predictor.__class__.__name__,
            "created_at": datetime.now().isoformat(),
            "file_path": str(model_path)
        }

        # モデルの保存
        with open(model_path, 'wb') as f:
            pickle.dump(predictor, f)

        # メタデータの保存
        self._save_metadata(metadata)

    def load_predictor(self, version: str) -> Optional[Predictor]:
        """
        指定されたバージョンの予測モデルを読み込む

        Args:
            version: モデルのバージョン

        Returns:
            Optional[Predictor]: 読み込んだPredictor、存在しない場合はNone
        """
        self._validate_version(version)
        
        metadata = self._load_metadata()
        if version not in metadata:
            return None

        model_info = metadata[version]
        model_path = Path(model_info["file_path"])

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        try:
            with open(model_path, 'rb') as f:
                predictor = pickle.load(f)
                
            # モデルクラスの検証
            if not isinstance(predictor, tuple(self.MODEL_CLASSES.values())):
                raise ValueError(
                    f"Loaded model is not a valid predictor: {type(predictor)}"
                )
                
            return predictor
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")

    def list_versions(self) -> list[str]:
        """
        利用可能なモデルバージョンの一覧を返す

        Returns:
            list[str]: バージョンのリスト
        """
        metadata = self._load_metadata()
        return sorted(metadata.keys())

    def get_version_info(self, version: str) -> Optional[Dict[str, Any]]:
        """
        指定されたバージョンの詳細情報を返す

        Args:
            version: モデルのバージョン

        Returns:
            Optional[Dict[str, Any]]: バージョン情報、存在しない場合はNone
        """
        self._validate_version(version)
        metadata = self._load_metadata()
        return metadata.get(version)

    def delete_version(self, version: str) -> bool:
        """
        指定されたバージョンのモデルを削除する

        Args:
            version: 削除するモデルのバージョン

        Returns:
            bool: 削除が成功したかどうか
        """
        self._validate_version(version)
        metadata = self._load_metadata()
        
        if version not in metadata:
            return False

        # モデルファイルの削除
        model_path = Path(metadata[version]["file_path"])
        if model_path.exists():
            model_path.unlink()

        # メタデータから削除
        del metadata[version]
        self._save_metadata(metadata)

        return True
