import pandas as pd
import numpy as np
import pickle
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tqdm import tqdm  # プログレスバー表示用

# --- モデルアルゴリズムの変更点 ---
# RandomForestClassifier の代わりに XGBoost をインポート
from xgboost import XGBClassifier

# from sklearn.ensemble import RandomForestClassifier # これは使用しないためコメントアウトまたは削除


class DataLoader:
    """データのロードと前処理を行うクラス。"""

    @staticmethod
    def load_titanic_data(path="data/Titanic.csv"):
        """Titanicデータセットをロードする。"""
        if not os.path.exists(path):
            print(
                f"データファイルが見つかりません: {path}。GitHubからダウンロードを試みます。"
            )
            try:
                # GitHubのrawデータURL
                github_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
                df = pd.read_csv(github_url)
                # ダウンロードしたデータを保存
                os.makedirs(os.path.dirname(path), exist_ok=True)
                df.to_csv(path, index=False)
                print(f"データを {path} に保存しました。")
            except Exception as e:
                print(f"データダウンロード中にエラーが発生しました: {e}")
                raise FileNotFoundError(
                    f"データファイルをロードできませんでした: {path}"
                )
        return pd.read_csv(path)

    @staticmethod
    def preprocess_titanic_data(df):
        """Titanicデータセットの前処理を行う。"""
        # 不要なカラムの削除
        df = df.drop(
            columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore"
        )

        # 特徴量 (X) とターゲット (y) の分離
        X = df.drop("Survived", axis=1)
        y = df["Survived"]

        return X, y


class DataValidator:
    """データ品質の検証を行うクラス。"""

    @staticmethod
    def validate_data(df):
        """
        データフレームの品質を検証する。
        - 必須カラムの存在チェック
        - 欠損値の許容範囲チェック
        - データ型のチェック
        """
        required_columns = [
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked",
            "Survived",
        ]

        # 1. 必須カラムの存在チェック
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"以下の必須カラムが見つかりません: {', '.join(missing_columns)}"
            )

        # 2. 欠損値の許容範囲チェック (例: Ageは20%まで許容、Embarkedは5%まで許容など)
        # 今回は前処理でImputerを使うため、ここでは厳密なチェックは行わないが、
        # 必要であればここで詳細なルールを定義できる
        if df["Age"].isnull().sum() / len(df) > 0.3:  # 例: Ageの欠損が30%を超える場合
            print("警告: 'Age'カラムの欠損値が30%を超えています。")
        if df["Embarked"].isnull().sum() > 2:  # 例: Embarkedの欠損が2つを超える場合
            print("警告: 'Embarked'カラムに複数の欠損値があります。")

        # 3. データ型のチェック (例: Survivedがint型であること)
        if df["Survived"].dtype not in ["int64", "int32"]:
            print("警告: 'Survived'カラムが整数型ではありません。")

        print("データ検証結果: 成功")
        return True


class ModelTester:
    """モデルの訓練、評価、保存を行うクラス。"""

    MODEL_DIR = "models"
    MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.pkl")

    @staticmethod
    def create_preprocessing_pipeline():
        """前処理パイプラインを作成する。"""
        # 数値特徴量とカテゴリカル特徴量の定義
        numeric_features = ["Age", "SibSp", "Parch", "Fare"]
        categorical_features = ["Pclass", "Sex", "Embarked"]

        # 数値特徴量のパイプライン: 欠損値補完 (中央値) とスケーリング
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        # カテゴリカル特徴量のパイプライン: 欠損値補完 (最頻値) とOne-Hot Encoding
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        # 全ての変換をまとめる
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="passthrough",  # 定義されていないカラムはそのまま残す
        )
        return preprocessor

    @staticmethod
    def train_model(X_train, y_train, model_params=None):
        """モデルを訓練する。"""
        # --- ここが変更点: モデルをXGBClassifierに変更 ---
        if model_params is None:
            # XGBoost のデフォルトパラメータを考慮 (二値分類向け)
            model_params = {
                "objective": "binary:logistic",  # 二値分類の場合
                "eval_metric": "logloss",  # 評価指標
                "use_label_encoder": False,  # 最新のXGBoostでは警告が出るためFalse推奨
                "random_state": 42,
                "n_estimators": 100,  # モデルの複雑さを制御
                "learning_rate": 0.1,  # 学習率
                "max_depth": 3,  # ツリーの最大深度
            }

        # モデルのインスタンス化
        model = XGBClassifier(**model_params)

        # 前処理パイプラインとモデルを結合した最終パイプライン
        preprocessing_pipeline = ModelTester.create_preprocessing_pipeline()
        full_pipeline = Pipeline(
            steps=[("preprocessor", preprocessing_pipeline), ("classifier", model)]
        )

        print("モデルを訓練中...")
        full_pipeline.fit(X_train, y_train)
        print("モデル訓練完了。")
        return full_pipeline

    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """モデルを評価し、各種メトリクスを返す。"""
        print("モデルを評価中...")
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
        }
        print("モデル評価完了。")
        return metrics

    @staticmethod
    def save_model(model, path=None):
        """訓練済みモデルを保存する。"""
        if path is None:
            path = ModelTester.MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"モデルを {path} に保存しました。")

    @staticmethod
    def load_model(path=None):
        """保存されたモデルをロードする。"""
        if path is None:
            path = ModelTester.MODEL_PATH
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"モデルを {path} からロードしました。")
        return model


def main():
    """メイン処理"""
    data_path = "data/Titanic.csv"
    model_path = ModelTester.MODEL_PATH  # main2.pyではデフォルトのモデル保存パスを使用

    # 1. データのロード
    df = DataLoader.load_titanic_data(data_path)

    # 2. データ検証
    DataValidator.validate_data(df)

    # 3. データの前処理と分割
    X, y = DataLoader.preprocess_titanic_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. モデルの訓練
    # main2.py では、train_model 内で XGBoost が使われる
    trained_model = ModelTester.train_model(X_train, y_train)

    # 5. モデルの評価
    metrics = ModelTester.evaluate_model(trained_model, X_test, y_test)
    print(f"精度: {metrics['accuracy']:.4f}")
    # print(f"適合率: {metrics['precision']:.4f}")
    # print(f"再現率: {metrics['recall']:.4f}")
    # print(f"F1スコア: {metrics['f1_score']:.4f}")

    # 6. モデルの保存
    ModelTester.save_model(trained_model, path=model_path)

    # 7. 推論時間の計測 (ロードしたモデルで実施)
    loaded_model = ModelTester.load_model(path=model_path)
    start_time = time.time()
    _ = loaded_model.predict(X_test.head(10))  # 少量のデータで推論時間を計測
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"推論時間: {inference_time:.4f}秒")

    # 8. ベースライン比較 (ここでは main2.py の精度と baseline_titanic_model.pkl の精度を比較)
    # test_model.py で厳密な比較を行うため、ここでは簡単な表示のみ
    baseline_model_path = os.path.join(
        ModelTester.MODEL_DIR, "baseline_titanic_model.pkl"
    )
    if os.path.exists(baseline_model_path):
        try:
            baseline_model = ModelTester.load_model(path=baseline_model_path)
            baseline_metrics = ModelTester.evaluate_model(
                baseline_model, X_test, y_test
            )
            baseline_accuracy = baseline_metrics["accuracy"]
            if metrics["accuracy"] > baseline_accuracy:
                print("ベースライン比較: 合格 (精度向上)")
            elif metrics["accuracy"] == baseline_accuracy:
                print("ベースライン比較: 合格 (精度同等)")
            else:
                print("ベースライン比較: 不合格 (精度低下)")
            print(f"    新しいモデルの精度: {metrics['accuracy']:.4f}")
            print(f"    従来モデルの精度: {baseline_accuracy:.4f}")
        except Exception as e:
            print(f"ベースラインモデルのロードまたは評価に失敗しました: {e}")
    else:
        print(
            "ベースラインモデルが見つからないため、ベースライン比較をスキップします。"
        )


if __name__ == "__main__":
    main()
