import os
import pytest
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import sys  # ★ 追加: ワークフローコマンド出力に必要


# Note: This script is designed to be run using the pytest framework.
# Ensure pytest is installed in your environment (e.g., pip install pytest).
# To run the tests, navigate to the appropriate directory in your terminal
# and use the command: pytest


# テスト用データとモデルパスを定義
DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/Titanic.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
MODEL_PATH = os.path.join(MODEL_DIR, "titanic_model.pkl")  # 現在のモデルの保存パス
BASELINE_MODEL_PATH = os.path.join(
    MODEL_DIR, "baseline_model.pkl"
)  # ★ 追加: ベースラインモデルの保存パス


@pytest.fixture
def sample_data():
    """テスト用データセットを読み込む"""
    if not os.path.exists(DATA_PATH):
        # データファイルが存在しない場合はダウンロードして保存する（テスト実行環境用）
        from sklearn.datasets import fetch_openml

        try:
            titanic = fetch_openml("titanic", version=1, as_frame=True)
            df = titanic.data
            df["Survived"] = titanic.target

            # 必要なカラムのみ選択
            df = df[
                [
                    "Pclass",
                    "Sex",
                    "Age",
                    "SibSp",
                    "Parch",
                    "Fare",
                    "Embarked",
                    "Survived",
                ]
            ]

            os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
            df.to_csv(DATA_PATH, index=False)
            print(
                f"::notice::データファイルをダウンロードして保存しました: {DATA_PATH}",
                file=sys.stdout,
            )  # ★ 追加: データ保存通知
        except Exception as e:
            print(
                f"::error::データファイルのダウンロードまたは保存に失敗しました: {e}",
                file=sys.stderr,
            )  # ★ 追加: エラー通知
            pytest.fail(f"データファイルのダウンロードまたは保存に失敗しました: {e}")

    return pd.read_csv(DATA_PATH)


@pytest.fixture
def preprocessor():
    """前処理パイプラインを定義"""
    # 数値カラムと文字列カラムを定義
    numeric_features = ["Age", "Pclass", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    # 数値特徴量の前処理（欠損値補完と標準化）
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # カテゴリカル特徴量の前処理（欠損値補完とOne-hotエンコーディング）
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # 前処理をまとめる
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


@pytest.fixture
def train_model(sample_data, preprocessor):
    """モデルの学習とテストデータの準備"""
    # データの分割とラベル変換
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42  # random_stateを固定
    )

    # モデルパイプラインの作成
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(n_estimators=100, random_state=42),
            ),  # random_stateを固定
        ]
    )

    # モデルの学習
    model.fit(X_train, y_train)

    # モデルの保存（必要であれば、テスト実行ごとに保存するかは検討）
    # test_model_accuracy_above_baseline で baseline_model.pkl をロードするため、
    # ここで現在のモデルを titanic_model.pkl として保存する必要はテストの性質上必須ではないかもしれません。
    # もし titanic_model.pkl の存在自体をテストしたい場合は残してください。
    # os.makedirs(MODEL_DIR, exist_ok=True)
    # with open(MODEL_PATH, "wb") as f:
    #     pickle.dump(model, f)

    return model, X_test, y_test  # 学習済みモデルとテストデータを返す


def test_model_exists():
    """モデルファイル (titanic_model.pkl) が存在するか確認"""
    # test_model_accuracy_above_baseline で baseline_model.pkl をチェックするので、
    # このテストは titanic_model.pkl の存在確認として残しておきます。
    if not os.path.exists(MODEL_PATH):
        print(
            f"::notice::モデルファイル (titanic_model.pkl) が存在しません。テストをスキップします。",
            file=sys.stdout,
        )  # ★ 追加: スキップ通知
        pytest.skip("モデルファイル (titanic_model.pkl) が存在しないためスキップします")
    print(
        f"::notice::モデルファイル (titanic_model.pkl) が存在します。", file=sys.stdout
    )  # ★ 追加: 存在通知
    assert os.path.exists(
        MODEL_PATH
    ), "モデルファイル (titanic_model.pkl) が存在しません"


def test_model_accuracy(train_model):
    """モデルの精度を検証"""
    model, X_test, y_test = train_model

    # 予測と精度計算
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # ★ 追加: GitHub Actions の notice コマンド形式で精度を表示
    print(
        f"::notice::test_model_accuracy: モデル精度 = {accuracy:.4f}", file=sys.stdout
    )

    # Titanicデータセットでは0.75以上の精度が一般的に良いとされる
    assert accuracy >= 0.75, f"モデルの精度が低すぎます: {accuracy:.4f}"


def test_model_inference_time(train_model):
    """モデルの推論時間を検証"""
    model, X_test, _ = train_model

    # 推論時間の計測
    start_time = time.time()
    model.predict(X_test)
    end_time = time.time()
    inference_time = end_time - start_time

    # ★ 追加: GitHub Actions の notice コマンド形式で推論時間を表示
    print(
        f"::notice::test_model_inference_time: 推論時間 = {inference_time:.4f}秒",
        file=sys.stdout,
    )

    # 推論時間が1秒未満であることを確認
    assert inference_time < 1.0, f"推論時間が長すぎます: {inference_time:.4f}秒"


def test_model_reproducibility(sample_data, preprocessor):
    """モデルの再現性を検証"""
    # データの分割
    X = sample_data.drop("Survived", axis=1)
    y = sample_data["Survived"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42  # random_stateを固定
    )

    # 同じパラメータで２つのモデルを作成
    model1 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    model2 = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # 学習
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)

    # 同じ予測結果になることを確認
    predictions1 = model1.predict(X_test)
    predictions2 = model2.predict(X_test)

    # ★ 追加: 再現性チェックの結果を表示 (成功なら差がないことを示す)
    is_reproducible = np.array_equal(predictions1, predictions2)
    print(
        f"::notice::test_model_reproducibility: 再現性チェック = {'成功' if is_reproducible else '失敗'}",
        file=sys.stdout,
    )

    assert is_reproducible, "モデルの予測結果に再現性がありません"


# --- 修正・追加するベースライン比較テスト関数 ---
# このテストで事前に保存した baseline_model.pkl をロードして比較します。
def test_model_accuracy_above_baseline(train_model):  # train_model フィクスチャを利用
    """
    現在のモデルの精度がベースラインモデルと比較して劣化していないかを検証
    （事前に保存した baseline_model.pkl をロードして比較）
    """
    current_model, X_test, y_test = train_model

    # ベースラインモデルをロード
    # test_model.py から見て ../models/baseline_model.pkl のパス
    # BASELINE_MODEL_PATH 変数はファイルの先頭で定義済み
    if not os.path.exists(BASELINE_MODEL_PATH):
        # ★ 追加: ベースラインモデルが存在しないことをnoticeで表示
        print(
            f"::notice::ベースラインモデルファイルが見つかりません: {BASELINE_MODEL_PATH}",
            file=sys.stdout,
        )
        pytest.fail(f"ベースラインモデルファイルが存在しません: {BASELINE_MODEL_PATH}")

    try:
        with open(BASELINE_MODEL_PATH, "rb") as f:
            baseline_model = pickle.load(f)
        print(
            f"::notice::ベースラインモデルをロードしました: {BASELINE_MODEL_PATH}",
            file=sys.stdout,
        )  # ★ 追加: ロード成功通知
    except Exception as e:
        # ★ 追加: ロード失敗をerrorで表示
        print(
            f"::error::ベースラインモデルのロードに失敗しました: {e}", file=sys.stderr
        )  # エラーはstderrへ
        pytest.fail(f"ベースラインモデルのロードに失敗しました: {e}")

    # 現在のモデルで予測・精度計算
    y_pred_current = current_model.predict(X_test)
    accuracy_current = accuracy_score(y_test, y_pred_current)

    # ベースラインモデルで予測・精度計算
    # ベースラインモデルも同じテストデータX_testに対して評価する
    y_pred_baseline = baseline_model.predict(X_test)
    accuracy_baseline = accuracy_score(y_test, y_pred_baseline)

    # ★ 追加: 現在の精度とベースライン精度を notice で表示
    print(
        f"::notice::test_model_accuracy_above_baseline: 現在のモデル精度 = {accuracy_current:.4f}",
        file=sys.stdout,
    )
    print(
        f"::notice::test_model_accuracy_above_baseline: ベースラインモデル精度 = {accuracy_baseline:.4f}",
        file=sys.stdout,
    )

    # 現在のモデル精度がベースライン精度以上であることを検証
    # 必要に応じて許容誤差 (tolerance) を設定しても良い
    assert (
        accuracy_current >= accuracy_baseline
    ), f"現在のモデル精度 ({accuracy_current:.4f}) がベースライン ({accuracy_baseline:.4f}) より低くなっています！"


# --- 演習2 main.py からのインポート部分と関連テスト ---
# 注: この部分と上のフィクスチャ部分でモデル学習の方法が重複しています。
# 宿題の要件に応じて、どちらかのモデル学習方法に統一しても良いでしょう。
# ここでは、提供された元のコード構造を維持しつつ、notice出力を追加します。

# このスクリプト（test_model.py）から見て2階層上の day5 ディレクトリに移動
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# そこから 演習2 ディレクトリを指定
enshu2_path = os.path.join(base_path, "演習2")

# Note: sys.path.append は環境によっては依存関係の問題を引き起こす可能性があります。
# 推奨されるのは、演習2のコードをパッケージとして適切にインポートするか、
# 演習3のコード内で必要な関数/クラスを再定義することです。
# 宿題の文脈では動作すれば良い場合もありますが、実務では注意が必要です。
if enshu2_path not in sys.path:  # ★ 追加: 重複して追加しないようにチェック
    sys.path.append(enshu2_path)

# try-except ImportError を追加すると、演習2のファイルが見つからない場合でも
# エラーを回避できますが、宿題ではファイルがある前提とします。
try:  # ★ 追加: インポートエラーを捕捉
    from main import (
        DataLoader as Enshu2DataLoader,
        ModelTester as Enshu2ModelTester,
    )  # ★ 演習2のクラス名を変更して区別
except ImportError:
    print(
        f"::warning::演習2のmain.pyが見つからないため、関連テストをスキップします。パス: {enshu2_path}",
        file=sys.stdout,
    )  # ★ 追加: 警告通知
    # 演習2のクラスが見つからなかったことを示すグローバルフラグなどを設定し、
    # 関連するテスト関数に pytest.mark.skipif を適用するのがより Pytest 的な方法です。
    # ここではシンプルに、クラスが見つからなかった場合は関連テスト関数を定義しないことでスキップと同等の効果を得ます。
    Enshu2DataLoader = None
    Enshu2ModelTester = None


# 演習2のクラスが見つかった場合のみ、関連フィクスチャとテストを定義
if Enshu2DataLoader is not None and Enshu2ModelTester is not None:

    @pytest.fixture(scope="module")
    def trained_model_and_data_from_enshu2():  # フィクスチャ名を変更して重複を避ける
        """演習2のクラスを使ってモデル学習とテストデータの準備"""
        data = Enshu2DataLoader.load_titanic_data()
        X, y = Enshu2DataLoader.preprocess_titanic_data(data)
        # データの分割（学習はtrainデータ、評価はtestデータを使用）
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42  # random_stateを固定
        )
        model = Enshu2ModelTester.train_model(X_train, y_train)  # trainデータで学習
        print(
            f"::notice::演習2クラス使用: モデル学習が完了しました。", file=sys.stdout
        )  # ★ 追加: 学習完了通知
        return model, X_test, y_test  # testデータと学習済みモデルを返す

    def test_model_performance_enshu2(
        trained_model_and_data_from_enshu2,
    ):  # 関数名を変更
        """演習2クラス使用: モデル性能のテスト (精度と推論時間)"""
        model, X_test, y_test = trained_model_and_data_from_enshu2
        metrics = Enshu2ModelTester.evaluate_model(model, X_test, y_test)

        # ★ 追加: GitHub Actions の notice コマンド形式で精度と推論時間を表示
        print(
            f"::notice::test_model_performance_enshu2: 精度 = {metrics['accuracy']:.4f}",
            file=sys.stdout,
        )
        print(
            f"::notice::test_model_performance_enshu2: 推論時間 = {metrics['inference_time']:.4f}秒",
            file=sys.stdout,
        )

        assert (
            metrics["accuracy"] >= 0.75
        ), f"精度が低すぎます: {metrics['accuracy']:.4f}"
        assert (
            metrics["inference_time"] < 1.0
        ), f"推論時間が長すぎます: {metrics['inference_time']:.3f}秒"

    # --- 修正・追加するベースライン比較テスト関数 (演習2クラス使用版) ---
    # このテストでベースラインモデルをロードして比較します。
    def test_model_accuracy_above_baseline_enshu2(
        trained_model_and_data_from_enshu2,
    ):  # 関数名を変更
        """
        演習2クラス使用: 現在のモデルの精度がベースラインモデルと比較して劣化していないかを検証
        （事前に保存した baseline_model.pkl をロードして比較）
        """
        current_model, X_test, y_test = trained_model_and_data_from_enshu2

        # ベースラインモデルをロード
        # test_model.py から見て ../models/baseline_model.pkl のパス
        # BASELINE_MODEL_PATH 変数はファイルの先頭で定義済み
        if not os.path.exists(BASELINE_MODEL_PATH):
            # ★ 追加: ベースラインモデルが存在しないことをnoticeで表示
            print(
                f"::notice::ベースラインモデルファイルが見つかりません: {BASELINE_MODEL_PATH}",
                file=sys.stdout,
            )
            pytest.fail(
                f"ベースラインモデルファイルが存在しません: {BASELINE_MODEL_PATH}"
            )

        try:
            with open(BASELINE_MODEL_PATH, "rb") as f:
                baseline_model = pickle.load(f)
            print(
                f"::notice::演習2クラス使用: ベースラインモデルをロードしました: {BASELINE_MODEL_PATH}",
                file=sys.stdout,
            )  # ★ 追加: ロード成功通知
        except Exception as e:
            # ★ 追加: ロード失敗をerrorで表示
            print(
                f"::error::演習2クラス使用: ベースラインモデルのロードに失敗しました: {e}",
                file=sys.stderr,
            )  # エラーはstderrへ
            pytest.fail(
                f"演習2クラス使用: ベースラインモデルのロードに失敗しました: {e}"
            )

        # 現在のモデルで予測・精度計算
        y_pred_current = current_model.predict(X_test)
        accuracy_current = accuracy_score(y_test, y_pred_current)

        # ベースラインモデルで予測・精度計算
        # ベースラインモデルも同じテストデータX_testに対して評価する
        y_pred_baseline = baseline_model.predict(X_test)
        accuracy_baseline = accuracy_score(y_test, y_pred_baseline)

        # ★ 追加: 現在の精度とベースライン精度を notice で表示
        print(
            f"::notice::test_model_accuracy_above_baseline_enshu2: 現在のモデル精度 = {accuracy_current:.4f}",
            file=sys.stdout,
        )
        print(
            f"::notice::test_model_accuracy_above_baseline_enshu2: ベースラインモデル精度 = {accuracy_baseline:.4f}",
            file=sys.stdout,
        )

        # 現在のモデル精度がベースライン精度以上であることを検証
        # 必要に応じて許容誤差 (tolerance) を設定しても良い
        assert (
            accuracy_current >= accuracy_baseline
        ), f"演習2クラス使用: 現在のモデル精度 ({accuracy_current:.4f}) がベースライン ({accuracy_baseline:.4f}) より低くなっています！"

else:
    # 演習2のクラスが見つからなかった場合、関連するテストをスキップする
    # pytest の skip マーカを動的に適用する方法はいくつかありますが、
    # シンプルにテスト関数を定義しないことでスキップと同等の効果を得ます。
    # より明示的にスキップしたい場合は、pytest.mark.skipif を使用します。
    pass  # テスト関数が定義されないため、pytest はこれらのテストを認識しません
