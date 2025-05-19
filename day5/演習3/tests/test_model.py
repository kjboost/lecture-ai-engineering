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
from itertools import (
    chain,
)  # ColumnTransformerのデバッグ情報に使用されているためインポート


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

# ベースラインモデルの学習に使用した特徴量列名（小文字）
# create_baseline_model.py と同じ定義である必要がある
BASELINE_FEATURE_COLUMNS = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
]
TARGET_COLUMN = "survived"

# 演習2のコードおよびベースラインモデルが期待するであろう列名と順序（ログから推測）
# エラーログから Age, Fare, SibSp, Parch, Pclass, Sex, Embarked が大文字始まりと推測
# ただし、test_model_accuracy_above_baseline が baseline_model に対して小文字で成功していることから、
# baseline_model は実際には小文字を期待している可能性が高い。
# 演習2のクラスが学習時に大文字を期待していると仮定して、ENSHU2_EXPECTED_COLUMNS を定義
ENSHU2_EXPECTED_COLUMNS_UPPER = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
]


@pytest.fixture
def sample_data():
    """テスト用データセットを読み込む"""
    if not os.path.exists(DATA_PATH):
        # データファイルが存在しない場合はダウンロードして保存する（テスト実行環境用）
        from sklearn.datasets import fetch_openml

        try:
            titanic = fetch_openml("titanic", version=1, as_frame=True)
            df = titanic.data  # 特徴量データフレーム
            df[TARGET_COLUMN] = titanic.target  # 目的変数をデータフレームに追加

            # ロードしたデータフレームの列名を全て小文字に変換
            df.columns = df.columns.str.lower()  # ★ 修正: ここで列名を小文字に変換

            # ベースラインモデルの学習に使用した必要なカラムのみを選択
            # 目的変数カラムも加えておく
            required_columns_with_target = BASELINE_FEATURE_COLUMNS + [TARGET_COLUMN]

            # 必要な列が存在するかチェックし、存在するものだけを選択
            missing_cols = [
                col for col in required_columns_with_target if col not in df.columns
            ]
            if missing_cols:
                print(
                    f"::error::ダウンロードしたデータに不足している列があります: {missing_cols}",
                    file=sys.stderr,
                )
                pytest.fail(
                    f"ダウンロードしたデータに不足している列があります: {missing_cols}"
                )

            df = df[
                required_columns_with_target
            ].copy()  # 必要な列を選択し、コピーを作成してSettingWithCopyWarningを防ぐ

            # データディレクトリが存在しない場合は作成
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

    # ローカルに保存されているデータファイルを読み込む
    data = pd.read_csv(DATA_PATH)
    # ロードしたデータの列名も小文字に統一（ダウンロード時と形式を合わせる）
    data.columns = data.columns.str.lower()  # ★ 修正: ここでも列名を小文字に変換

    # 必要な列のみを選択して返す（ファイルに保存されているデータに余分な列が含まれている可能性も考慮）
    required_columns_with_target = BASELINE_FEATURE_COLUMNS + [TARGET_COLUMN]
    # 必要な列が存在するかチェック
    missing_cols = [
        col for col in required_columns_with_target if col not in data.columns
    ]
    if missing_cols:
        print(
            f"::error::ロードしたデータに不足している列があります: {missing_cols}",
            file=sys.stderr,
        )
        pytest.fail(f"ロードしたデータに不足している列があります: {missing_cols}")

    return data[required_columns_with_target].copy()  # 必要な列を選択して返す


@pytest.fixture
def preprocessor():
    """前処理パイプラインを定義"""
    # 数値カラムと文字列カラムを定義（列名を小文字に統一している想定）
    numeric_features = ["age", "pclass", "sibsp", "parch", "fare"]  # Pclass -> pclass
    categorical_features = ["sex", "embarked"]

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
    # ColumnTransformerに渡す特徴量列名は、ベースラインモデル学習時と同じである必要がある
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numeric_features,
            ),  # ★ 修正: 列名リストとトランスフォーマーの位置を修正
            (
                "cat",
                categorical_transformer,
                categorical_features,
            ),  # ★ 修正: 列名リストとトランスフォーマーの位置を修正
        ],
        remainder="drop",  # 指定されていない列は削除
    )

    return preprocessor


@pytest.fixture
def train_model(sample_data, preprocessor):
    """モデルの学習とテストデータの準備"""
    # sample_dataフィクスチャから取得したデータは既に列名が小文字になっている想定

    # データの分割とラベル変換
    X = sample_data.drop(TARGET_COLUMN, axis=1)  # 列名を小文字に修正
    y = sample_data[TARGET_COLUMN].astype(int)  # 列名を小文字に修正
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
    # sample_dataフィクスチャから取得したデータは既に列名が小文字になっている想定

    # データの分割
    X = sample_data.drop(TARGET_COLUMN, axis=1)  # 列名を小文字に修正
    y = sample_data[TARGET_COLUMN].astype(int)  # 列名を小文字に修正
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
    current_model, X_test, y_test = (
        train_model  # train_modelフィクスチャから取得したX_testは列名が小文字になっている想定
    )

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
    y_pred_baseline = baseline_model.predict(
        X_test
    )  # X_testの列名がベースラインモデルの期待する形式になっている必要がある
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
        # 演習2のDataLoaderが返すデータの列名形式に注意が必要
        # ここでも列名を小文字に変換する処理を追加するのが安全
        if isinstance(data, pd.DataFrame):
            data.columns = data.columns.str.lower()  # ★ 追加: 列名を小文字に変換

        # 必要な特徴量列のみを選択（演習2のDataLoader/ModelTesterが不要な列を削除しているか不明なため安全策）
        # 目的変数カラムも加えておく
        required_columns_with_target = BASELINE_FEATURE_COLUMNS + [TARGET_COLUMN]
        # 必要な列が存在するかチェック
        missing_cols = [
            col for col in required_columns_with_target if col not in data.columns
        ]
        if missing_cols:
            print(
                f"::error::演習2クラス使用: ロードしたデータに不足している列があります: {missing_cols}",
                file=sys.stderr,
            )
            pytest.fail(
                f"演習2クラス使用: ロードしたデータに不足している列があります: {missing_cols}"
            )

        data = data[
            required_columns_with_target
        ].copy()  # 必要な列を選択し、コピーを作成

        # preprocess_titanic_dataがXとyを返す想定だが、
        # Enshu2DataLoader.load_titanic_dataでsurvived列を含んだDataFrameを返しているので、
        # ここでは単にXとyに分割する処理を行う（演習3のDataLoaderと同様の修正を適用）
        if (
            isinstance(data, pd.DataFrame) and TARGET_COLUMN in data.columns
        ):  # dataにsurvived列を含む場合
            y = data[TARGET_COLUMN]
            X = data.drop(TARGET_COLUMN, axis=1)
        else:
            print(
                f"::error::演習2クラス使用: データに '{TARGET_COLUMN}' 列が見つかりません。",
                file=sys.stderr,
            )
            pytest.fail(
                f"演習2クラス使用: データに '{TARGET_COLUMN}' 列が見つかりません。"
            )

        # データの分割（学習はtrainデータ、評価はtestデータを使用）
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42  # random_stateを固定
        )

        # 演習2のModelTester.train_modelに渡す前に、列名を演習2が期待する形式に変換し、列順序を合わせる
        # 演習2のColumnTransformerは ENSHU2_EXPECTED_COLUMNS_UPPER と同じ列名（大文字始まり）を期待していると仮定
        column_mapping_to_enshu2 = {
            "pclass": "Pclass",
            "sex": "Sex",
            "age": "Age",
            "sibsp": "SibSp",
            "parch": "Parch",
            "fare": "Fare",
            "embarked": "Embarked",
        }

        X_train_enshu2 = X_train.copy()
        # マッピングに基づいて列名を変換
        X_train_enshu2.rename(columns=column_mapping_to_enshu2, inplace=True)
        # 演習2が期待する順序に列を並べ替え (存在しない列はエラーになるので注意)
        try:
            X_train_enshu2 = X_train_enshu2[
                ENSHU2_EXPECTED_COLUMNS_UPPER
            ]  # ★ 修正: 列の並べ替えに使用するリストを修正
        except KeyError as e:
            print(
                f"::error::演習2クラス使用: 学習用データ変換エラー - 期待する列の一部がデータに存在しません: {e}",
                file=sys.stderr,
            )
            pytest.fail(
                f"演習2クラス使用: 学習用データ変換エラー - 期待する列の一部がデータに存在しません: {e}"
            )

        # データ型を標準化（任意だが安全のため）
        # 演習2のColumnTransformerが期待する型に合わせる必要がある
        try:
            X_train_enshu2 = X_train_enshu2.astype(
                {col: "float64" for col in ["Age", "Fare"]}
                | {col: "object" for col in ["Sex", "Embarked"]}
                | {col: "int64" for col in ["Pclass", "SibSp", "Parch"]}
            )  # ★ 追加: データ型の標準化
        except Exception as e:
            print(
                f"::warning::演習2クラス使用: 学習用データ型変換エラー: {e}",
                file=sys.stdout,
            )

        # ★ 追加: 演習2の学習に渡す直前の列名と型を出力
        print(
            f"::notice::演習2クラス使用: 学習用データ(X_train_enshu2)の列名: {list(X_train_enshu2.columns)}",
            file=sys.stdout,
        )
        print(
            f"::notice::演習2クラス使用: 学習用データ(X_train_enshu2)のデータ型:\n{X_train_enshu2.dtypes}",
            file=sys.stdout,
        )

        model = Enshu2ModelTester.train_model(
            X_train_enshu2, y_train
        )  # ★ 修正: 変換・並べ替え後のデータで学習
        print(
            f"::notice::演習2クラス使用: モデル学習が完了しました。", file=sys.stdout
        )  # ★ 追加: 学習完了通知
        return model, X_test, y_test  # testデータと学習済みモデルを返す

    def test_model_performance_enshu2(
        trained_model_and_data_from_enshu2,
    ):  # 関数名を変更
        """演習2クラス使用: モデル性能のテスト (精度と推論時間)"""
        model, X_test, y_test = (
            trained_model_and_data_from_enshu2  # X_testは列名が小文字になっている想定
        )

        # 演習2クラス使用のテストでは、X_testも演習2が期待する形式に変換し、列順序を合わせる必要がある
        column_mapping_to_enshu2 = {
            "pclass": "Pclass",
            "sex": "Sex",
            "age": "Age",
            "sibsp": "SibSp",
            "parch": "Parch",
            "fare": "Fare",
            "embarked": "Embarked",
        }
        X_test_enshu2 = X_test.copy()
        # マッピングに基づいて列名を変換
        X_test_enshu2.rename(columns=column_mapping_to_enshu2, inplace=True)
        # 演習2が期待する順序に列を並べ替え (存在しない列はエラーになるので注意)
        try:
            X_test_enshu2 = X_test_enshu2[
                ENSHU2_EXPECTED_COLUMNS_UPPER
            ]  # ★ 修正: 列の並べ替えに使用するリストを修正
        except KeyError as e:
            print(
                f"::error::演習2クラス使用: 評価用データ変換エラー - 期待する列の一部がデータに存在しません: {e}",
                file=sys.stderr,
            )
            pytest.fail(
                f"演習2クラス使用: 評価用データ変換エラー - 期待する列の一部がデータに存在しません: {e}"
            )

        # データ型を標準化（任意だが安全のため）
        # 演習2のColumnTransformerが期待する型に合わせる必要がある
        try:
            X_test_enshu2 = X_test_enshu2.astype(
                {col: "float64" for col in ["Age", "Fare"]}
                | {col: "object" for col in ["Sex", "Embarked"]}
                | {col: "int64" for col in ["Pclass", "SibSp", "Parch"]}
            )  # ★ 追加: データ型の標準化
        except Exception as e:
            print(
                f"::warning::演習2クラス使用: 評価用データ型変換エラー: {e}",
                file=sys.stdout,
            )

        # ★ 追加: 演習2の評価に渡す直前の列名と型を出力
        print(
            f"::notice::演習2クラス使用: 評価用データ(X_test_enshu2)の列名: {list(X_test_enshu2.columns)}",
            file=sys.stdout,
        )
        print(
            f"::notice::演習2クラス使用: 評価用データ(X_test_enshu2)のデータ型:\n{X_test_enshu2.dtypes}",
            file=sys.stdout,
        )

        metrics = Enshu2ModelTester.evaluate_model(
            model, X_test_enshu2, y_test
        )  # ★ 修正: 変換・並べ替え後のデータで評価

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
    # このテストで事前に保存した baseline_model.pkl をロードして比較します。
    def test_model_accuracy_above_baseline_enshu2(
        trained_model_and_data_from_enshu2,
    ):  # 関数名を変更
        """
        演習2クラス使用: 現在のモデルの精度がベースラインモデルと比較して劣化していないかを検証
        （事前に保存した baseline_model.pkl をロードして比較）
        """
        current_model, X_test, y_test = (
            trained_model_and_data_from_enshu2  # X_testは列名が小文字になっている想定
        )

        # 演習2クラス使用のテストでは、ベースラインモデルに渡すデータもベースラインモデルが期待する形式に変換
        # baseline_model.pkl は create_baseline_model.py で作成されており、
        # create_baseline_model.py の ColumnTransformer は ENSHU2_AND_BASELINE_EXPECTED_COLUMNS と同じ列名（大文字始まり）を期待していると推測される
        # しかし、test_model_accuracy_above_baseline が X_test (小文字) で成功しているため、
        # baseline_model は小文字の列名を期待している可能性が高い。
        # ここでは baseline_model には小文字の列名を持つ X_test をそのまま渡す。
        # current_model は演習2クラスで学習されており、大文字の列名で学習されていると仮定し、
        # current_model には列名変換した X_test_baseline を渡す。

        column_mapping_to_enshu2 = {
            "pclass": "Pclass",
            "sex": "Sex",
            "age": "Age",
            "sibsp": "SibSp",
            "parch": "Parch",
            "fare": "Fare",
            "embarked": "Embarked",
        }
        X_test_baseline = X_test.copy()
        # マッピングに基づいて列名を変換
        X_test_baseline.rename(columns=column_mapping_to_enshu2, inplace=True)
        # 演習2が期待する順序に列を並べ替え (存在しない列はエラーになるので注意)
        try:
            X_test_baseline = X_test_baseline[
                ENSHU2_EXPECTED_COLUMNS_UPPER
            ]  # ★ 修正: 列の並べ替えに使用するリストを修正
        except KeyError as e:
            print(
                f"::error::演習2クラス使用: ベースライン比較用データ変換エラー - 期待する列の一部がデータに存在しません: {e}",
                file=sys.stderr,
            )
            pytest.fail(
                f"演習2クラス使用: ベースライン比較用データ変換エラー - 期待する列の一部がデータに存在しません: {e}"
            )

        # データ型を標準化（任意だが安全のため）
        # 演習2のColumnTransformerが期待する型に合わせる必要がある
        try:
            X_test_baseline = X_test_baseline.astype(
                {col: "float64" for col in ["Age", "Fare"]}
                | {col: "object" for col in ["Sex", "Embarked"]}
                | {col: "int64" for col in ["Pclass", "SibSp", "Parch"]}
            )  # ★ 追加: データ型の標準化
        except Exception as e:
            print(
                f"::warning::演習2クラス使用: ベースライン比較用データ型変換エラー: {e}",
                file=sys.stdout,
            )

        # ★ 追加: ベースライン比較(演習2)に渡す直前の列名と型を出力 (current_model用)
        print(
            f"::notice::演習2クラス使用: current_model比較用データ(X_test_baseline)の列名: {list(X_test_baseline.columns)}",
            file=sys.stdout,
        )
        print(
            f"::notice::演習2クラス使用: current_model比較用データ(X_test_baseline)のデータ型:\n{X_test_baseline.dtypes}",
            file=sys.stdout,
        )

        # ★ 追加: ベースライン比較(演習2)に渡す直前の列名と型を出力 (baseline_model用)
        print(
            f"::notice::演習2クラス使用: baseline_model比較用データ(X_test)の列名: {list(X_test.columns)}",
            file=sys.stdout,
        )
        print(
            f"::notice::演習2クラス使用: baseline_model比較用データ(X_test)のデータ型:\n{X_test.dtypes}",
            file=sys.stdout,
        )

        # ベースラインモデルをロード
        # test_model.py から見て ../models/baseline_model.pkl のパス
        # BASELINE_MODEL_PATH 変数はファイルの先頭で定義済み
        if not os.path.exists(BASELINE_MODEL_PATH):
            # ★ 追加: ベースラインモデルが存在しないことをnoticeで表示
            print(
                f"::notice::ベースラインモデルファイルが見つかりません: {BASELINE_MODEL_MODEL_PATH}",
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

            # ★ 追加: ベースラインモデルのpreprocessorが期待する列名を出力
            try:
                # ベースラインモデルの最初のステップがColumnTransformerであると仮定
                # ColumnTransformerがPipelineの最初のステップでない場合、このアクセスは失敗する可能性があります。
                if isinstance(baseline_model.steps[0][1], ColumnTransformer):
                    baseline_preprocessor_feature_names_in = baseline_model.steps[0][
                        1
                    ].feature_names_in_
                    print(
                        f"::notice::演習2クラス使用: ベースラインモデルpreprocessor期待列名: {list(baseline_preprocessor_feature_names_in)}",
                        file=sys.stdout,
                    )
                else:
                    print(
                        f"::warning::演習2クラス使用: ベースラインモデルの最初のステップはColumnTransformerではありません。",
                        file=sys.stdout,
                    )

            except Exception as e:
                print(
                    f"::warning::演習2クラス使用: ベースラインモデルpreprocessor期待列名取得失敗: {e}",
                    file=sys.stdout,
                )

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
        y_pred_current = current_model.predict(
            X_test_baseline
        )  # ★ 修正: 変換・並べ替え後のデータで予測
        accuracy_current = accuracy_score(y_test, y_pred_current)

        # ベースラインモデルで予測・精度計算
        # ベースラインモデルも同じテストデータX_testに対して評価する
        y_pred_baseline = baseline_model.predict(
            X_test
        )  # ★ 修正: 小文字の列名データで予測
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
