import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from huggingface_hub import hf_hub_download, login # 関数のインポートを追記

# ────────────
# 1. Hugging Face トークンの取得とログイン
# ────────────
# Streamlit の "Secrets" に保存したトークンを読み込み
hf_token = st.secrets["HUGGINGFACE_TOKEN"]

# Hugging Face 上で認証セッションを開始
login(token=hf_token)

# ────────────
# 2. モデル読み込み関数の定義
# ────────────
@st.cache_resource
def load_yolo_model(repo_id: str, filename: str) -> YOLO:
    """
    YOLO モデルを Hugging Face からダウンロードして読み込む関数。
    @st.cache_resource を付けることで、
    「一度ダウンロード・初期化したモデル」をキャッシュし、以降は再利用します。

    Args:
        repo_id (str): Hugging Face 上のリポジトリパス（例: "username/model-name"）
        filename (str): リポジトリ内のモデルファイル名（例: "best.pt"）

    Returns:
        YOLO: 初期化済みの YOLO モデルオブジェクト
    """
    # モデルファイルを認証付きでダウンロード
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=hf_token
    )
    
    # YOLO モデルを初期化
    model = YOLO(model_path)
    return model

# ────────────
# 3. 実際にモデルをロード（初回だけダウンロード／初期化される）
# ────────────
MODEL_REPO = "username/modelname"  # ←自分のリポジトリに書き換えてください
MODEL_FILE = "best.pt"                   # ←実際のモデルファイル名に書き換えてください

model = load_yolo_model(repo_id=MODEL_REPO, filename=MODEL_FILE)

# ────────────
# 4. streamlitアプリの中身
# ────────────

st.title("🍣 寿司ネタ 検知アプリ")
st.write("画像をアップロードすると、寿司ネタを検出します。")


# Input
upload_img = st.file_uploader("画像ファイルを選択", type = ['png', 'jpg'])

# Process
if upload_img is not None:

    # 読み込んだ画像をPythonで扱えるようにするための処理
    bytes_data = upload_img.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8),
    cv2.IMREAD_COLOR)

    # 物体検知を実行
    results = model(img)

    # 結果を出力
    output_img = results[0].plot(labels = True, conf = True)

# Output
    st.image(output_img, channels="BGR", caption='出力画像')

