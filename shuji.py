import streamlit as st
import easyocr
import cv2
import numpy as np
import hashlib
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from difflib import SequenceMatcher
import os

# Streamlitページ設定
st.set_page_config(page_title="習字サポート", layout="centered")
st.title('Shuji')
st.write('習字の練習をサポートします。')
st.write('一文字だけアップロードしてください。')


# 画像から文字をトリミングする。
def find_largest_bbox(contours):
    if not contours:  # 輪郭が検出されなかった場合
        return None

    # 全ての輪郭を包含する最小の矩形を見つける
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    # 整数にキャストする
    return (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))


def crop_and_resize_character(image, new_size=(300, 300)):
    # PILのImageオブジェクトをNumPy配列に変換
    if isinstance(image, Image.Image):
        image = np.array(image)

    # OpenCVを使用して画像をグレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二値化処理
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ノイズ除去
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # 輪郭検出
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 最大の外接矩形を見つける
    bbox = find_largest_bbox(contours)
    if bbox:
        x, y, w, h = bbox
        # トリミング
        cropped_image = image[y:y + h, x:x + w]
    else:
        # 輪郭が見つからなかった場合は元の画像を使用
        cropped_image = image

    # 画像のサイズを変更
    resized_image = cv2.resize(cropped_image, new_size, interpolation=cv2.INTER_AREA)

    return resized_image


def add_text_to_image(image, text, reference_char, font_path='玉ねぎ楷書激無料版v7改.ttf'):
    # オリジナル画像のサイズを取得
    image_size = image.size

    # フォントサイズを調整
    font_size = image_size[1] - 30

    # テキスト画像の生成
    text_image = Image.new("RGBA", image_size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(text_image)

    # フォントの読み込み
    font = ImageFont.truetype(font_path, font_size)

    # テキストのバウンディングボックスを計算
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # テキストを中央に配置
    x = (image_size[0] - text_width) / 2
    y = (image_size[1] - text_height) / 2

    # テキストの描画
    draw.text((x, y), text, fill=(255, 0, 0, 110), font=font)

    # 点線の補助線を描画
    line_color = (0, 0, 0, 140)  # 黒色
    dotted_line_space = 6  # 点線の間隔

    # 水平線
    for i in range(0, image_size[0], dotted_line_space * 2):
        draw.line([(i, image_size[1] // 2), (i + dotted_line_space, image_size[1] // 2)], fill=line_color)

    # 垂直線
    for i in range(0, image_size[1], dotted_line_space * 2):
        draw.line([(image_size[0] // 2, i), (image_size[0] // 2, i + dotted_line_space)], fill=line_color)

    # オリジナル画像の上にテキスト画像を重ねる
    combined_image = Image.alpha_composite(image.convert("RGBA"), text_image)

    # お手本の文字を追加
    reference_font_size = 30
    reference_font = ImageFont.truetype(font_path, reference_font_size)
    reference_text_width, reference_text_height = draw.textbbox((0, 0), reference_char, font=reference_font)[2:4]
    reference_x = (image_size[0] - reference_text_width) / 2
    reference_y = (image_size[1] - reference_text_height) / 4  # Adjust position as needed
    draw.text((reference_x, reference_y), reference_char, fill=(0, 0, 255, 110), font=reference_font)

    return combined_image

# フォント差異の評価
def evaluate_font_similarity(reference_font_path, uploaded_font_path):
    # フォントの比較処理（実際の比較ロジックは実際のフォントファイルに依存します）
    # 例: フォントのハッシュ値を計算して比較する
    with open(reference_font_path, 'rb') as rf:
        reference_font_hash = hashlib.md5(rf.read()).hexdigest()

    with open(uploaded_font_path, 'rb') as uf:
        uploaded_font_hash = hashlib.md5(uf.read()).hexdigest()

    return reference_font_hash == uploaded_font_hash

# 文字列の類似度比率を計算する関数
def calculate_similarity_score(reference_char, uploaded_text):
    return SequenceMatcher(None, reference_char, uploaded_text).ratio()

# ファイルアップロード
uploaded_file = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])

# アップロードされたフォントファイルの保存先ディレクトリ
upload_dir = 'uploaded_fonts'
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

# ストリームリットアプリケーションのメイン部分
if uploaded_file is not None:
    # PILを使用して画像を読み込む
    image = Image.open(BytesIO(uploaded_file.getvalue()))

    # 画像をリサイズしてファイルサイズを減らす
    base_width = 600
    w_percent = (base_width / float(image.size[0]))
    h_size = int((float(image.size[1]) * float(w_percent)))
    image = image.resize((base_width, h_size), Image.Resampling.LANCZOS)

    # 画像をJPEG形式に変換してさらにサイズを軽減
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    image = Image.open(buffered)

    # EasyOCRリーダーの初期化
    reader = easyocr.Reader(['ja'])

    # OCR処理
    image_for_ocr = np.array(image)  # PIL ImageをNumPy配列に変換
    results = reader.readtext(image_for_ocr)

    if len(results) >= 2:
        st.error('一文字だけで再度アップロードしてください。')
    else:
        text = results[0][1]

        # OCRの結果を表示
        st.write(f"OCRの結果: {text}")

        # ユーザが手本の文字を選ぶ
        reference_char = st.text_input("手本の文字を入力してください:", value=text[0])

        # フォント差異の評価
        reference_font_path = '玉ねぎ楷書激無料版v7改.ttf'
        uploaded_font_path = os.path.join(upload_dir, 'uploaded_font.ttf')  # アップロードされたフォントの保存先
        if not os.path.exists(uploaded_font_path):  # ファイルが存在しない場合は作成
            with open(uploaded_font_path, 'wb'):
                pass

        # 文字の一致度を計算
        font_similarity_score = evaluate_font_similarity(reference_font_path, uploaded_font_path)
        similarity_score = calculate_similarity_score(reference_char, text)

        # 総合的な採点スコアを計算
        total_score = (font_similarity_score + similarity_score) / 2  # 例として単純な平均をとる

        # 採点スコアを表示
        st.write(f"フォント差異スコア: {font_similarity_score * 100:.2f}%")
        st.write(f"一致度スコア: {similarity_score * 100:.2f}%")
        st.write(f"総合的な採点スコア: {total_score * 100:.2f}%")

        # トリミングとリサイズ
        resized_image = crop_and_resize_character(image)  # PIL Imageオブジェクトを渡す
        resized_image = Image.fromarray(resized_image)  # NumPy配列をPIL Imageに変換

        # お手本テキストの追加
        final_image = add_text_to_image(resized_image, text, reference_char)

        # 処理した画像を表示
        st.image(final_image, caption='Processed Image', use_column_width=True)
