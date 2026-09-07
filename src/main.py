"""
生徒証スキャン → CSVログ記録 メインループ

前提:
  - calibrate_roi.py を先に実行して roi_config.json を作っておくこと
  - roster.csv に学籍番号と氏名のマスタがあること

流れ:
  IDLE -> 検出中 -> 安定確認 -> OCR実行 -> 照合 -> 記録 -> クールダウン -> IDLE
"""
import csv
import glob
import json
import os
import re
import time
from collections import Counter, deque
from datetime import datetime

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont

from tts_announcer import Announcer, SOUND_FAIL, SOUND_SUCCESS, format_duration, play_sound

CAMERA_INDEX = "/dev/video3"  # L-12W。パスを直接指定（整数indexは内部列挙とズレることがある）
ROOM_NAME = "自習室"  # 退室アナウンスで使う部屋名

# パスはスクリプトの場所基準で一度だけ解決し、以後は全関数でこの解決済みパスを使う
# （os.path.exists()とopen()で別の基準を使うと存在チェックがズレるバグの元になる）
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_BASE_DIR, "data", "roi_config.json")
ROSTER_PATH = os.path.join(_BASE_DIR, "data", "roster.csv")
LOG_PATH = os.path.join(_BASE_DIR, "data", "log.csv")

# 差分検出のしきい値・安定確認フレーム数
DIFF_THRESHOLD = 15          # 1ピクセルあたりの差分をこの値で二値化
DIFF_PIXEL_RATIO = 0.03      # 枠内の何%のピクセルが変化したらトリガーか
STABLE_FRAMES = 8            # このフレーム数連続でほぼ無変化なら「安定」とみなす
STABLE_DIFF_RATIO = 0.01     # 安定判定の変化許容率
REMOVE_STABLE_FRAMES = 5     # カードが取り除かれたと判定するまでの連続フレーム数

ID_ROI_PADDING = 0           # id_roiの外側にさらに余白を足すピクセル数（環境によっては誤読の元になるので0でも可）
OCR_SAMPLES = 7              # OCR多数決に使うフレーム数

FEEDBACK_SECONDS = 1.8       # 記録結果を画面に表示し続ける秒数


_JP_FONT_CANDIDATES = [
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/TTF/NotoSansCJK-Regular.ttc",
]
_jp_font_cache = {}


def get_jp_font(size):
    """日本語対応フォントを探して読み込む（cv2.putTextは日本語を描画できないため）"""
    if size in _jp_font_cache:
        return _jp_font_cache[size]
    for path in _JP_FONT_CANDIDATES:
        if os.path.exists(path):
            font = ImageFont.truetype(path, size)
            _jp_font_cache[size] = font
            return font
    found = glob.glob("/usr/share/fonts/**/*CJK*.ttc", recursive=True) + \
        glob.glob("/usr/share/fonts/**/*CJK*.otf", recursive=True)
    if found:
        font = ImageFont.truetype(found[0], size)
        _jp_font_cache[size] = font
        return font
    _jp_font_cache[size] = None
    return None


def put_japanese_text(img_bgr, text, org, color_bgr, size=32):
    """日本語を含むテキストをBGR画像に描画する。対応フォントが無ければ何もしない"""
    font = get_jp_font(size)
    if font is None:
        return img_bgr
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(org, text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_roster():
    """
    roster.csv を読み込み、漢字名（画面表示・ログ用）とふりがな（音声用）を分けて返す。
    furigana列が無い既存CSVは漢字名をそのまま音声用に使う（後方互換）。
    """
    roster = {}    # student_id -> 漢字名
    readings = {}  # student_id -> ふりがな
    with open(ROSTER_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = row["student_id"].strip()
            name = row["name"].strip()
            roster[sid] = name
            reading = row.get("furigana", "").strip()
            readings[sid] = reading or name
    return roster, readings


def ensure_log_file():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp", "student_id", "name", "event"])


def get_last_event(student_id):
    """CSVを末尾から見て、student_idの直近のイベント行(dict)を返す。記録が無ければ None"""
    if not os.path.exists(LOG_PATH):
        return None
    last = None
    with open(LOG_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["student_id"] == student_id:
                last = dict(row)
    return last


def append_log(student_id, name, event):
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([datetime.now().isoformat(timespec="seconds"), student_id, name, event])


def find_camera():
    """/dev/video* を直接パスで順に試して、実際にフレームが取れたデバイスを返す"""
    import glob

    for path in sorted(glob.glob("/dev/video*")):
        cap = cv2.VideoCapture(path)  # バックエンド自動選択
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        if ok and frame is not None:
            print(f"{path} でカメラを開けた（解像度: {frame.shape[1]}x{frame.shape[0]}）")
            return cap
        cap.release()
    raise RuntimeError(
        "有効なカメラが見つからなかった。`v4l2-ctl --list-devices` で"
        "ノード番号を確認して CAMERA_INDEX に \"/dev/videoN\" の形で直接指定してみて"
    )


def open_camera(retries=5, retry_wait=1.0):
    """CAMERA_INDEXを開く。USB瞬断対策で数回リトライする"""
    if CAMERA_INDEX is None:
        return find_camera()
    for attempt in range(retries):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
        cap.release()
        print(f"カメラを開けなかった（{attempt + 1}/{retries}回目）。{retry_wait}秒待って再試行...")
        time.sleep(retry_wait)
    raise RuntimeError(
        f"カメラを開けなかった（CAMERA_INDEX={CAMERA_INDEX!r}）。USB切断の可能性が高い"
    )


def crop(frame, roi):
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    return frame[y:y + h, x:x + w]


def crop_padded(frame, roi, padding):
    """roiの外側にpaddingピクセル分の余白を足して切り出す（フレーム範囲は超えない）"""
    height, width = frame.shape[:2]
    x = max(0, roi["x"] - padding)
    y = max(0, roi["y"] - padding)
    x2 = min(width, roi["x"] + roi["w"] + padding)
    y2 = min(height, roi["y"] + roi["h"] + padding)
    return frame[y:y2, x:x2]


def diff_ratio(frame_a, frame_b, threshold=DIFF_THRESHOLD):
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_a, gray_b)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return np.count_nonzero(mask) / mask.size


def ocr_student_id(id_crop, debug_save=True):
    gray = cv2.cvtColor(id_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)  # MJPEGのブロックノイズを軽減
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ラミネート反射などで生じる小さい穴・ノイズを軽く均す
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    if debug_save:
        # ROIが正しい位置を切り出せているか目視確認するためのデバッグ画像
        cv2.imwrite(os.path.join(_BASE_DIR, "data", "debug_id_crop.png"), id_crop)
        cv2.imwrite(os.path.join(_BASE_DIR, "data", "debug_id_binary.png"), binary)

    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(binary, config=config)
    digits = re.sub(r"\D", "", text)
    return digits


def capture_empty_base(cap, card_roi):
    """
    起動直後にいきなり最初のフレームを基準にすると、
    机の上にカードが乗ったままの状態を「空」として誤って記憶してしまう。
    ユーザーが本当に机を空にしてからキーを押すまで待つ。
    """
    print("机の上に何も置かれていない状態にしてください。準備できたら's'キーを押してください。")
    frame = None
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        display = frame.copy()
        x, y, w, h = card_roi["x"], card_roi["y"], card_roi["w"], card_roi["h"]
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(display, "press 's' when desk is empty", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("id_scanner", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            break
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            raise SystemExit(0)
    return crop(frame, card_roi)


def main():
    config = load_config()
    roster, readings = load_roster()
    ensure_log_file()

    card_roi = config["card_roi"]
    id_roi = config["id_roi"]

    cap = open_camera()
    announcer = Announcer()  # kokoro初期化をバックグラウンドで開始（初回はモデルDLで時間がかかる）

    try:
        empty_base = capture_empty_base(cap, card_roi)  # 「何も置かれていない机」の基準フレーム

        state = "IDLE"
        stable_buffer = deque(maxlen=STABLE_FRAMES)
        remove_buffer = deque(maxlen=REMOVE_STABLE_FRAMES)
        ocr_samples = []
        last_prev_card = empty_base

        # 画面フィードバック用の状態（記録の成否をしばらく表示する）
        feedback_until = 0
        feedback_text = ""
        feedback_color = (0, 255, 0)

        print("起動した。ガイド枠にカードを置いてください。qで終了。")

        fail_count = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                fail_count += 1
                if fail_count >= 10:
                    print("カメラからの読み取りが続けて失敗。再接続を試みる...")
                    cap.release()
                    cap = open_camera()
                    fail_count = 0
                continue
            fail_count = 0

            card_now = crop(frame, card_roi)
            display = frame.copy()
            x, y, w, h = card_roi["x"], card_roi["y"], card_roi["w"], card_roi["h"]

            now = time.time()
            if now < feedback_until:
                # 成功/失敗を大きく表示している間は枠を太く・色付きにする
                cv2.rectangle(display, (x, y), (x + w, y + h), feedback_color, 6)
                display = put_japanese_text(display, feedback_text, (10, 45), feedback_color, size=32)
            else:
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display, state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("id_scanner", display)

            if state == "IDLE":
                ratio = diff_ratio(empty_base, card_now)
                if ratio > DIFF_PIXEL_RATIO:
                    state = "DETECTING"
                    stable_buffer.clear()

            elif state == "DETECTING":
                ratio = diff_ratio(last_prev_card, card_now)
                stable_buffer.append(ratio < STABLE_DIFF_RATIO)
                if len(stable_buffer) == STABLE_FRAMES and all(stable_buffer):
                    state = "OCR"
                    ocr_samples = []

            elif state == "OCR":
                id_crop = crop_padded(frame, id_roi, ID_ROI_PADDING)
                result = ocr_student_id(id_crop)
                ocr_samples.append(result)

                if len(ocr_samples) >= OCR_SAMPLES:
                    # 数字らしい長さ(6〜10桁)の結果だけを対象に多数決。無ければ全サンプルから多数決。
                    plausible = [s for s in ocr_samples if 6 <= len(s) <= 10]
                    pool = plausible if plausible else ocr_samples
                    student_id, count = Counter(pool).most_common(1)[0]
                    print(f"[OCR多数決] {ocr_samples} -> '{student_id}' ({count}/{len(ocr_samples)}票)")

                    if student_id in roster:
                        name = roster[student_id]
                        last_event = get_last_event(student_id)
                        event = "out" if last_event and last_event["event"] == "in" else "in"
                        append_log(student_id, name, event)
                        print(f"[記録] {student_id} {name} -> {event}")

                        if event == "in":
                            feedback_text = f"{name} 入室"
                            feedback_color = (0, 200, 0)
                            play_sound(SOUND_SUCCESS)  # 即時ビープ
                            announcer.speak(
                                f"{readings[student_id]}さん、入室を確認しました。",
                                fallback_sound=SOUND_SUCCESS,
                            )
                        else:
                            dur_text = format_duration(
                                (datetime.now() - datetime.fromisoformat(last_event["timestamp"]))
                                .total_seconds()
                            )
                            feedback_text = f"{name} 退室"
                            feedback_color = (0, 165, 255)
                            play_sound(SOUND_SUCCESS)  # 即時ビープ
                            announcer.speak(
                                f"{readings[student_id]}さん、{ROOM_NAME}を{dur_text}利用しました。退室を確認しました。",
                                fallback_sound=SOUND_SUCCESS,
                            )
                        feedback_until = now + FEEDBACK_SECONDS
                    else:
                        print(f"[未照合] OCR結果: '{student_id}' はroster.csvに一致なし。"
                              "一度カードをどけてから置き直してください（debug_id_crop.pngを確認）")

                        feedback_text = "認識できません。置き直してください"
                        feedback_color = (0, 0, 255)
                        feedback_until = now + FEEDBACK_SECONDS
                        play_sound(SOUND_FAIL)  # 即時エラー音
                        announcer.speak("認識できませんでした。カードを置き直してください。",
                                        fallback_sound=SOUND_FAIL)

                    # 結果に関わらず、カードが取り除かれるまでは再トリガーしない
                    remove_buffer.clear()
                    state = "WAIT_REMOVAL"

            elif state == "WAIT_REMOVAL":
                ratio = diff_ratio(empty_base, card_now)
                remove_buffer.append(ratio < DIFF_PIXEL_RATIO)
                if len(remove_buffer) == REMOVE_STABLE_FRAMES and all(remove_buffer):
                    print("カードが取り除かれた。待機状態に戻ります。")
                    state = "IDLE"

            last_prev_card = card_now

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中断された。終了処理をします。")
