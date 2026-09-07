"""
生徒証スキャン → CSVログ記録 メインループ

前提:
  - calibrate_roi.py を先に実行して roi_config.json を作っておくこと
  - roster.csv に学籍番号と氏名のマスタがあること

流れ:
  IDLE -> 検出中 -> 安定確認 -> OCR実行 -> 照合 -> 記録 -> クールダウン -> IDLE
"""
import csv
import json
import os
import re
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import pytesseract

CAMERA_INDEX = "/dev/video3"  # L-12W。パスを直接指定（整数indexは内部列挙とズレることがある）
CONFIG_PATH = "./data/roi_config.json"
ROSTER_PATH = "./data/roster.csv"
LOG_PATH = "./data/log.csv"

# 差分検出のしきい値・安定確認フレーム数
DIFF_THRESHOLD = 15          # 1ピクセルあたりの差分をこの値で二値化
DIFF_PIXEL_RATIO = 0.03      # 枠内の何%のピクセルが変化したらトリガーか
STABLE_FRAMES = 8            # このフレーム数連続でほぼ無変化なら「安定」とみなす
STABLE_DIFF_RATIO = 0.01     # 安定判定の変化許容率
COOLDOWN_SECONDS = 5         # 同じイベント後、次のトリガーまでの無視時間


def load_config():
    with open(os.path.join(os.path.dirname(__file__), CONFIG_PATH), encoding="utf-8") as f:
        return json.load(f)


def load_roster():
    roster = {}
    with open(os.path.join(os.path.dirname(__file__), ROSTER_PATH), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            roster[row["student_id"].strip()] = row["name"].strip()
    return roster


def ensure_log_file():
    if not os.path.exists(LOG_PATH):
        with open(os.path.join(os.path.dirname(__file__), LOG_PATH), "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp", "student_id", "name", "event"])


def get_last_status(student_id):
    """CSVを末尾から見て、直近のイベントが in か out かを返す。記録が無ければ None"""
    if not os.path.exists(LOG_PATH):
        return None
    last = None
    with open(os.path.join(os.path.dirname(__file__), LOG_PATH), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["student_id"] == student_id:
                last = row["event"]
    return last


def append_log(student_id, name, event):
    with open(os.path.join(os.path.dirname(__file__), LOG_PATH), "a", newline="", encoding="utf-8") as f:
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


def diff_ratio(frame_a, frame_b, threshold=DIFF_THRESHOLD):
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_a, gray_b)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return np.count_nonzero(mask) / mask.size


def ocr_student_id(id_crop):
    gray = cv2.cvtColor(id_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(binary, config=config)
    digits = re.sub(r"\D", "", text)
    return digits


def main():
    config = load_config()
    roster = load_roster()
    ensure_log_file()

    card_roi = config["card_roi"]
    id_roi = config["id_roi"]

    cap = open_camera()

    ok, base_frame = cap.read()
    if not ok:
        raise RuntimeError("最初のフレームが取得できなかった")
    base_card = crop(base_frame, card_roi)

    state = "IDLE"
    stable_buffer = deque(maxlen=STABLE_FRAMES)
    last_prev_card = base_card
    cooldown_until = 0

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
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(display, state, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("id_scanner", display)

        now = time.time()

        if state == "IDLE":
            if now >= cooldown_until:
                ratio = diff_ratio(base_card, card_now)
                if ratio > DIFF_PIXEL_RATIO:
                    state = "DETECTING"
                    stable_buffer.clear()

        elif state == "DETECTING":
            ratio = diff_ratio(last_prev_card, card_now)
            stable_buffer.append(ratio < STABLE_DIFF_RATIO)
            if len(stable_buffer) == STABLE_FRAMES and all(stable_buffer):
                state = "OCR"

        elif state == "OCR":
            id_crop = crop(frame, id_roi)
            student_id = ocr_student_id(id_crop)

            if student_id in roster:
                name = roster[student_id]
                last_status = get_last_status(student_id)
                event = "out" if last_status == "in" else "in"
                append_log(student_id, name, event)
                print(f"[記録] {student_id} {name} -> {event}")
            else:
                print(f"[未照合] OCR結果: '{student_id}' はroster.csvに一致なし。再スキャンしてください")

            cooldown_until = now + COOLDOWN_SECONDS
            state = "COOLDOWN"

        elif state == "COOLDOWN":
            if now >= cooldown_until:
                # クールダウン明けにベースを更新（カードが置かれたままでも誤検出しないように）
                base_card = card_now
                state = "IDLE"

        last_prev_card = card_now

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
