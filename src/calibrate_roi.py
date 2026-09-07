"""
ROIキャリブレーションツール

固定カメラの映像を見ながら
  1) カード全体が置かれる枠 (card_roi)
  2) 学籍番号(No.)が印字されている領域 (id_roi)
をそれぞれドラッグで指定して roi_config.json に保存する。

使い方:
    python calibrate_roi.py
    -> ウィンドウが開くのでまずカード全体の枠をドラッグしてEnter
    -> 続けて学籍番号欄の枠をドラッグしてEnter
    -> qで終了
"""
import json
import time
import cv2
import os

CAMERA_INDEX = "/dev/video3"  # L-12W。パスを直接指定（整数indexは内部列挙とズレることがある）
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "./data/roi_config.json")


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


def select_roi(window_name, frame):
    print(f"[{window_name}] ウィンドウが開いたらドラッグして範囲選択 → 選択後は Enter か Space で確定（Escや'c'はキャンセル）")
    r = cv2.selectROI(window_name, frame, showCrosshair=True)
    cv2.destroyWindow(window_name)
    x, y, w, h = r
    if w == 0 or h == 0:
        raise RuntimeError(
            f"[{window_name}] 選択がキャンセルされた（幅か高さが0）。"
            "ドラッグで範囲を作ってからEnter/Spaceで確定してね"
        )
    print(f"[{window_name}] 確定: x={x}, y={y}, w={w}, h={h}")
    return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}


def main():
    if CAMERA_INDEX is None:
        cap = find_camera()
    else:
        cap = None
        for attempt in range(5):
            cap = cv2.VideoCapture(CAMERA_INDEX)  # バックエンド自動選択
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    break
            cap.release()
            cap = None
            print(f"開けなかった（{attempt + 1}/5回目）。1秒待って再試行...")
            time.sleep(1)
        if cap is None:
            raise RuntimeError(
                f"カメラを開けなかった（CAMERA_INDEX={CAMERA_INDEX!r}）。"
                "USB切断が起きている可能性が高いので dmesg -w を確認して"
            )

    print("カードを置く前の、何もない机の状態でプレビューを確認してください。")
    print("キーボードで's'を押すとその瞬間のフレームでROI選択に入ります。")

    frame = None
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        cv2.imshow("preview (press 's' to capture)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            cv2.destroyWindow("preview (press 's' to capture)")
            break
        if key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()

    print("=== 1. カード全体の枠を選択します ===")
    card_roi = select_roi("1. card_roi - drag then Enter", frame)
    print("=== 2. 学籍番号(No.)欄を選択します ===")
    id_roi = select_roi("2. id_roi - drag then Enter", frame)

    config = {"card_roi": card_roi, "id_roi": id_roi}
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"保存した: {CONFIG_PATH}")
    print(config)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
