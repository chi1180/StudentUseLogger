"""
利用状況レポート

log.csv (timestamp, student_id, name, event) から:
  1) 現在の在室者一覧
  2) 日別・生徒別の利用時間集計
を出力する。

使い方:
    python report.py            # 在室者一覧のみ
    python report.py --daily    # 日別集計も表示
    python report.py --daily --csv out.csv   # 日別集計をCSVにも書き出す
"""
import argparse
import csv
import os
from collections import defaultdict
from datetime import datetime

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(_BASE_DIR, "data", "log.csv")


def load_events():
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def current_occupants(events):
    """各学籍番号の直近イベントを見て、inのまま(=在室中)の人を返す"""
    last_event = {}
    for row in events:
        last_event[row["student_id"]] = row
    occupants = [row for row in last_event.values() if row["event"] == "in"]
    occupants.sort(key=lambda r: r["timestamp"])
    return occupants


def daily_usage(events):
    """
    (日付, 学籍番号) -> 合計利用秒数 を計算する。
    in-outが正しくペアになっている前提。日をまたいだ利用は日付が変わった時点で打ち切る簡易実装。
    """
    totals = defaultdict(float)
    open_in = {}  # student_id -> in時刻

    for row in events:
        ts = datetime.fromisoformat(row["timestamp"])
        sid = row["student_id"]
        if row["event"] == "in":
            open_in[sid] = ts
        elif row["event"] == "out" and sid in open_in:
            in_ts = open_in.pop(sid)
            if in_ts.date() == ts.date():
                totals[(in_ts.date().isoformat(), sid)] += (ts - in_ts).total_seconds()
            # 日をまたいだ場合は簡易実装のため切り捨て（要件が出たら日単位で分割する）

    return totals


def format_duration(seconds):
    minutes = int(seconds // 60)
    h, m = divmod(minutes, 60)
    return f"{h}時間{m}分" if h else f"{m}分"


def main():
    parser = argparse.ArgumentParser(description="利用状況レポート")
    parser.add_argument("--daily", action="store_true", help="日別・生徒別の利用時間集計も表示する")
    parser.add_argument("--csv", metavar="PATH", help="日別集計をCSVファイルにも書き出す")
    args = parser.parse_args()

    events = load_events()
    if not events:
        print("log.csvにまだ記録がありません。")
        return

    name_by_id = {row["student_id"]: row["name"] for row in events}

    print("=== 現在の在室者 ===")
    occupants = current_occupants(events)
    if not occupants:
        print("(誰もいません)")
    else:
        for row in occupants:
            print(f"  {row['student_id']} {row['name']}  (入室: {row['timestamp']})")

    if args.daily:
        print("\n=== 日別利用時間 ===")
        totals = daily_usage(events)
        rows_out = []
        for (date, sid), seconds in sorted(totals.items()):
            name = name_by_id.get(sid, "?")
            print(f"  {date}  {sid} {name}: {format_duration(seconds)}")
            rows_out.append([date, sid, name, int(seconds)])

        if args.csv:
            with open(args.csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["date", "student_id", "name", "seconds"])
                writer.writerows(rows_out)
            print(f"\nCSVに書き出した: {args.csv}")


if __name__ == "__main__":
    main()
