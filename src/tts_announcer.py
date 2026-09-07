"""
kokoro (日本語 jf_alpha) による音声アナウンス

カメラループをブロックしないよう、バックグラウンドスレッド + キューで
テキストを合成（24kHz WAV）し、paplay で再生する。

使い方（カメラ無しの動作テスト）:
    uv run python src/tts_announcer.py
"""
import os
import queue
import subprocess
import tempfile
import threading

import numpy as np

VOICE = "jf_alpha"
SPEED = 1.0
SAMPLE_RATE = 24000

SOUND_SUCCESS = "/usr/share/sounds/freedesktop/stereo/complete.oga"
SOUND_FAIL = "/usr/share/sounds/freedesktop/stereo/dialog-error.oga"


def play_sound(path):
    """通知音を非同期で即時再生。paplayが無い/ファイルが無い環境でも落ちないようにする"""
    try:
        subprocess.Popen(
            ["paplay", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass  # paplayが無い環境では無音で継続


def format_duration(seconds):
    """秒を日本語の時間表記にする（例: 2時間35分, 35分）"""
    total = max(0, int(seconds // 60))
    h, m = divmod(total, 60)
    if h and m:
        return f"{h}時間{m}分"
    if h:
        return f"{h}時間"
    if m:
        return f"{m}分"
    return "1分未満"


def _play_wav(path):
    """生成済みWAVを再生（ブロッキング。終わったら呼び出し側で削除してよい）"""
    try:
        subprocess.run(
            ["paplay", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        pass


class Announcer:
    """
    kokoroでテキストを合成して発話する。

    - コンストラクタでバックグラウンドの初期化/ワーカースレッドを開始する
      （初回はモデルDLのため初期化に数分かかる。その間の speak() はキューに溜まる）
    - speak() は即座に戻るため、カメラループをブロックしない
    - 合成に失敗した場合は fallback_sound の通知音にフォールバックする
    """

    def __init__(self, voice=VOICE, speed=SPEED):
        self.voice = voice
        self.speed = speed
        self._pipeline = None
        self._ready = threading.Event()
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        threading.Thread(target=self.warmup, daemon=True).start()

    def warmup(self):
        """モデルDL + KPipeline初期化をバックグラウンドで行う（初回起動は低速）"""
        try:
            from kokoro import KPipeline

            print("kokoro TTS (日本語 jf_alpha) の初期化を開始。初回はモデルDLで数分かかります...")
            self._pipeline = KPipeline(lang_code="j")
            print("kokoro TTS の初期化が完了しました。")
        except Exception as e:
            print(f"kokoro TTS の初期化に失敗しました: {e}")
        finally:
            self._ready.set()

    def speak(self, text, fallback_sound=None):
        """テキストを発話キューに積む（非同期・即復帰）"""
        self._queue.put((text, fallback_sound))

    def _worker(self):
        while True:
            text, fallback = self._queue.get()
            try:
                self._ready.wait()
                if self._pipeline is None:
                    if fallback:
                        play_sound(fallback)
                    continue
                try:
                    self._speak_sync(text)
                except Exception as e:
                    print(f"[TTS失敗] {e}")
                    if fallback:
                        play_sound(fallback)
            finally:
                self._queue.task_done()

    def _speak_sync(self, text):
        audios = []
        for _gs, _ps, audio in self._pipeline(text, voice=self.voice, speed=self.speed):
            audios.append(audio.detach().cpu().numpy())
        if not audios:
            raise RuntimeError("TTSの合成結果が空だった")
        combined = np.concatenate(audios)

        fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="kokoro_")
        os.close(fd)
        try:
            import soundfile as sf

            sf.write(wav_path, combined, SAMPLE_RATE)
            _play_wav(wav_path)
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass


if __name__ == "__main__":
    announcer = Announcer()
    for text in [
        "渡辺さん、入室を確認しました。",
        "渡辺さん、自習室を2時間35分利用しました。退室を確認しました。",
        "認識できませんでした。カードを置き直してください。",
    ]:
        print(f"発話: {text}")
        announcer.speak(text)
    announcer._queue.join()  # 全発話の合成・再生が完了するまで待つ
    print("全発話が完了しました。")