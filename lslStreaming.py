# -*- coding: utf-8 -*-
import cv2
import csv
import json
import time
import queue
import pathlib
import threading
from datetime import datetime
from typing import Dict, Any, Optional

from pylsl import StreamInfo, StreamOutlet, local_clock
from eegTool import EEGRecorder


class EventWriter:
    """Asynchronous CSV writer for events."""
    def __init__(self, out_csv: str):
        self.out_csv = out_csv
        self._q: "queue.Queue[tuple]" = queue.Queue()
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._stop.clear()
        self._th.start()

    def stop(self):
        self._stop.set()
        self._th.join(timeout=2.0)

    def push(self, kind: str, label: str, extra: Dict[str, Any]):
        ts = local_clock()
        self._q.put((ts, kind, label, json.dumps(extra, ensure_ascii=False)))

    def _loop(self):
        with open(self.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ts_lsl", "kind", "label", "extra_json"])
            while not self._stop.is_set() or not self._q.empty():
                try:
                    ts, kind, label, extra_json = self._q.get(timeout=0.2)
                    w.writerow([f"{ts:.9f}", kind, label, extra_json])
                except queue.Empty:
                    pass


class Teleprompter:
    """Runs the script only after .start() is called; supports pause/resume."""
    def __init__(self, script: list, evt: EventWriter):
        self.script = script
        self.state: Dict[str, Any] = {"current": None}
        self._stop = threading.Event()
        self._paused = threading.Event()   # when set() => paused
        self._th: Optional[threading.Thread] = None
        self.evt = evt

    def start(self):
        if self._th and self._th.is_alive():
            return
        self._stop.clear()
        self._paused.clear()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()
        self._paused.clear()
        if self._th:
            self._th.join(timeout=2.0)
        self.state["current"] = None

    def pause(self):
        self._paused.set()

    def resume(self):
        self._paused.clear()

    def _loop(self):
        for block in self.script:
            if self._stop.is_set():
                break

            label = str(block["label"])
            total = float(block["duration_s"])
            prompt = str(block.get("prompt", ""))

            # event: phase_start
            phase_start_lsl = local_clock()
            self.evt.push("phase_start", label, {"prompt": prompt, "planned_dur_s": total})

            remaining = total
            last_tick = time.perf_counter()
            active_elapsed = 0.0

            while not self._stop.is_set() and remaining > 0:
                # pause holds the timer (no countdown during pause)
                if self._paused.is_set():
                    time.sleep(0.05)
                    last_tick = time.perf_counter()  # prevent time jump after resume
                    continue

                now = time.perf_counter()
                dt = now - last_tick
                last_tick = now
                remaining = max(0.0, remaining - dt)
                active_elapsed = total - remaining

                self.state["current"] = {
                    "label": label,
                    "prompt": prompt,
                    "remain_s": remaining,
                    "since_lsl": active_elapsed  # active time (excludes paused)
                }
                time.sleep(0.05)

            # event: phase_end
            self.evt.push("phase_end", label, {"actual_dur_s": active_elapsed})

        self.state["current"] = None
        print("[Teleprompter] Script ended.")


def draw_overlay(frame, state: Optional[Dict[str, Any]], status_text: str):
    """Render teleprompter overlay + recording status on the frame."""
    if state:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        bar_h = max(80, int(0.22 * h))
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

        margin = 20
        label = state.get("label", "")
        prompt = state.get("prompt", "")
        remain = float(state.get("remain_s", 0.0))

        cv2.putText(frame, f"{label}", (margin, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{prompt}", (margin, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Remaining: {remain:4.1f}s", (margin, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # bottom-left recording status
    h, _ = frame.shape[:2]
    cv2.putText(frame, status_text, (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 0, 255) if status_text in ("RECORDING", "PAUSED") else (255, 255, 255),
                2, cv2.LINE_AA)
    return frame


def run_streaming(cfg: dict):
    """Main runner: LSL outputs, events, teleprompter, EEG inlet, video+hotkeys."""
    outdir = pathlib.Path(cfg["output_dir"] + f"/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    outdir.mkdir(exist_ok=True, parents=True)

    # LSL outputs: Markers / VideoTime
    marker_info = StreamInfo(
        name=cfg["lsl"]["marker_stream_name"], type='Markers', channel_count=1,
        nominal_srate=0, channel_format='string', source_id='exp_markers_v1'
    )
    marker_out = StreamOutlet(marker_info)

    video_info = StreamInfo(
        name=cfg["lsl"]["video_stream_name"], type='Video', channel_count=1,
        nominal_srate=0, channel_format='double64', source_id='video_time_v1'
    )
    video_out = StreamOutlet(video_info)

    def lsl_mark(kind: str, label: str):
        marker_out.push_sample([f"{kind}:{label}"], timestamp=local_clock())

    # Events + Teleprompter
    events_csv = str(outdir / "events.csv")
    evt = EventWriter(events_csv); evt.start()
    tele = Teleprompter(cfg["script"], evt)   # <-- remove .start() here
    with open(outdir / "script_used.json", "w", encoding="utf-8") as f:
        json.dump(cfg["script"], f, ensure_ascii=False, indent=2)

    # EEG inlet
    eeg_csv = str(outdir / "eeg_lsl.csv")
    eeg_rec = EEGRecorder(cfg, eeg_csv); eeg_rec.start()

    # Camera / window
    cam = cfg["camera"]
    win_title = cfg.get("window_title", "Recording + Teleprompter")
    vid_path = str(outdir / "video.mp4")
    vid_ts_csv = str(outdir / "video_ts.csv")

    cap = cv2.VideoCapture(cam["index"], cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])
    cap.set(cv2.CAP_PROP_FPS, cam["fps"])

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or cam["width"]
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or cam["height"]
    fps = cap.get(cv2.CAP_PROP_FPS) or cam["fps"]

    if cam.get("fullscreen", False):
        cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(vid_path, fourcc, fps, (width, height))
    tsf = open(vid_ts_csv, "w", newline="", encoding="utf-8")
    tsw = csv.writer(tsf); tsw.writerow(["frame_idx", "video_lsl_ts", "recording_state"])

    # Hotkeys
    hk = cfg["hotkeys"]
    key_start = ord(hk["record_start"])
    key_pause = ord(hk["pause_resume"])
    key_quit  = ord(hk["quit"])
    key_marker = ord(hk["marker"])  # space: ' '

    recording = False
    paused = False
    frame_idx = 0

    print(f"[INFO] Hotkeys: Start='{hk['record_start']}', Pause/Resume='{hk['pause_resume']}', "
          f"Marker='SPACE', Quit='{hk['quit']}'")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            status_text = "READY"
            if recording:
                status_text = "PAUSED" if paused else "RECORDING"

            frame_overlay = draw_overlay(frame, tele.state["current"], status_text)
            cv2.imshow(win_title, frame_overlay)

            k = cv2.waitKey(1) & 0xFF
            if k == key_start:  # 'R'
                if not recording:
                    recording = True
                    paused = False
                    tele.start()  # <-- start teleprompter when recording starts
                    evt.push("video_start", "video", {"width": width, "height": height, "fps": fps})
                    lsl_mark("video_start", "video")
                    print("[R] Recording started.")

            elif k == key_pause:
                if recording:
                    paused = not paused
                    if paused:
                        tele.pause()    # <-- pause teleprompter countdown
                    else:
                        tele.resume()   # <-- resume teleprompter countdown
                    state = "paused" if paused else "resumed"
                    evt.push(f"video_{state}", "video", {})
                    lsl_mark(f"video_{state}", "video")
                    print(f"[P] Recording {state}.")
            elif k == key_marker:
                cur = tele.state["current"]
                label = cur["label"] if cur else "none"
                evt.push("marker", label, {"key": "space"})
                lsl_mark("marker", label)
            elif k == key_quit:
                if recording:
                    evt.push("video_end", "video", {"frames": frame_idx})
                    lsl_mark("video_end", "video")
                break

            if recording and not paused:
                ts_lsl = local_clock()
                video_out.push_sample([ts_lsl], timestamp=ts_lsl)
                vw.write(frame_overlay)
                tsw.writerow([frame_idx, f"{ts_lsl:.9f}", status_text])
                frame_idx += 1
    finally:
        cap.release(); vw.release(); tsf.close()
        cv2.destroyWindow(win_title)

        tele.stop()
        eeg_rec.stop()
        evt.stop()

        print(f"[DONE] Files saved to: {outdir.resolve()}")
