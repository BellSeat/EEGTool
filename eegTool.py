# -*- coding: utf-8 -*-
import csv
import time
import threading


from typing import Optional

from pylsl import resolve_streams, StreamInlet

class EEGRecorder:
    """
    Robust LSL subscriber with retry + verbose logging.
    """
    def __init__(self, config: dict, out_csv_path: str):

        self.config = config
        self.out_csv_path = out_csv_path
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _find_stream(self) -> Optional[StreamInlet]:
        prop = self.config["lsl"]["eeg_resolve"]["prop"]   # "name" or "type"
        val  = self.config["lsl"]["eeg_resolve"]["value"]  # e.g., "obci_eeg1" or "EEG"
        timeout = float(self.config["lsl"]["eeg_resolve"]["timeout"])

        while not self._stop.is_set():
            streams = resolve_streams(wait_time=timeout)  # discover everything
            if not streams:
                print("[EEG] No LSL streams visible yet...")
            else:
                print(f"[EEG] Visible streams ({len(streams)}):")
                for i, si in enumerate(streams):
                    try:
                        print(f"  [{i}] name={si.name()} type={si.type()} "
                              f"ch={si.channel_count()} srate={si.nominal_srate()} "
                              f"source_id={si.source_id()}")
                    except Exception:
                        pass

                # filter by prop
                def match(si):
                    try:
                        if prop.lower() == "name": return si.name() == val
                        if prop.lower() == "type": return si.type() == val
                        if prop.lower() == "source_id": return si.source_id() == val
                        # fallback: custom property in XML
                        return si.desc().child_value(prop) == val
                    except Exception:
                        return False

                matches = [s for s in streams if match(s)]
                if matches:
                    si = matches[0]
                    print(f"[EEG] Matched stream: name={si.name()} type={si.type()} "
                          f"ch={si.channel_count()} srate={si.nominal_srate()}")
                    inlet = StreamInlet(si, max_buflen=300)
                    corr = inlet.time_correction()
                    print(f"[EEG] time_correction ≈ {corr:.4f}s")
                    return inlet
                else:
                    print(f"[EEG] No stream matched {prop}='{val}'. Retrying...")

            time.sleep(2.0)  # wait and retry

        return None

    def _loop(self):
        inlet = self._find_stream()
        if inlet is None:
            print("[EEG] Stopped before a stream was found.")
            return

        header_written = False
        wrote = 0
        last_report = time.time()

        with open(self.out_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            while not self._stop.is_set():
                chunk, ts = inlet.pull_chunk(timeout=1.0, max_samples=256)
                if ts:
                    if not header_written:
                        ch_n = len(chunk[0]) if chunk else 0
                        w.writerow(["lsl_ts"] + [f"ch{i+1}" for i in range(ch_n)])
                        header_written = True
                        print(f"[EEG] Writing CSV with {ch_n} channels -> {self.out_csv_path}")

                    for row, t in zip(chunk, ts):
                        w.writerow([f"{t:.9f}"] + list(row))
                    wrote += len(ts)

                # progress log
                now = time.time()
                if now - last_report >= 1.0:
                    if wrote:
                        print(f"[EEG] +{wrote} samples written in last second")
                        wrote = 0
                    last_report = now

        print("[EEG] Loop ended.")

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
