# -*- coding: utf-8 -*-
import json

from lslStreaming import run_streaming


def load_config(path: str = "configure.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    cfg = load_config()
    run_streaming(cfg)
