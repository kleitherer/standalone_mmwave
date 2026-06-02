#!/usr/bin/env python3
"""
Wrapper to run OSC plotting with gesture.mode forced to config5.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--settings", type=Path, default=_ROOT / "config/live_radar_to_max.json")
    args, rest = parser.parse_known_args()

    settings_path = args.settings.expanduser().resolve()
    if not settings_path.is_file():
        raise FileNotFoundError(f"Settings not found: {settings_path}")

    settings = json.loads(settings_path.read_text())
    gesture = settings.get("gesture", {})
    if not isinstance(gesture, dict):
        gesture = {}
        settings["gesture"] = gesture
    gesture["mode"] = "config5"

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        tf.write(json.dumps(settings, indent=2))
        temp_settings = Path(tf.name)

    cmd = [
        sys.executable,
        str((_ROOT / "utils" / "plot_config3_osc.py").resolve()),
        "--settings",
        str(temp_settings),
        *rest,
    ]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    finally:
        temp_settings.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

