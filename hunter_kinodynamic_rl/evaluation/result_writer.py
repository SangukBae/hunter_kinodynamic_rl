"""Write per-episode results + aggregate metrics to CSV/JSON (section 38) --
figure-ready without any additional post-processing step."""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List


def write_episode_csv(path: str, episodes: List[dict]) -> None:
    if not episodes:
        return
    fieldnames = sorted({k for e in episodes for k in e.keys() if not isinstance(e[k], (list, dict))})
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for e in episodes:
            writer.writerow(e)


def write_episode_jsonl(path: str, episodes: List[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for e in episodes:
            f.write(json.dumps(e) + "\n")


def write_summary_json(path: str, summary: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
