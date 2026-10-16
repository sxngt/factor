"""Build the per-move table of the 2026-10-16 real runs from the runner logs.

Each move's end time is the timestamp in the name of its hindsight record (``move_YYYYMMDD_HHMMSS``);
the interval between consecutive records is the wall-clock time of the later move.
실기 실행 로그에서 수별 표를 만든다. 수의 끝 시각은 기록 파일 이름의 시각이고, 연속 기록 사이 간격이
뒤 수의 소요 시간이다.

    python results/build_real_table.py > results/real_moves_20261016.csv
"""

from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = [
    # (log, label, settings) / (로그, 이름, 설정)
    ("raw/real/run_145001_cautious.log", "cautious", "step 3 mm, insert lead 4 mm, grip wait 2 s, NUDGE on"),
    ("raw/real/run_150353_bold.log", "bold", "step 10 mm, insert lead 12 mm, grip wait 0.8 s, NUDGE off"),
]


def parse(path: Path):
    moves, cur = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"== (ring\d): (\w+) -> (\w+)", line)
        if m:
            cur = {"ring": m.group(1), "from": m.group(2), "to": m.group(3), "outcome": "", "seated": "", "end": None}
            moves.append(cur)
            continue
        if cur is None:
            continue
        r = re.search(r"recorded (\S+) \(seated (\w+)\)", line)
        if r:
            cur["seated"] = r.group(2)
            t = re.search(r"move_(\d{8}_\d{6})", r.group(1))
            cur["end"] = dt.datetime.strptime(t.group(1), "%Y%m%d_%H%M%S") if t else None
        for word in ("released", "max_steps", "uncertain"):
            if line.strip() == word and not cur["outcome"]:
                cur["outcome"] = word
        if "servo stream refused" in line:
            # The traceback can print before the move's own "released" line (stdout buffering), so the
            # refusal belongs to the next move. / 출력 순서상 거부는 다음 수의 것이다.
            cur["refused_after"] = True
    return moves


def main() -> None:
    w = sys.stdout.write
    w("run,index,ring,from,to,outcome,seated,end_time,interval_s,settings\n")
    for log, label, settings in RUNS:
        prev = None
        for i, mv in enumerate(parse(HERE / log), start=1):
            if not mv["outcome"]:
                mv["outcome"] = "not started (e-stop)"
            elif mv["outcome"] == "released" and mv["seated"] == "True":
                mv["outcome"] = "seated"
            interval = ""
            if mv["end"] is not None and prev is not None:
                interval = f"{(mv['end'] - prev).total_seconds():.0f}"
            end = mv["end"].strftime("%H:%M:%S") if mv["end"] else ""
            w(f'{label},{i},{mv["ring"]},{mv["from"]},{mv["to"]},{mv["outcome"]},{mv["seated"]},{end},{interval},"{settings}"\n')
            prev = mv["end"] or prev


if __name__ == "__main__":
    main()
