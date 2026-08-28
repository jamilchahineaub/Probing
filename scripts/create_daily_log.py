#!/usr/bin/env python3
"""Create today's research-log template without overwriting an existing log."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def daily_log_template(log_date: date) -> str:
    value = log_date.isoformat()
    return f"""# Daily Research Log — {value}

## What was done

## Why it was done

## Experiments

## Results

## What we learned

## Failures / unexpected behavior

## Changes to our understanding

## Next steps
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, default=date.today())
    parser.add_argument(
        "--log-directory", type=Path, default=REPOSITORY_ROOT / "lab" / "daily"
    )
    arguments = parser.parse_args()
    arguments.log_directory.mkdir(parents=True, exist_ok=True)
    output = arguments.log_directory / f"{arguments.date.isoformat()}.md"
    try:
        with output.open("x", encoding="utf-8") as stream:
            stream.write(daily_log_template(arguments.date))
    except FileExistsError:
        print(f"Daily log already exists; left unchanged: {output}")
        return 0
    print(f"Created daily log: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

