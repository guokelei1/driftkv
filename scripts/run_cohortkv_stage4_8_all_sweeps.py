from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
FAMILIES = (
    (
        "staggered_renewal",
        SCRIPT_DIR / "run_cohortkv_stage4_8_staggered_renewal_sweep.py",
    ),
    (
        "token_debt",
        SCRIPT_DIR / "run_cohortkv_stage4_8_token_debt_sweep.py",
    ),
    (
        "aoi_maxweight",
        SCRIPT_DIR / "run_cohortkv_stage4_8_aoi_maxweight_sweep.py",
    ),
    (
        "model_time_renewal",
        SCRIPT_DIR / "run_cohortkv_stage4_8_model_time_renewal_sweep.py",
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", nargs=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args(argv)


def family_command(
    script: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [sys.executable, str(script)]
    if args.devices is not None:
        command.extend(["--devices", *args.devices])
    if args.force:
        command.append("--force")
    if args.smoke_test:
        command.append("--smoke-test")
    return command


def run_all(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    for index, (family, script) in enumerate(FAMILIES, start=1):
        print(
            f"[{index}/{len(FAMILIES)}] start {family}",
            flush=True,
        )
        family_started = time.perf_counter()
        completed = subprocess.run(
            family_command(script, args),
            cwd=ROOT,
            check=False,
        )
        elapsed = time.perf_counter() - family_started
        if completed.returncode != 0:
            print(
                f"[{index}/{len(FAMILIES)}] failed {family} "
                f"returncode={completed.returncode} "
                f"elapsed_seconds={elapsed:.1f}; stop",
                flush=True,
            )
            return completed.returncode or 1
        print(
            f"[{index}/{len(FAMILIES)}] complete {family} "
            f"elapsed_seconds={elapsed:.1f}",
            flush=True,
        )
    print(
        "all Stage 4.8 families complete "
        f"elapsed_seconds={time.perf_counter() - started:.1f}",
        flush=True,
    )
    return 0


def main() -> None:
    raise SystemExit(run_all(parse_args()))


if __name__ == "__main__":
    main()
