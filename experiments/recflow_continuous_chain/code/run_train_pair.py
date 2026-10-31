"""One declared daily update and its first sampled evaluation; no next model."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from experiments.recflow_full_population.code.data_common import json_save
from experiments.recflow_continuous_chain.code.prepare_update import cache_source_path, resolve_recorded


def read(path):
    return json.loads(Path(path).read_text())


def validate_dispatch_spec(spec):
    """Recheck real ancestry and the trainer's parent before any subprocess."""
    parent = resolve_recorded(spec.get("training_parent") or spec["parent"])
    if spec.get("parent") and resolve_recorded(spec["parent"]) != parent:
        raise ValueError("Legacy parent and training_parent disagree")
    source = resolve_recorded(spec.get("cache_source") or parent)
    path = cache_source_path(parent, source)
    recorded = spec.get("cache_source_to_training_parent_path")
    if recorded is not None and recorded != path:
        raise ValueError("Declared cache-source ancestry differs from actual optimizer parents")
    command = spec["training_command"]
    if (sum(token == "--parent" or token.startswith("--parent=") for token in command) != 1
            or "--parent" not in command or command[-1] == "--parent"
            or resolve_recorded(command[command.index("--parent")+1]) != parent):
        raise ValueError("Training command must retain the declared actual training parent")
    return parent, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    run = parser.parse_args().run.resolve()
    if (run / "active_process.json").exists():
        raise ValueError("This experiment was already dispatched; inspect its actual process")
    spec = read(run / "experiment.json")
    training_parent, cache_source = validate_dispatch_spec(spec)
    repo = Path(__file__).resolve().parents[3]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3", OMP_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1",
               PYTHONPATH=str(repo / "src") + os.pathsep + str(repo),
               EVOKV_ATTENTION_BACKEND="auto")

    def execute(stage, command):
        with (run / f"{stage}.log").open("x") as log:
            child = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
            state = dict(stage=stage, status="running", pid=child.pid, command=command,
                         started_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
            json_save(run / "active_process.json", state)
            result = child.wait()
        state.update(status="completed" if result == 0 else "failed", returncode=result,
                     finished_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
        json_save(run / f"{stage}_process.json", state)
        json_save(run / "active_process.json", state)
        if result:
            raise RuntimeError(f"{stage} exited {result}; preserve the run and inspect its log")

    execute("training", spec["training_command"])
    training = Path(spec["training_output"])
    trained = read(training / "completion.json")
    if trained["status"] != "completed" or trained["next_epoch"] != 1 or trained["next_step"] != 0:
        raise ValueError("The declared single full training epoch did not finish")
    checkpoint = training / read(training / "latest.json")["checkpoint"]
    meta = read(checkpoint.with_suffix(".json"))["metadata"]
    if (not checkpoint.is_file() or Path(meta["arguments"]["parent"]).resolve() != training_parent
            or meta["days"] != [spec["training_day"]]
            or meta["arguments"]["listwise_weight"] != spec["listwise_weight"]
            or meta.get("quality_use_permitted") is not True):
        raise ValueError("The saved endpoint differs from the declared actual update")
    evaluate = Path(spec["evaluation_directory"])
    prepare_command = [sys.executable, "-m", "experiments.recflow_continuous_chain.code.prepare_pair",
        "--source-checkpoint", str(cache_source), "--target-checkpoint", str(checkpoint),
        "--source-label", spec["source_label"], "--target-label", spec["target_label"],
        "--name", evaluate.name, "--users", str(spec["evaluation_users"]),
        "--offset", "0", "--question", spec["question"]]
    if spec.get("publication_scenario"):
        prepare_command.extend(["--scenario", spec["publication_scenario"]])
    execute("prepare_evaluation", prepare_command)
    execute("evaluation", [sys.executable, "-m", "experiments.recflow_continuous_chain.code.run_one",
                           "--run", str(evaluate)])
    result = read(evaluate / "completion.json")
    if result["status"] != "completed":
        raise ValueError("Sampled evaluation and independent metric audit did not finish")
    json_save(run / "completion.json", dict(status="completed", checkpoint=str(checkpoint),
        training_parent=str(training_parent), cache_source=str(cache_source),
        evaluation=str(evaluate), primary_result=result["primary_result"],
        automatic_successors=False, completed_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()))


if __name__ == "__main__":
    main()
