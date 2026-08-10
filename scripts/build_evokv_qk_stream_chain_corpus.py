from __future__ import annotations

import argparse
import json
from pathlib import Path

from hstu_kvcache.data.qk_stream_chain import (
    PROTOCOL,
    QKStreamChainConfig,
    build_corpus,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    path = parse_args().config
    document = json.loads(path.read_text())
    if document.get("protocol") != PROTOCOL:
        raise ValueError("QK stream data config protocol differs")
    data = document["data"]
    roles = document["roles"]
    outputs = document["outputs"]
    summary = build_corpus(
        QKStreamChainConfig(
            source=Path(data["source"]),
            member=str(data["member"]),
            catalog=Path(data["catalog"]),
            user_lengths=Path(data["user_lengths"]),
            theta0_corpus=Path(data["theta0_corpus"]),
            final_workload=Path(data["final_workload"]),
            roles_output=Path(outputs["roles"]),
            corpus_output=Path(outputs["corpus"]),
            summary_output=Path(outputs["summary"]),
            base_prefix=int(data["base_prefix"]),
            maximum_sequence_length=int(data["maximum_sequence_length"]),
            update_count=int(data["update_count"]),
            stream_train_users=int(roles["stream_train_users"]),
            fit_tuning_users=int(roles["fit_tuning_users"]),
            qualification_users=int(roles["qualification_users"]),
            final_users=int(roles["final_users"]),
            short_diagnostic_users=int(roles["short_diagnostic_users"]),
            minimum_long_events=int(roles["minimum_long_events"]),
            short_minimum_events=int(roles["short_minimum_events"]),
            short_maximum_events=int(roles["short_maximum_events"]),
            selection_salt=str(roles["selection_salt"]),
            chunk_size=int(data["chunk_size"]),
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
