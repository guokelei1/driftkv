from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .kuairand import KuaiRandTrace, load_kuairand


@dataclass
class StreamingDataPlan:
    """Day-segmented streaming data plan for temporal online-learning evaluation.

    Implements the standard protocol:
      1. Base training on [day0, day_base) -> theta_0
      2. For each streaming day t:
         a. EVAL: predict day t's interactions using theta_{t-1}
            (history = each user's interactions up to end of day t-1)
         b. INGEST: append day t's interactions into user histories
         c. TRAIN: incremental update on day t's data -> theta_t,  dtheta_t

    This produces realistic per-day Delta-theta (small, hourly-scale updates) and
    leak-free next-day evaluation - exactly the scenario the drift research needs.
    """

    trace: KuaiRandTrace
    base_dates: list[str]
    stream_dates: list[str]
    max_seq_len: int = 128
    max_items: int | None = None
    user_histories: dict[int, dict] = field(default_factory=dict)
    daily_segments: dict[str, pd.DataFrame] = field(default_factory=dict)

    def __post_init__(self) -> None:
        df = self.trace.interactions.copy()
        df["date"] = df["date"].astype(str)
        for date, grp in df.groupby("date"):
            self.daily_segments[date] = grp.sort_values("time_ms").reset_index(drop=True)
        all_dates = sorted(self.daily_segments.keys())
        if not self.base_dates:
            self.base_dates = all_dates[: max(1, len(all_dates) // 2)]
        if not self.stream_dates:
            self.stream_dates = [d for d in all_dates if d not in set(self.base_dates)]
        for u in range(1, self.trace.num_users + 1):
            self.user_histories[u] = {
                "item_ids": np.array([], dtype=np.int64),
                "behaviors": np.array([], dtype=np.int64),
                "time_deltas": np.array([], dtype=np.float32),
                "timestamps": np.array([], dtype=np.int64),
            }

    @classmethod
    def from_csvs(
        cls,
        csv_paths,
        base_num_days: int = 14,
        max_seq_len: int = 128,
        max_items: int = 20000,
        min_interactions_per_user: int = 5,
    ) -> StreamingDataPlan:
        trace = load_kuairand(
            csv_paths,
            min_interactions_per_user=min_interactions_per_user,
            max_seq_len=max_seq_len,
            max_items=max_items,
        )
        all_dates = sorted(trace.interactions["date"].astype(str).unique())
        base_dates = all_dates[:base_num_days]
        stream_dates = all_dates[base_num_days:]
        return cls(
            trace=trace,
            base_dates=base_dates,
            stream_dates=stream_dates,
            max_seq_len=max_seq_len,
            max_items=max_items,
        )

    def _append_day_to_history(self, u: int, day_df: pd.DataFrame) -> None:
        if u not in self.user_histories:
            return
        items = day_df["item_idx"].to_numpy(dtype=np.int64)
        behs = day_df["behavior"].to_numpy(dtype=np.int64)
        ts = day_df["time_ms"].to_numpy(dtype=np.int64)
        hist = self.user_histories[u]
        if len(hist["timestamps"]) > 0:
            td = (ts - hist["timestamps"][-1]) / 1000.0
            td = np.clip(td, 0.0, 86400.0 * 7.0)
            td = np.full(len(items), td, dtype=np.float32)
            td[0] = max(0.0, (ts[0] - hist["timestamps"][-1]) / 1000.0)
            for i in range(1, len(items)):
                td[i] = max(0.0, (ts[i] - ts[i - 1]) / 1000.0)
        else:
            td = np.zeros(len(items), dtype=np.float32)
            for i in range(1, len(items)):
                td[i] = max(0.0, (ts[i] - ts[i - 1]) / 1000.0)
        hist["item_ids"] = np.concatenate([hist["item_ids"], items])
        hist["behaviors"] = np.concatenate([hist["behaviors"], behs])
        hist["time_deltas"] = np.concatenate([hist["time_deltas"], td])
        hist["timestamps"] = np.concatenate([hist["timestamps"], ts])
        cap = self.max_seq_len * 4
        if len(hist["item_ids"]) > cap:
            for k in hist:
                hist[k] = hist[k][-cap:]

    def init_base(self) -> None:
        """Load all base-period interactions into user histories."""
        for date in self.base_dates:
            day_df = self.daily_segments.get(date)
            if day_df is None:
                continue
            for u, grp in day_df.groupby("user_idx"):
                self._append_day_to_history(int(u), grp)

    def _build_seq(self, u: int, truncate: int | None = None) -> dict:
        hist = self.user_histories.get(u)
        if hist is None or len(hist["item_ids"]) == 0:
            return None
        cap = truncate or self.max_seq_len
        n = len(hist["item_ids"])
        start = max(0, n - cap)
        return {
            "item_ids": hist["item_ids"][start:],
            "behaviors": hist["behaviors"][start:],
            "time_deltas": hist["time_deltas"][start:],
            "user_id": u,
        }

    def get_eval_set(self, date: str, max_users: int | None = None) -> list[dict]:
        """Eval samples for `date`: history up to yesterday + today's positive items.

        Returns list of {history, pos_items} for each active user on `date`.
        Eval must be called BEFORE ingest_day(date).
        """
        day_df = self.daily_segments.get(date)
        if day_df is None:
            return []
        samples = []
        users = day_df["user_idx"].unique()
        if max_users:
            users = users[:max_users]
        for u in users:
            u = int(u)
            seq = self._build_seq(u)
            if seq is None or len(seq["item_ids"]) < 1:
                continue
            pos_items = day_df[day_df["user_idx"] == u]["item_idx"].unique()
            if len(pos_items) == 0:
                continue
            samples.append({"history": seq, "pos_items": pos_items.tolist()})
        return samples

    def ingest_day(self, date: str) -> None:
        """Append `date`'s interactions into user histories (call after eval)."""
        day_df = self.daily_segments.get(date)
        if day_df is None:
            return
        for u, grp in day_df.groupby("user_idx"):
            self._append_day_to_history(int(u), grp)

    def iter_train_batches(self, date: str, batch_size: int = 32) -> Iterator[dict]:
        """Training batches for `date`: each active user's extended sequence.

        After ingest_day, each user's history includes `date`'s interactions.
        We build sequences from the extended history and train with next-item
        prediction (in-batch negatives). One batch = `batch_size` users.
        """
        day_df = self.daily_segments.get(date)
        if day_df is None:
            return
        active_users = [int(u) for u in day_df["user_idx"].unique()]
        np.random.shuffle(active_users)
        for i in range(0, len(active_users), batch_size):
            batch_users = active_users[i : i + batch_size]
            seqs = [self._build_seq(u) for u in batch_users]
            seqs = [s for s in seqs if s is not None and len(s["item_ids"]) >= 2]
            if not seqs:
                continue
            from .kuairand import collate_batch

            yield collate_batch(seqs, max_seq_len=self.max_seq_len, pad_to=self.max_seq_len)

    def iter_base_train_batches(self, batch_size: int = 32, shuffle_users: bool = True) -> Iterator[dict]:
        """Training batches over the full base-period histories (for theta_0)."""
        users = [u for u in self.user_histories if len(self.user_histories[u]["item_ids"]) >= 2]
        if shuffle_users:
            np.random.shuffle(users)
        for i in range(0, len(users), batch_size):
            batch_users = users[i : i + batch_size]
            seqs = [self._build_seq(u) for u in batch_users]
            seqs = [s for s in seqs if s is not None]
            if not seqs:
                continue
            from .kuairand import collate_batch

            yield collate_batch(seqs, max_seq_len=self.max_seq_len, pad_to=self.max_seq_len)

    @property
    def num_items(self) -> int:
        return self.trace.num_items

    @property
    def num_behaviors(self) -> int:
        return self.trace.num_behaviors

    @property
    def num_users(self) -> int:
        return self.trace.num_users
