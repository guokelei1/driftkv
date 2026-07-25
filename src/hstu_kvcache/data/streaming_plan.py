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
         c. TRAIN: incremental update on day t's data -> theta_t

    This produces realistic model-version checkpoints and leak-free next-day
    evaluation for cache migration experiments.
    """

    trace: KuaiRandTrace
    base_dates: list[str]
    stream_dates: list[str]
    max_seq_len: int = 128
    max_items: int | None = None
    history_window_days: int | None = None
    user_histories: dict[int, dict] = field(default_factory=dict)
    daily_segments: dict[str, pd.DataFrame] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.history_window_days is not None and self.history_window_days < 1:
            raise ValueError("history_window_days must be positive")
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
                "labels": np.array([], dtype=np.int64),
                "timestamps": np.array([], dtype=np.int64),
            }

    @classmethod
    def from_csvs(
        cls,
        csv_paths,
        base_num_days: int = 14,
        max_seq_len: int = 128,
        max_items: int = 20000,
        max_users: int | None = None,
        min_interactions_per_user: int = 5,
        fit_vocabulary_on_base: bool = False,
        context_hash_buckets: int = 0,
        history_window_days: int | None = None,
        total_num_days: int | None = None,
    ) -> StreamingDataPlan:
        trace = load_kuairand(
            csv_paths,
            min_interactions_per_user=min_interactions_per_user,
            max_seq_len=max_seq_len,
            max_items=max_items,
            max_users=max_users,
            fit_num_days=base_num_days if fit_vocabulary_on_base else None,
            context_hash_buckets=context_hash_buckets,
        )
        all_dates = sorted(trace.interactions["date"].astype(str).unique())
        if total_num_days is not None:
            if total_num_days <= base_num_days:
                raise ValueError("total_num_days must exceed base_num_days")
            if len(all_dates) < total_num_days:
                raise ValueError(
                    f"requested {total_num_days} dates but trace contains {len(all_dates)}"
                )
            all_dates = all_dates[:total_num_days]
            trace.interactions = trace.interactions[
                trace.interactions["date"].astype(str).isin(all_dates)
            ].reset_index(drop=True)
        base_dates = all_dates[:base_num_days]
        stream_dates = all_dates[base_num_days:]
        return cls(
            trace=trace,
            base_dates=base_dates,
            stream_dates=stream_dates,
            max_seq_len=max_seq_len,
            max_items=max_items,
            history_window_days=history_window_days,
        )

    def _append_day_to_history(self, u: int, day_df: pd.DataFrame) -> None:
        if u not in self.user_histories:
            return
        items = day_df["item_idx"].to_numpy(dtype=np.int64)
        behs = day_df["behavior"].to_numpy(dtype=np.int64)
        labels = day_df["label"].to_numpy(dtype=np.int64)
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
        hist["labels"] = np.concatenate([hist["labels"], labels])
        hist["timestamps"] = np.concatenate([hist["timestamps"], ts])
        if self.history_window_days is None:
            cap = self.max_seq_len * 4
            if len(hist["item_ids"]) > cap:
                for k in hist:
                    hist[k] = hist[k][-cap:]
        elif len(hist["timestamps"]) > 0:
            cutoff = hist["timestamps"][-1] - self.history_window_days * 86400 * 1000
            start = int(np.searchsorted(hist["timestamps"], cutoff, side="left"))
            if start > 0:
                for k in hist:
                    hist[k] = hist[k][start:]

    def init_base(self) -> None:
        """Load all base-period interactions into user histories."""
        for date in self.base_dates:
            day_df = self.daily_segments.get(date)
            if day_df is None:
                continue
            for u, grp in day_df.groupby("user_idx"):
                self._append_day_to_history(int(u), grp)

    def _build_seq(
        self,
        u: int,
        truncate: int | None = None,
        as_of_timestamp: int | None = None,
    ) -> dict:
        hist = self.user_histories.get(u)
        if hist is None or len(hist["item_ids"]) == 0:
            return None
        cap = truncate or self.max_seq_len
        n = len(hist["item_ids"])
        window_start = 0
        if self.history_window_days is not None and as_of_timestamp is not None:
            cutoff = as_of_timestamp - self.history_window_days * 86400 * 1000
            window_start = int(
                np.searchsorted(hist["timestamps"], cutoff, side="left")
            )
        available_length = n - window_start
        start = max(window_start, n - cap)
        if start >= n:
            return None
        return {
            "item_ids": hist["item_ids"][start:],
            "behaviors": hist["behaviors"][start:],
            "time_deltas": hist["time_deltas"][start:],
            "labels": hist["labels"][start:],
            "timestamps": hist["timestamps"][start:],
            "user_id": u,
            "available_length_before_token_cap": available_length,
            "token_truncated": available_length > cap,
        }

    def _frame_sequence(self, u: int, frame: pd.DataFrame) -> dict:
        frame = frame.sort_values("time_ms")
        timestamps = frame["time_ms"].to_numpy(dtype=np.int64)
        if "time_delta" in frame:
            time_deltas = frame["time_delta"].to_numpy(dtype=np.float32)
        else:
            time_deltas = np.zeros(len(frame), dtype=np.float32)
            if len(frame) > 1:
                time_deltas[1:] = np.diff(timestamps).clip(0, 86400 * 7 * 1000) / 1000.0
        return {
            "item_ids": frame["item_idx"].to_numpy(dtype=np.int64),
            "behaviors": frame["behavior"].to_numpy(dtype=np.int64),
            "time_deltas": time_deltas,
            "labels": frame["label"].to_numpy(dtype=np.int64),
            "timestamps": timestamps,
            "user_id": u,
        }

    def _chunk_sequence(self, sequence: dict) -> list[dict]:
        length = len(sequence["item_ids"])
        if length < 2:
            return []
        stride = max(1, self.max_seq_len - 1)
        output = []
        for start in range(0, length - 1, stride):
            end = min(length, start + self.max_seq_len)
            chunk = {
                name: values[start:end] if isinstance(values, np.ndarray) else values
                for name, values in sequence.items()
            }
            if len(chunk["item_ids"]) >= 2:
                output.append(chunk)
        return output

    def get_eval_set(self, date: str, max_users: int | None = None) -> list[dict]:
        """Eval samples for `date`: history up to yesterday + today's positive items.

        Returns list of {history, pos_items} for each active user on `date`.
        Eval must be called BEFORE ingest_day(date).
        """
        day_df = self.daily_segments.get(date)
        if day_df is None:
            return []
        samples = []
        for u, user_day in day_df.groupby("user_idx", sort=False):
            u = int(u)
            seq = self._build_seq(
                u,
                as_of_timestamp=int(user_day["time_ms"].min()),
            )
            if seq is None or len(seq["item_ids"]) < 1:
                continue
            pos_items = user_day.loc[user_day["label"] > 0, "item_idx"].unique()
            if len(pos_items) == 0:
                continue
            samples.append({"history": seq, "pos_items": pos_items.tolist()})
            if max_users is not None and len(samples) >= max_users:
                break
        return samples

    def ingest_day(self, date: str) -> None:
        """Append `date`'s interactions into user histories (call after eval)."""
        day_df = self.daily_segments.get(date)
        if day_df is None:
            return
        for u, grp in day_df.groupby("user_idx"):
            self._append_day_to_history(int(u), grp)

    def iter_train_batches(
        self,
        date: str,
        batch_size: int = 32,
        all_chunks: bool = False,
        bucket_by_length: bool = False,
        pad_to_max_seq_len: bool = True,
    ) -> Iterator[dict]:
        """Training batches for `date`: each active user's extended sequence.

        After ingest_day, each user's history includes `date`'s interactions.
        We build sequences from the extended history and train with next-item
        prediction (in-batch negatives). One batch = `batch_size` users.
        """
        day_df = self.daily_segments.get(date)
        if day_df is None:
            return
        sequences = []
        for value, user_day in day_df.groupby("user_idx", sort=False):
            u = int(value)
            history = self.user_histories.get(u)
            truncate = None if not all_chunks or history is None else len(history["item_ids"])
            seq = self._build_seq(u, truncate=truncate)
            if seq is None:
                continue
            timestamps = user_day["time_ms"].to_numpy()
            seq["train_mask"] = np.isin(seq["timestamps"], timestamps)
            candidates = self._chunk_sequence(seq) if all_chunks else [seq]
            if all_chunks:
                candidates = [
                    chunk
                    for chunk in candidates
                    if np.any(chunk["train_mask"][1:] & (chunk["labels"][1:] > 0))
                ]
            sequences.extend(candidates)
        np.random.shuffle(sequences)
        if bucket_by_length:
            sequences.sort(key=lambda sequence: len(sequence["item_ids"]))
        grouped = [
            sequences[i : i + batch_size]
            for i in range(0, len(sequences), batch_size)
        ]
        if bucket_by_length:
            np.random.shuffle(grouped)
        for seqs in grouped:
            if not seqs:
                continue
            from .kuairand import collate_batch

            yield collate_batch(
                seqs,
                max_seq_len=self.max_seq_len,
                pad_to=self.max_seq_len if pad_to_max_seq_len else None,
            )

    def iter_base_train_batches(
        self,
        batch_size: int = 32,
        shuffle_users: bool = True,
        all_chunks: bool = False,
        bucket_by_length: bool = False,
        pad_to_max_seq_len: bool = True,
    ) -> Iterator[dict]:
        """Training batches over the full base-period histories (for theta_0)."""
        if all_chunks:
            frames = [self.daily_segments[date] for date in self.base_dates if date in self.daily_segments]
            base = pd.concat(frames, ignore_index=True)
            sequences = []
            for value, frame in base.groupby("user_idx"):
                sequence = self._frame_sequence(int(value), frame)
                sequences.extend(self._chunk_sequence(sequence))
        else:
            sequences = [
                self._build_seq(u)
                for u in self.user_histories
                if len(self.user_histories[u]["item_ids"]) >= 2
            ]
            sequences = [sequence for sequence in sequences if sequence is not None]
        if shuffle_users:
            np.random.shuffle(sequences)
        if bucket_by_length:
            sequences.sort(key=lambda sequence: len(sequence["item_ids"]))
        for sequence in sequences:
            sequence["train_mask"] = np.ones(len(sequence["item_ids"]), dtype=np.bool_)
        grouped = [
            sequences[i : i + batch_size]
            for i in range(0, len(sequences), batch_size)
        ]
        if bucket_by_length and shuffle_users:
            np.random.shuffle(grouped)
        for seqs in grouped:
            if not seqs:
                continue
            from .kuairand import collate_batch

            yield collate_batch(
                seqs,
                max_seq_len=self.max_seq_len,
                pad_to=self.max_seq_len if pad_to_max_seq_len else None,
            )

    @property
    def num_items(self) -> int:
        return self.trace.num_items

    @property
    def num_prediction_items(self) -> int:
        return self.trace.num_prediction_items

    @property
    def num_behaviors(self) -> int:
        return self.trace.num_behaviors

    @property
    def num_users(self) -> int:
        return self.trace.num_users
