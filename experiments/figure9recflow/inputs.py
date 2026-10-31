"""Read-only, per-user RecFlow inputs for independent release-edge evaluation.

Fit users are a fixed permutation of the original cache_fit role, never chosen
by future activity. Quality/cost users are the original train role, including
inactive users. All valid role-2 requests and their full candidate sets survive;
unknown (-1) labels are not negatives and zero-positive requests are retained.

The published source initializes a cache at evaluation-day 00:00 Shanghai time.
The caller repairs it once, before the first query OR behavior append. All
queries at timestamp t precede ALL behavior events at t. Recorded feedback is
assumed available after its timestamp block: the dataset supplies no actual
feedback arrival delay. This is the existing timestamp-visibility assumption,
not proof that delayed real-world labels would already be available.

No model loading, CUDA, file mutation, or file-fingerprint checks occur here.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator

import numpy as np

from experiments.recflow_full_population.code.protocol import day_bounds_ms


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "experiments/recflow_full_population/data"
DEFAULT_CHAIN = ROOT / "experiments/recflow_continuous_chain/chain_manifest.json"
SEED = 20261030
HISTORY_LIMIT = 1024
LABEL_VISIBILITY = (
    "All requests at t see only history events with timestamp < t; recorded "
    "behavior/feedback at t becomes visible after every request at t. Actual "
    "feedback arrival delays are unavailable. Unknown candidate labels remain -1."
)


def _load(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Required RecFlow input is missing: {path}")
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _permutation(values, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int32)
    if len(np.unique(values)) != len(values):
        raise ValueError("Population role contains duplicate users")
    # The original population file fixes the pre-outcome starting order.
    return np.random.default_rng(seed).permutation(values)


def event_deltas(events: np.ndarray, previous_ts: int | None = None) -> np.ndarray:
    """Seconds since the actual preceding event; only a fresh prefix starts at 0."""
    result = np.zeros(len(events), np.float32)
    if len(events):
        times = events["ts"]
        if len(times) > 1:
            differences = np.diff(times)
            if np.any(differences < 0):
                raise ValueError("History events are not chronological")
            result[1:] = differences / 1000.0
        if previous_ts is not None:
            if times[0] < previous_ts:
                raise ValueError("Append precedes its previous history event")
            result[0] = (int(times[0]) - previous_ts) / 1000.0
    return result


@dataclass(frozen=True)
class ReleaseEdge:
    id: str
    source: str
    target: str
    evaluation_day: int
    source_checkpoint: Path
    target_checkpoint: Path
    evidence: dict

    @property
    def interval_ms(self) -> tuple[int, int]:
        return day_bounds_ms(self.evaluation_day)


@dataclass(frozen=True)
class HistorySnapshot:
    uid: int
    events: np.ndarray
    deltas: np.ndarray

    @property
    def features(self):
        return self.events["features"]

    @property
    def behaviors(self):
        return self.events["behavior"].astype(np.int64) + 1


@dataclass(frozen=True)
class RequestPayload:
    uid: int
    rid: int
    ts: int
    features: np.ndarray
    labels: np.ndarray
    history_length: int
    query_delta: float

    @property
    def candidate_features(self):
        return self.features

    @property
    def candidate_ids(self):
        return self.features[:, 0]


@dataclass(frozen=True)
class TimelineBlock:
    kind: str
    ts: int
    requests: tuple[RequestPayload, ...] = ()
    events: np.ndarray | None = None
    deltas: np.ndarray | None = None


@dataclass(frozen=True)
class UserPayload:
    """Bounded user data, safe to send from CPU prefetch workers.

    Arrays are mmap views in the producing process. Pickling this payload
    transports only its user slices, not an entire daily candidate catalogue.
    A DayInputs object itself reopens mmaps after pickling instead of copying.
    """
    uid: int
    day: int
    initial_history: np.ndarray
    initial_deltas: np.ndarray
    events: np.ndarray
    event_deltas: np.ndarray
    requests: tuple[RequestPayload, ...]

    def timeline(self) -> Iterator[TimelineBlock]:
        """Merge the full day, including maintenance after the final request."""
        i = j = 0
        while i < len(self.requests) or j < len(self.events):
            qt = self.requests[i].ts if i < len(self.requests) else None
            et = int(self.events["ts"][j]) if j < len(self.events) else None
            if qt is not None and (et is None or qt <= et):
                stop = i + 1
                while stop < len(self.requests) and self.requests[stop].ts == qt:
                    stop += 1
                yield TimelineBlock("request", qt, self.requests[i:stop])
                i = stop
            else:
                stop = int(np.searchsorted(self.events["ts"], et, side="right"))
                yield TimelineBlock("append", et, events=self.events[j:stop],
                                    deltas=self.event_deltas[j:stop])
                j = stop


class RecFlowInputs:
    """Population, accepted published endpoints, and lazy worker-local history."""

    def __init__(self, data=DEFAULT_DATA, chain_manifest=DEFAULT_CHAIN, *,
                 fit_users: int = 256, seed: int = SEED):
        self.data = Path(data).resolve()
        self.chain_manifest = Path(chain_manifest).resolve()
        self.seed = int(seed)
        with np.load(self.data / "population.npz", allow_pickle=False) as population:
            groups = {key: _permutation(population[key], self.seed) for key in
                      ("train", "validation", "cache_fit", "cache_validation")}
        if not 0 < fit_users <= len(groups["cache_fit"]):
            raise ValueError("Requested fit size exceeds the original cache_fit role")
        for i, name in enumerate(groups):
            for other in list(groups)[i + 1:]:
                if np.intersect1d(groups[name], groups[other]).size:
                    raise ValueError(f"Population roles overlap: {name}, {other}")
        self.eval_uids = groups["train"]
        self.fit_uids = groups["cache_fit"][:fit_users]
        self.cache_validation_uids = groups["cache_validation"]
        self.validation_uids = groups["validation"]
        self._events = self._offsets = None
        for name in ("events.npy", "user_offsets.npy"):
            if not (self.data / "history" / name).is_file():
                raise FileNotFoundError(self.data / "history" / name)
        manifest = json.loads(self.chain_manifest.read_text())
        if manifest.get("history_limit") != HISTORY_LIMIT:
            raise ValueError("The accepted chain must use history limit 1024")
        nodes = {node["id"]: node for node in manifest["nodes"]}
        edges = []
        for edge in manifest["edges"]:
            source, target = nodes[edge["source"]], nodes[edge["target"]]
            # Preferred future placeholders and rejected candidates are not models.
            if not source.get("checkpoint") or not target.get("checkpoint"):
                continue
            if not edge.get("evidence"):
                raise ValueError(f"Actual edge lacks recorded evidence: {edge['id']}")
            paths = [Path(node["checkpoint"]).resolve() for node in (source, target)]
            for path in paths:
                if not path.is_file():
                    raise FileNotFoundError(f"Accepted checkpoint missing: {path}")
            if edges and (edges[-1].target != edge["source"] or
                          edges[-1].evaluation_day >= edge["evaluation_day"]):
                raise ValueError("Accepted edges are not a chronological published chain")
            edges.append(ReleaseEdge(edge["id"], edge["source"], edge["target"],
                                     int(edge["evaluation_day"]), *paths, edge["evidence"]))
        self.edges = tuple(edges)
        if not self.edges:
            raise ValueError("No actual accepted release edges found")

    def __getstate__(self):
        return dict(self.__dict__, _events=None, _offsets=None)

    def _history(self):
        if self._events is None:
            self._events = _load(self.data / "history/events.npy")
            self._offsets = _load(self.data / "history/user_offsets.npy")
        return self._events, self._offsets

    def _user_events(self, uid: int):
        events, offsets = self._history()
        if not 0 <= uid < len(offsets) - 1:
            raise ValueError(f"UID outside actual history index: {uid}")
        start, stop = int(offsets[uid]), int(offsets[uid + 1])
        return events[start:stop], start

    def history(self, uid: int, cutoff_ms: int) -> HistorySnapshot:
        events, _ = self._user_events(int(uid))
        stop = int(np.searchsorted(events["ts"], cutoff_ms, side="left"))
        prefix = events[max(0, stop - HISTORY_LIMIT):stop]
        return HistorySnapshot(int(uid), prefix, event_deltas(prefix))

    def fit_snapshots(self, day: int) -> list[HistorySnapshot]:
        """Fixed-role fit order; retain empty histories instead of substituting users."""
        start, _ = day_bounds_ms(day)
        return [self.history(int(uid), start) for uid in self.fit_uids]

    def day(self, day: int, *, limit: int | None = None, offset: int = 0,
            uids=None) -> "DayInputs":
        if uids is None:
            if offset < 0 or (limit is not None and limit < 0):
                raise ValueError("Panel offset/limit must be nonnegative")
            uids = self.eval_uids[offset:None if limit is None else offset + limit]
        elif limit is not None or offset:
            raise ValueError("Use either explicit UIDs or a fixed population prefix")
        return DayInputs(self, int(day), uids)


class DayInputs:
    """Map-style CPU dataset: one item is one user, including inactive users.

    Compatible with a DataLoader using batch_size=None or an identity collator.
    Worker processes lazily reopen mmaps; candidate arrays are contiguous slices.
    """

    def __init__(self, inputs: RecFlowInputs, day: int, uids):
        self.inputs, self.day = inputs, day
        self.uids = np.asarray(uids, dtype=np.int32).copy()
        if len(np.unique(self.uids)) != len(self.uids) or not np.isin(
                self.uids, inputs.eval_uids).all():
            raise ValueError("Quality/cost panel must be distinct original train users")
        self.start_ms, self.end_ms = day_bounds_ms(day)
        self.selection_path = inputs.data / f"selections/role2_d{day}_d{day}.npy"
        self.stage = inputs.data / f"stage_v2/d{day:02d}"
        for path in (self.selection_path, self.stage / "features.npy",
                     self.stage / "labels.npy", self.stage / "index.npy"):
            if not path.is_file():
                raise FileNotFoundError(f"Required complete-day input missing: {path}")
        self._records = self._features = self._labels = None
        self._order = self._request_offsets = None

    def __getstate__(self):
        return dict(self.__dict__, _records=None, _features=None, _labels=None,
                    _order=None, _request_offsets=None)

    def _open(self):
        if self._records is None:
            self._records = _load(self.selection_path)
            self._features = _load(self.stage / "features.npy")
            self._labels = _load(self.stage / "labels.npy")
            r = self._records
            self._order = np.lexsort((r["rid"], r["ts"], r["uid"]))
            _, history_offsets = self.inputs._history()
            counts = np.bincount(r["uid"], minlength=len(history_offsets) - 1)
            self._request_offsets = np.r_[0, counts.cumsum()]

    @property
    def records(self):
        self._open()
        return self._records

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, index: int) -> UserPayload:
        self._open()
        uid = int(self.uids[index])
        history, base = self.inputs._user_events(uid)
        times = history["ts"]
        begin, end = np.searchsorted(times, [self.start_ms, self.end_ms], side="left")
        begin, end = int(begin), int(end)
        initial = history[max(0, begin - HISTORY_LIMIT):begin]
        events = history[begin:end]
        previous = int(times[begin - 1]) if begin else None
        lo, hi = self._request_offsets[uid:uid + 2]
        records = self._records[self._order[lo:hi]]
        requests = []
        for record in records:
            ts = int(record["ts"])
            stop = int(np.searchsorted(times, ts, side="left"))
            length = min(HISTORY_LIMIT, stop)
            if (not self.start_ms <= ts < self.end_ms or
                    int(record["history_stop"]) != base + stop or
                    int(record["history_length"]) != length):
                raise ValueError(f"Causal request/history mismatch: UID {uid}, RID {record['rid']}")
            start, finish = int(record["start"]), int(record["stop"])
            if not 0 <= start < finish <= len(self._labels):
                raise ValueError("Invalid full-candidate request bounds")
            requests.append(RequestPayload(uid, int(record["rid"]), ts,
                self._features[start:finish], self._labels[start:finish], length,
                (ts - int(times[stop - 1])) / 1000.0 if stop else 0.0))
        return UserPayload(uid, self.day, initial, event_deltas(initial), events,
                           event_deltas(events, previous), tuple(requests))

    def coverage(self) -> dict:
        """CPU preparation audit; all selected requests and full-day maintenance.

        Does not scan/copy the huge candidate matrices. Counts and inclusion are
        checked against the actual daily index and role-2 ledger; candidate
        feature/label values retain the original ingestion audit's semantics.
        """
        self._open()
        r = self._records
        original = _load(self.stage / "index.npy")
        expected = original[np.isin(original["uid"], self.inputs.eval_uids) &
                            original["valid_request"]]
        if expected.dtype != r.dtype or not np.array_equal(expected, r):
            raise ValueError("Role-2 selection does not contain every valid train-user request")
        manifest = json.loads((self.inputs.data / "selection_manifest.json").read_text())
        entry = manifest["datasets"][f"2:{self.day}"]
        if len(r) != entry["requests"]:
            raise ValueError("Complete-day request ledger count disagrees")
        if len(np.unique(r["rid"])) != len(r):
            raise ValueError("Duplicate request IDs in complete-day selection")
        if (np.any(r["day"] != self.day) or
                np.any((r["ts"] < self.start_ms) | (r["ts"] >= self.end_ms))):
            raise ValueError("Request date falls outside its complete-day window")
        if len(self._features) != len(self._labels):
            raise ValueError("Candidate feature/label row counts differ")
        selected = r[np.isin(r["uid"], self.uids)]
        if len(selected) and (np.any(selected["start"] < 0) or
                np.any(selected["stop"] <= selected["start"]) or
                np.any(selected["stop"] > len(self._labels))):
            raise ValueError("Invalid candidate ranges")
        counts = {"initial_history_rows": 0, "empty_initial_history_users": 0,
                  "full_initial_history_users": 0, "appended_history_events": 0,
                  "behavior_active_users": 0, "first_consumer_users": 0,
                  "once_repair_history_rows": 0, "maintenance_after_last_request_events": 0}
        for uid in self.uids:
            events, _ = self.inputs._user_events(int(uid))
            times = events["ts"]
            begin, end = np.searchsorted(times, [self.start_ms, self.end_ms], side="left")
            n = min(HISTORY_LIMIT, int(begin))
            lo, hi = self._request_offsets[uid:uid + 2]
            counts["initial_history_rows"] += n
            counts["empty_initial_history_users"] += int(n == 0)
            counts["full_initial_history_users"] += int(n == HISTORY_LIMIT)
            counts["appended_history_events"] += int(end - begin)
            counts["behavior_active_users"] += int(end > begin)
            consumes = hi > lo or end > begin
            counts["first_consumer_users"] += int(consumes)
            counts["once_repair_history_rows"] += n if consumes else 0
            last = int(r["ts"][self._order[hi - 1]]) if hi > lo else self.start_ms
            counts["maintenance_after_last_request_events"] += int(
                end - np.searchsorted(times, last, side="left"))
        fit = self.inputs.fit_snapshots(self.day)
        active = len(np.unique(selected["uid"]))
        return dict(status="passed", day=self.day,
            interval_ms_half_open=[self.start_ms, self.end_ms],
            population_seed=self.inputs.seed, population_selection="fixed role permutation; no future filtering",
            quality_users=len(self.uids), cost_users=len(self.uids), request_active_users=active,
            request_inactive_users=len(self.uids) - active, requests=len(selected),
            positive_requests=int((selected["positive"] > 0).sum()),
            zero_positive_requests=int((selected["positive"] == 0).sum()),
            zero_observed_requests=int((selected["observed"] == 0).sum()),
            candidates=int((selected["stop"] - selected["start"]).sum()),
            observed_candidates=int(selected["observed"].sum()),
            positive_candidates=int(selected["positive"].sum()),
            excluded_invalid_requests=int((np.isin(original["uid"], self.uids) &
                                           ~original["valid_request"]).sum()),
            full_train_requests=len(r), selection_file=str(self.selection_path),
            fit_role="cache_fit", fit_users=len(fit), fit_history_rows=sum(len(x.events) for x in fit),
            fit_empty_history_users=sum(not len(x.events) for x in fit),
            fit_quality_user_overlap=0, history_limit=HISTORY_LIMIT,
            label_visibility_assumption=LABEL_VISIBILITY,
            candidate_values_audit="original stage_v2 ingestion; this audit checks index coverage/ranges",
            **counts)
