"""Label-free population selection primitives for the Yambda scale matrix."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable


UID_SELECTOR_NAMESPACE = "evokv:yambda500m:medium:v1"


def uid_selector_digest(
    uid: int, *, namespace: str = UID_SELECTOR_NAMESPACE
) -> str:
    """Return the stable selector digest for one raw UID.

    Decimal ASCII is used deliberately so the selector is independent of host
    byte order and integer width.
    """

    if int(uid) < 0:
        raise ValueError("uid must be non-negative")
    material = f"{namespace}:{int(uid)}".encode("ascii")
    return hashlib.sha256(material).hexdigest()


def select_medium_uids(
    eligible_uids: Iterable[int],
    *,
    count: int,
    namespace: str = UID_SELECTOR_NAMESPACE,
) -> list[int]:
    """Select the first ``count`` eligible users by stable digest then UID."""

    unique = sorted({int(uid) for uid in eligible_uids})
    if count < 1:
        raise ValueError("count must be positive")
    if count > len(unique):
        raise ValueError("count exceeds the eligible population")
    ranked = sorted(unique, key=lambda uid: (uid_selector_digest(uid, namespace=namespace), uid))
    return ranked[:count]
