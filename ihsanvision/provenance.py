"""Checkpoint SHA-256 identity, bound to an unchanged on-disk file snapshot.

Only trusted operator-selected checkpoints are read. Large checkpoint files are
streamed; bytes and paths are never copied into history or API output.
A cheap (device, inode, size, mtime, ctime) cache avoids rehashing on every
prediction. This is *traceability*, not a signature or authentication of
the checkpoint's training origin, safety or dataset provenance.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import os


@dataclass(frozen=True)
class CheckpointIdentity:
    digest: str
    signature: tuple[int, int, int, int, int]


def _signature(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        stat.st_dev, stat.st_ino, stat.st_size,
        stat.st_mtime_ns, stat.st_ctime_ns,
    )


class CheckpointFingerprinter:
    """For serialized single-process API inference; not a tamper-proof registry."""

    def __init__(self) -> None:
        self._cache: dict[Path, CheckpointIdentity] = {}

    def identify(self, checkpoint: Path) -> CheckpointIdentity:
        path = Path(checkpoint)
        if path.suffix.lower() != ".ckpt" or not path.is_file():
            raise ValueError("Expected a trusted local .ckpt")
        initial = _signature(path.stat())
        previous = self._cache.get(path)
        if previous is not None and previous.signature == initial:
            return previous

        digest = sha256()
        # fstat catches replacement during hashing, while re-stat catches
        # path swaps. Neither assumes a hash from a file name or timestamp.
        with path.open("rb") as file:
            if _signature(os.fstat(file.fileno())) != initial:
                raise RuntimeError("Checkpoint changed before fingerprinting")
            while True:
                chunk = file.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            if _signature(os.fstat(file.fileno())) != initial:
                raise RuntimeError("Checkpoint changed while fingerprinting")
        if _signature(path.stat()) != initial:
            raise RuntimeError("Checkpoint replaced while fingerprinting")

        identity = CheckpointIdentity(digest.hexdigest(), initial)
        self._cache[path] = identity
        return identity

    @staticmethod
    def verify_unchanged(checkpoint: Path, identity: CheckpointIdentity) -> None:
        if _signature(Path(checkpoint).stat()) != identity.signature:
            raise RuntimeError(
                "Checkpoint changed during inference; refusing inaccurate provenance"
            )
