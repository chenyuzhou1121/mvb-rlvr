"""Define the interface shared by answer verifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from mvb_rlvr.schema import BenchmarkRecord, VerifierVerdict


class Verifier(ABC):
    """Map a benchmark record to a verdict identified by ``name``."""

    name: str

    @abstractmethod
    def verify(self, record: BenchmarkRecord) -> VerifierVerdict:
        """Compare the record's candidate with its reference and return details."""
        raise NotImplementedError
