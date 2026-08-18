"""Deterministic paired latency samples for LIBERO evaluation."""

import dataclasses
import hashlib
import json
import math
import numbers


SAMPLER_VERSION = "sha256_box_muller_v1"
NEGATIVE_POLICY = "deterministic_resample"
DEFAULT_MEAN_NS = 300_000_000
DEFAULT_STDDEV_NS = 60_000_000
DEFAULT_SEED = 42
SAMPLE_KEY_FIELDS = (
    "namespace",
    "seed",
    "suite",
    "task_id",
    "trial_index",
    "request_ordinal",
)


def _require_nonnegative_integer(value, *, label):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def _require_nonempty_string(value, *, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


@dataclasses.dataclass(frozen=True)
class LatencySampleKeyV1:
    namespace: str
    seed: int
    suite: str
    task_id: int
    trial_index: int
    request_ordinal: int

    def __post_init__(self):
        object.__setattr__(
            self,
            "namespace",
            _require_nonempty_string(self.namespace, label="namespace"),
        )
        object.__setattr__(self, "seed", _require_nonnegative_integer(self.seed, label="seed"))
        object.__setattr__(self, "suite", _require_nonempty_string(self.suite, label="suite"))
        object.__setattr__(
            self,
            "task_id",
            _require_nonnegative_integer(self.task_id, label="task_id"),
        )
        object.__setattr__(
            self,
            "trial_index",
            _require_nonnegative_integer(self.trial_index, label="trial_index"),
        )
        object.__setattr__(
            self,
            "request_ordinal",
            _require_nonnegative_integer(self.request_ordinal, label="request_ordinal"),
        )

    def canonical_values(self):
        return (
            self.namespace,
            self.seed,
            self.suite,
            self.task_id,
            self.trial_index,
            self.request_ordinal,
        )

    def to_dict(self):
        return dict(zip(SAMPLE_KEY_FIELDS, self.canonical_values()))

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) != set(SAMPLE_KEY_FIELDS):
            raise ValueError("latency sample key must be an exact JSON object")
        return cls(**value)


class NormalLatencySamplerV1:
    """Versioned SHA-256 + Box-Muller normal sampler with no mutable RNG state."""

    def __init__(self, *, mean_ns=DEFAULT_MEAN_NS, stddev_ns=DEFAULT_STDDEV_NS):
        self._mean_ns = _require_nonnegative_integer(mean_ns, label="mean_ns")
        self._stddev_ns = _require_nonnegative_integer(stddev_ns, label="stddev_ns")
        if self._stddev_ns == 0:
            raise ValueError("stddev_ns must be positive")

    @property
    def mean_ns(self):
        return self._mean_ns

    @property
    def stddev_ns(self):
        return self._stddev_ns

    @property
    def sampler_version(self):
        return SAMPLER_VERSION

    @property
    def negative_policy(self):
        return NEGATIVE_POLICY

    def sample_target_ns(self, key):
        if not isinstance(key, LatencySampleKeyV1):
            raise ValueError("key must be a LatencySampleKeyV1")
        attempt = 0
        while True:
            payload = json.dumps(
                [SAMPLER_VERSION, *key.canonical_values(), attempt],
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
            digest = hashlib.sha256(payload).digest()
            denominator = float(1 << 64)
            uniform_one = (int.from_bytes(digest[:8], "big") + 0.5) / denominator
            uniform_two = (int.from_bytes(digest[8:16], "big") + 0.5) / denominator
            standard_normal = math.sqrt(-2.0 * math.log(uniform_one)) * math.cos(2.0 * math.pi * uniform_two)
            target_ns = math.floor(self._mean_ns + self._stddev_ns * standard_normal + 0.5)
            if target_ns >= 0:
                return target_ns
            attempt += 1
