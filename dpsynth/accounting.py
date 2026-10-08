"""Privacy accounting with zero-concentrated differential privacy (zCDP).

All mechanisms in this package are analysed under rho-zCDP (Bun & Steinke, 2016),
which composes additively and converts to (epsilon, delta)-DP at the end.

Neighbouring datasets: add/remove one record (unbounded DP).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def rho_to_eps_delta(rho: float, delta: float) -> float:
    """Convert rho-zCDP to (epsilon, delta)-DP.

    Uses the standard bound  eps = rho + 2 * sqrt(rho * log(1/delta))
    (Bun & Steinke 2016, Prop. 1.3).
    """
    if rho < 0:
        raise ValueError("rho must be non-negative")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


def eps_delta_to_rho(epsilon: float, delta: float) -> float:
    """Largest rho such that rho-zCDP implies (epsilon, delta)-DP.

    Closed-form inverse of :func:`rho_to_eps_delta`:
        rho = (sqrt(eps + L) - sqrt(L))^2,  L = log(1/delta)
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    log_term = math.log(1.0 / delta)
    return (math.sqrt(epsilon + log_term) - math.sqrt(log_term)) ** 2


def gaussian_sigma(rho: float, l2_sensitivity: float = 1.0) -> float:
    """Noise scale for the Gaussian mechanism to satisfy rho-zCDP.

    Gaussian mechanism with std sigma and L2 sensitivity D is (D^2 / 2 sigma^2)-zCDP.
    """
    if rho <= 0:
        raise ValueError("rho must be positive")
    return l2_sensitivity / math.sqrt(2.0 * rho)


def exponential_mech_eps(rho: float) -> float:
    """Epsilon for the exponential mechanism given a zCDP budget.

    The exponential mechanism with parameter eps is (eps^2 / 8)-zCDP
    (Cesar & Rogers 2021, bounded-range analysis).
    """
    if rho <= 0:
        raise ValueError("rho must be positive")
    return math.sqrt(8.0 * rho)


@dataclass
class PrivacyLedger:
    """Tracks zCDP spending and refuses to exceed the total budget."""

    total_rho: float
    entries: list[tuple[str, float]] = field(default_factory=list)

    @property
    def spent(self) -> float:
        return sum(r for _, r in self.entries)

    @property
    def remaining(self) -> float:
        return self.total_rho - self.spent

    def spend(self, rho: float, label: str) -> float:
        # Tolerance for floating point error when the budget is split evenly.
        if rho > self.remaining + 1e-12:
            raise RuntimeError(
                f"Privacy budget exceeded: tried to spend {rho:.4g} on '{label}', "
                f"only {self.remaining:.4g} remaining"
            )
        self.entries.append((label, rho))
        return rho

    def summary(self, delta: float) -> str:
        lines = [f"{'step':<40} {'rho':>10}"]
        for label, r in self.entries:
            lines.append(f"{label:<40} {r:>10.5f}")
        lines.append(f"{'TOTAL':<40} {self.spent:>10.5f}")
        lines.append(
            f"=> ({rho_to_eps_delta(self.spent, delta):.3f}, {delta:g})-DP"
        )
        return "\n".join(lines)
