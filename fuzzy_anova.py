"""
Fuzzy ANOVA (Analysis of Variance for Triangular Fuzzy Data)
============================================================

This module provides a comprehensive implementation of One-Way Analysis of
Variance (ANOVA) for triangular fuzzy observations, based on the statistical
framework introduced by Hsien-Chung Wu (2007):
    "Analysis of variance for fuzzy data", International Journal of Systems Science,
    38(3), 235-246.

Each observation is represented as a Triangular Fuzzy Number (TFN):
    y_ij = (L_ij, Core_ij, U_ij)
where:
    - L_ij: Lower bound of support (membership degree = 0)
    - Core_ij: Modal/Core value (membership degree = 1)
    - U_ij: Upper bound of support (membership degree = 0)
    - Condition: L_ij <= Core_ij <= U_ij

Key Components:
---------------
1. Alpha-Cut (h-level set) Method:
   - Evaluates lower and upper bound real-valued representations for any h in [0, 1]:
     (Y_ij)^L_h = (1 - h) * L_ij + h * Core_ij
     (Y_ij)^U_h = (1 - h) * U_ij + h * Core_ij
   - Generates h-level sense ANOVA tables for lower and upper bound cases (Table 2 & Table 3).

2. Decision Rules with Optimistic and Pessimistic Degrees:
   - Formulates quadratic constraint optimization problems on h in [0, 1].
   - Computes interval sets K_{H0} and K_{H1}.
   - Calculates the Optimistic degree lambda(K) as the interval measure / length.
   - Calculates the Pessimistic degree h_{H} as the supremum of the opposing set.
   - Derives overall Confidence degree = min(optimistic, 1 - pessimistic).
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Harshit Trivedi, Dr. Masum Raj"

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union
import math
import numpy as np

# Robust F-distribution import with pure Python fallback if scipy is blocked/unavailable
try:
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except Exception:
    _scipy_stats = None
    _HAS_SCIPY = False


def _betacf(a: float, b: float, x: float, max_iter: int = 200, eps: float = 1e-14) -> float:
    """Continued fraction for regularized incomplete beta function (Lentz's method)."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        # Even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        # Odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        del_h = d * c
        h *= del_h
        if abs(del_h - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    bt = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    else:
        return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _f_cdf(f: float, df1: int, df2: int) -> float:
    """Cumulative distribution function (CDF) for Snedecor's F-distribution."""
    if f <= 0.0:
        return 0.0
    if _HAS_SCIPY and _scipy_stats is not None:
        try:
            return float(_scipy_stats.f.cdf(f, df1, df2))
        except Exception:
            pass
    x = (df1 * f) / (df1 * f + df2)
    return _betainc(df1 / 2.0, df2 / 2.0, x)


def _f_ppf(p: float, df1: int, df2: int, tol: float = 1e-10) -> float:
    """Percent point function (inverse CDF) for Snedecor's F-distribution."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return float("inf")
    if _HAS_SCIPY and _scipy_stats is not None:
        try:
            return float(_scipy_stats.f.ppf(p, df1, df2))
        except Exception:
            pass
    low, high = 0.0, 1.0
    while _f_cdf(high, df1, df2) < p:
        low = high
        high *= 2.0
        if high > 1e12:
            break
    for _ in range(100):
        mid = 0.5 * (low + high)
        val = _f_cdf(mid, df1, df2)
        if abs(val - p) < tol or (high - low) < tol:
            return mid
        if val < p:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


__all__ = [
    "fuzzy_anova",
    "get_alpha_cut",
    "alpha_cut_anova",
    "FuzzyANOVAResult",
    "AlphaCutANOVAResult",
    "AlphaCutTable",
    "QuadraticCoefficients",
    "main",
    "__version__",
    "__author__",
]

# Type aliases
FuzzyObservation = Tuple[float, float, float]
GroupInput = Union[
    Mapping[str, Sequence[Sequence[float]]],
    Sequence[Sequence[Sequence[float]]],
]


@dataclass(frozen=True)
class QuadraticCoefficients:
    """
    Quadratic equation coefficients A*h^2 + B*h + C for level-cut analysis.

    Supports tuple unpacking:
        A, B, C = result.lower_coeffs
    """
    A: float
    B: float
    C: float

    def __iter__(self) -> Iterator[float]:
        """Allows unpacking: A, B, C = coeffs."""
        yield self.A
        yield self.B
        yield self.C

    def evaluate(self, h: float) -> float:
        """Evaluates A*h^2 + B*h + C at a given alpha/level cut h."""
        return self.A * (h ** 2) + self.B * h + self.C

    def as_tuple(self) -> Tuple[float, float, float]:
        """Returns the coefficients as a tuple (A, B, C)."""
        return (self.A, self.B, self.C)

    def roots(self, tol: float = 1e-12) -> List[float]:
        """Finds real roots of A*h^2 + B*h + C = 0."""
        A, B, C = self.A, self.B, self.C
        if abs(A) < tol:
            if abs(B) < tol:
                return []
            return [-C / B]
        disc = B ** 2 - 4.0 * A * C
        if disc < -tol:
            return []
        safe_disc = max(0.0, disc)
        sdisc = math.sqrt(safe_disc)
        r1 = (-B - sdisc) / (2.0 * A)
        r2 = (-B + sdisc) / (2.0 * A)
        return sorted([r1, r2])


@dataclass
class AlphaCutTable:
    """
    Represents an h-level sense ANOVA table for either the lower or upper bound case
    (corresponding to Table 2 or Table 3 in Wu 2007).
    """
    bound_name: str  # "lower" or "upper"
    h: float
    sstr: float
    sse: float
    ssto: float
    df_between: int
    df_within: int
    df_total: int
    mstr: float
    mse: float
    f_stat: float
    p_value: float

    def format_table(self) -> str:
        """Returns the formatted ANOVA table for this bound."""
        bound_title = "Lower Bound" if self.bound_name.lower() == "lower" else "Upper Bound"
        lines = [
            f"ANOVA Table: {bound_title} Case (h-level = {self.h:.2f})",
            "-" * 72,
            f"{'Source of Variation':<26} {'SS':>12} {'df':>6} {'MS':>12} {'F':>10}",
            "-" * 72,
            f"{'Between Treatments':<26} {self.sstr:>12.4f} {self.df_between:>6} {self.mstr:>12.4f} {self.f_stat:>10.4f}",
            f"{'Error (within Treatments)':<26} {self.sse:>12.4f} {self.df_within:>6} {self.mse:>12.4f}",
            f"{'Total':<26} {self.ssto:>12.4f} {self.df_total:>6}",
            "-" * 72,
        ]
        return "\n".join(lines)


@dataclass
class AlphaCutANOVAResult:
    """
    Container holding the lower and upper bound ANOVA calculations at a given h-level.
    """
    h: float
    lower: AlphaCutTable
    upper: AlphaCutTable
    f_crit: float

    def summary(self) -> str:
        """Returns the formatted string containing both Table 2 and Table 3."""
        return f"{self.lower.format_table()}\n\n{self.upper.format_table()}"


@dataclass
class FuzzyANOVAResult:
    """
    Structured result container for One-Way Fuzzy ANOVA based on Wu (2007).

    Attributes
    ----------
    decision : str
        Hypothesis test decision: 'H0' (equal group means) or 'H1' (different group means).
    optimistic_degree : float
        Optimistic degree of belief in the decision in [0.0, 1.0].
        Equal to lambda(K_{H0}) if H0 is accepted, or lambda(K_{H1}) if H1 is accepted.
    pessimistic_degree : float
        Pessimistic degree of belief in the decision in [0.0, 1.0].
        Equal to sup(K_{H1}) if H0 is accepted, or sup(K_{H0}) if H1 is accepted.
    confidence_degree : float
        Overall confidence degree = min(optimistic_degree, 1.0 - pessimistic_degree) in [0.0, 1.0].
    sstr : float
        Treatment sum of squares (Between-group SS) computed on core values (h = 1).
    sse : float
        Error sum of squares (Within-group SS) computed on core values (h = 1).
    f_stat : float
        F-statistic calculated on core values: (SSTR / df1) / (SSE / df2).
    f_crit : float
        Critical F value from Snedecor's F-distribution at significance level alpha.
    p_value : float
        P-value of the F-statistic on core values.
    rho : float
        Critical threshold multiplier: (df1 / df2) * F_crit.
    df_between : int
        Degrees of freedom between groups (r - 1).
    df_within : int
        Degrees of freedom within groups (nT - r).
    lower_coeffs : QuadraticCoefficients
        Quadratic coefficients (AL, BL, CL) for the lower fuzzy bound.
    upper_coeffs : QuadraticCoefficients
        Quadratic coefficients (AU, BU, CU) for the upper fuzzy bound.
    alpha_sig : float
        Significance level used for the statistical test.
    k_h0_intervals : List[Tuple[float, float]]
        Disjoint intervals forming set K_{H0} in [0, 1].
    k_h1_intervals : List[Tuple[float, float]]
        Disjoint intervals forming set K_{H1} in [0, 1].
    lambda_k_h0 : float
        Length of set K_{H0}.
    lambda_k_h1 : float
        Length of set K_{H1}.
    sup_k_h0 : float
        Supremum / maximal degree of set K_{H0}.
    sup_k_h1 : float
        Supremum / maximal degree of set K_{H1}.
    alpha_cut_tables : Optional[AlphaCutANOVAResult]
        Calculated h-level ANOVA tables (Tables 2 & 3).
    """
    decision: str
    optimistic_degree: float
    pessimistic_degree: float
    confidence_degree: float
    sstr: float
    sse: float
    f_stat: float
    f_crit: float
    p_value: float
    rho: float
    df_between: int
    df_within: int
    lower_coeffs: QuadraticCoefficients
    upper_coeffs: QuadraticCoefficients
    alpha_sig: float
    k_h0_intervals: List[Tuple[float, float]] = field(default_factory=list)
    k_h1_intervals: List[Tuple[float, float]] = field(default_factory=list)
    lambda_k_h0: float = 0.0
    lambda_k_h1: float = 0.0
    sup_k_h0: float = 0.0
    sup_k_h1: float = 0.0
    alpha_cut_tables: Optional[AlphaCutANOVAResult] = None

    # Backward-compatible property aliases
    @property
    def SSTR(self) -> float:
        return self.sstr

    @property
    def SSE(self) -> float:
        return self.sse

    @property
    def F_crit(self) -> float:
        return self.f_crit

    @property
    def AL(self) -> float:
        return self.lower_coeffs.A

    @property
    def BL(self) -> float:
        return self.lower_coeffs.B

    @property
    def CL(self) -> float:
        return self.lower_coeffs.C

    @property
    def AU(self) -> float:
        return self.upper_coeffs.A

    @property
    def BU(self) -> float:
        return self.upper_coeffs.B

    @property
    def CU(self) -> float:
        return self.upper_coeffs.C

    @property
    def is_null_rejected(self) -> bool:
        """Returns True if the null hypothesis H0 is rejected (i.e. decision is H1)."""
        return self.decision == "H1"

    def summary(self) -> str:
        """
        Returns a formatted summary of the test results matching the structure,
        tables (Table 2 & 3), and conclusion wording from Wu (2007).
        """
        roots_l = self.lower_coeffs.roots()
        roots_u = self.upper_coeffs.roots()
        roots_l_str = ", ".join(f"{r:.4f}" for r in roots_l) if roots_l else "None"
        roots_u_str = ", ".join(f"{r:.4f}" for r in roots_u) if roots_u else "None"

        # Format intervals for display
        def _fmt_intervals(intervals: List[Tuple[float, float]]) -> str:
            if not intervals:
                return "Empty set"
            return " U ".join(f"[{s:.4f}, {e:.4f}]" for s, e in intervals)

        k_h0_str = _fmt_intervals(self.k_h0_intervals)
        k_h1_str = _fmt_intervals(self.k_h1_intervals)

        # Conclusion text matching Wu (2007) Page 15 & Page 17 exactly
        if self.decision == "H1":
            conclusion_text = (
                f"We conclude that we accept the alternative hypothesis H1 at the level of significance "
                f"alpha = {self.alpha_sig:.2f} with optimistic degree {self.optimistic_degree:.4f} and "
                f"pessimistic degree {self.pessimistic_degree:.4f}. Now the confidence degree is "
                f"min{{{self.optimistic_degree:.4f}, 1 - {self.pessimistic_degree:.4f}}} = {self.confidence_degree:.4f}. "
                f"Therefore we can also say that we accept the alternative hypothesis H1 at the level of "
                f"significance alpha = {self.alpha_sig:.2f} with confidence degree {self.confidence_degree:.4f}."
            )
            fact_text = (
                f'Since the confidence degree is {"high" if self.confidence_degree >= 0.5 else "moderate"}, '
                f'we have a {"strong" if self.confidence_degree >= 0.7 else "moderate"} confidence to accept the following fact:\n'
                f'    "we accept the alternative hypothesis H1 at the level of significance alpha = {self.alpha_sig:.2f}"'
            )
        else:
            conclusion_text = (
                f"We conclude that we accept the null hypothesis H0 at the level of significance "
                f"alpha = {self.alpha_sig:.2f} with optimistic degree {self.optimistic_degree:.4f} and "
                f"pessimistic degree {self.pessimistic_degree:.4f}. Now the confidence degree is "
                f"min{{{self.optimistic_degree:.4f}, 1 - {self.pessimistic_degree:.4f}}} = {self.confidence_degree:.4f}. "
                f"Therefore we can also say that we accept the null hypothesis H0 at the level of "
                f"significance alpha = {self.alpha_sig:.2f} with confidence degree {self.confidence_degree:.4f}."
            )
            fact_text = (
                f'Since the confidence degree is {"high" if self.confidence_degree >= 0.5 else "moderate"}, '
                f'we have a {"strong" if self.confidence_degree >= 0.7 else "moderate"} confidence to accept the following fact:\n'
                f'    "we accept the null hypothesis H0 at the level of significance alpha = {self.alpha_sig:.2f}"'
            )

        sections = [
            "=" * 72,
            "                   ONE-WAY FUZZY ANOVA SUMMARY",
            "=" * 72,
            f"Hypothesis Decision            : Accept {self.decision}",
            f"Significance Level (alpha)     : {self.alpha_sig:.4f}",
            f"Critical Value F_(1-a; r-1, nT-r): {self.f_crit:.4f}",
            f"Critical Multiplier rho        : {self.rho:.4f}",
            "-" * 72,
            "ANOVA Quantities on Core Values (h = 1):",
            f"  * SSTR (Between Groups)      : {self.sstr:.4f} (df = {self.df_between})",
            f"  * SSE  (Within Groups)       : {self.sse:.4f} (df = {self.df_within})",
            f"  * F-Statistic                : {self.f_stat:.4f}",
            f"  * P-Value (Core)             : {self.p_value:.4e}",
            f"  * Condition (SSTR > rho*SSE) : {self.sstr:.4f} {'>' if self.sstr > self.rho * self.sse else '<='} {self.rho * self.sse:.4f}",
        ]

        if self.alpha_cut_tables is not None:
            sections.extend([
                "-" * 72,
                "LOWER AND UPPER BOUND ANOVA TABLES:",
                self.alpha_cut_tables.lower.format_table(),
                "",
                self.alpha_cut_tables.upper.format_table(),
            ])

        sections.extend([
            "-" * 72,
            "Fuzzy Boundary Quadratic Forms & Roots:",
            f"  * Lower: qL(h) = ({self.lower_coeffs.A:.4f})*h^2 + ({self.lower_coeffs.B:.4f})*h + ({self.lower_coeffs.C:.4f})",
            f"    Roots of qL(h) = 0         : {roots_l_str}",
            f"  * Upper: qU(h) = ({self.upper_coeffs.A:.4f})*h^2 + ({self.upper_coeffs.B:.4f})*h + ({self.upper_coeffs.C:.4f})",
            f"    Roots of qU(h) = 0         : {roots_u_str}",
            "-" * 72,
            "Interval Sets K_{H0} and K_{H1}:",
            f"  * K_{{H0}} (qL <= 0 and qU <= 0) : {k_h0_str}",
            f"    Length lambda(K_{{H0}})        : {self.lambda_k_h0:.4f}",
            f"    Supremum h_{{H0}}             : {self.sup_k_h0:.4f}",
            f"  * K_{{H1}} (qL >= 0 and qU >= 0) : {k_h1_str}",
            f"    Length lambda(K_{{H1}})        : {self.lambda_k_h1:.4f}",
            f"    Supremum h_{{H1}}             : {self.sup_k_h1:.4f}",
            "-" * 72,
            "Degree of Belief Metrics:",
            f"  * Optimistic Degree          : {self.optimistic_degree:.4f}",
            f"  * Pessimistic Degree         : {self.pessimistic_degree:.4f}",
            f"  * Confidence Degree          : {self.confidence_degree:.4f}",
            "-" * 72,
            "CONCLUSION:",
            conclusion_text,
            "",
            fact_text,
            "=" * 72,
        ])
        return "\n".join(sections)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the ANOVA result to a dictionary for serialization / API responses."""
        return {
            "decision": self.decision,
            "is_null_rejected": self.is_null_rejected,
            "optimistic_degree": self.optimistic_degree,
            "pessimistic_degree": self.pessimistic_degree,
            "confidence_degree": self.confidence_degree,
            "sstr": self.sstr,
            "sse": self.sse,
            "f_stat": self.f_stat,
            "f_crit": self.f_crit,
            "p_value": self.p_value,
            "rho": self.rho,
            "rejection_threshold": self.rho * self.sse,
            "df_between": self.df_between,
            "df_within": self.df_within,
            "lower_coefficients": {
                "A": self.lower_coeffs.A,
                "B": self.lower_coeffs.B,
                "C": self.lower_coeffs.C,
            },
            "upper_coefficients": {
                "A": self.upper_coeffs.A,
                "B": self.upper_coeffs.B,
                "C": self.upper_coeffs.C,
            },
            "k_h0_intervals": self.k_h0_intervals,
            "k_h1_intervals": self.k_h1_intervals,
            "lambda_k_h0": self.lambda_k_h0,
            "lambda_k_h1": self.lambda_k_h1,
            "sup_k_h0": self.sup_k_h0,
            "sup_k_h1": self.sup_k_h1,
            "alpha_sig": self.alpha_sig,
        }

    def __repr__(self) -> str:
        return (
            f"FuzzyANOVAResult(decision='{self.decision}', "
            f"optimistic_degree={self.optimistic_degree:.4f}, "
            f"pessimistic_degree={self.pessimistic_degree:.4f}, "
            f"confidence_degree={self.confidence_degree:.4f}, "
            f"f_stat={self.f_stat:.4f}, "
            f"f_crit={self.f_crit:.4f})"
        )


def _validate_fuzzy_inputs(
    groups: GroupInput, alpha_sig: float
) -> Dict[str, List[Tuple[float, float, float]]]:
    """
    Validates input groups, triangular fuzzy numbers, and significance level.

    Raises
    ------
    TypeError
        If inputs have invalid data types.
    ValueError
        If groups count < 2, total N <= k, observations are invalid,
        or triangular fuzzy conditions (L <= Core <= U) are violated.
    """
    if not isinstance(alpha_sig, (int, float)) or math.isnan(alpha_sig):
        raise TypeError(f"alpha_sig must be a valid float, got {type(alpha_sig).__name__}.")
    if not (0.0 < alpha_sig < 1.0):
        raise ValueError(f"Significance level alpha_sig must be in (0, 1), got {alpha_sig}.")

    if isinstance(groups, Mapping):
        normalized_groups = dict(groups.items())
    elif isinstance(groups, Sequence) and not isinstance(groups, (str, bytes)):
        normalized_groups = {f"Group_{i+1}": g for i, g in enumerate(groups)}
    else:
        raise TypeError(
            f"groups must be a dictionary or sequence of groups, got {type(groups).__name__}."
        )

    num_groups = len(normalized_groups)
    if num_groups < 2:
        raise ValueError(f"Fuzzy ANOVA requires at least 2 groups, but received {num_groups}.")

    validated_groups: Dict[str, List[Tuple[float, float, float]]] = {}
    total_observations = 0

    for group_name, obs_list in normalized_groups.items():
        if not isinstance(group_name, str):
            group_name = str(group_name)

        if not isinstance(obs_list, Sequence) or isinstance(obs_list, (str, bytes)):
            raise TypeError(
                f"Observations for group '{group_name}' must be a sequence of fuzzy triplets, "
                f"got {type(obs_list).__name__}."
            )

        if len(obs_list) == 0:
            raise ValueError(f"Group '{group_name}' contains 0 observations.")

        parsed_obs: List[Tuple[float, float, float]] = []
        for idx, obs in enumerate(obs_list):
            if not isinstance(obs, Sequence) or isinstance(obs, (str, bytes)) or len(obs) != 3:
                raise ValueError(
                    f"Observation {idx} in group '{group_name}' is not a triplet (L, Core, U): {obs}."
                )

            try:
                L, core, U = float(obs[0]), float(obs[1]), float(obs[2])
            except (ValueError, TypeError) as e:
                raise TypeError(
                    f"All components of observation {idx} in group '{group_name}' must be numeric: {obs}."
                ) from e

            if math.isnan(L) or math.isnan(core) or math.isnan(U) or \
               math.isinf(L) or math.isinf(core) or math.isinf(U):
                raise ValueError(
                    f"Observation {idx} in group '{group_name}' contains NaN or Inf values: ({L}, {core}, {U})."
                )

            if not (L <= core <= U):
                raise ValueError(
                    f"Invalid triangular fuzzy number in group '{group_name}' at index {idx}: "
                    f"({L}, {core}, {U}) violates condition L <= Core <= U."
                )

            parsed_obs.append((L, core, U))

        validated_groups[group_name] = parsed_obs
        total_observations += len(parsed_obs)

    if total_observations <= num_groups:
        raise ValueError(
            f"Total number of observations ({total_observations}) must be strictly greater than "
            f"the number of groups ({num_groups}) to yield valid degrees of freedom (df_within > 0)."
        )

    return validated_groups


def get_alpha_cut(
    groups: GroupInput, h: float
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Computes the explicit alpha-cut (h-level set) interval endpoints for all observations.

    According to Wu (2007) Eq. (11):
        (Y_ij)^L_h = (1 - h) * Y^L_ij + h * Y_ij
        (Y_ij)^U_h = (1 - h) * Y^U_ij + h * Y_ij

    Parameters
    ----------
    groups : Mapping or Sequence
        Fuzzy data with triplets (L, Core, U).
    h : float
        The alpha-cut / h-level degree in [0.0, 1.0].

    Returns
    -------
    Dict[str, List[Tuple[float, float]]]
        Dictionary mapping each group to a list of (lower_bound, upper_bound) tuples at level h.
    """
    if not isinstance(h, (int, float)) or math.isnan(h) or not (0.0 <= h <= 1.0):
        raise ValueError(f"Cut level h must be a float in [0.0, 1.0], got {h}.")

    validated_groups = _validate_fuzzy_inputs(groups, alpha_sig=0.05)
    cut_dict: Dict[str, List[Tuple[float, float]]] = {}

    for g_name, obs_list in validated_groups.items():
        cut_list: List[Tuple[float, float]] = []
        for L, core, U in obs_list:
            lower_h = (1.0 - h) * L + h * core
            upper_h = (1.0 - h) * U + h * core
            cut_list.append((float(lower_h), float(upper_h)))
        cut_dict[g_name] = cut_list

    return cut_dict


def alpha_cut_anova(
    groups: GroupInput,
    h: float = 0.0,
    alpha_sig: float = 0.05
) -> AlphaCutANOVAResult:
    """
    Performs ANOVA on the lower and upper bounds of the fuzzy observations at a given h-level,
    generating Table 2 and Table 3 from Wu (2007).

    Parameters
    ----------
    groups : Mapping or Sequence
        Fuzzy data with triplets (L, Core, U).
    h : float, default=0.0
        The h-level cut in [0.0, 1.0]. Defaults to 0.0 (support bounds).
    alpha_sig : float, default=0.05
        Significance level for the F-test.

    Returns
    -------
    AlphaCutANOVAResult
        Container with Table 2 (lower bound) and Table 3 (upper bound) ANOVA calculations.
    """
    if not isinstance(h, (int, float)) or math.isnan(h) or not (0.0 <= h <= 1.0):
        raise ValueError(f"Cut level h must be a float in [0.0, 1.0], got {h}.")

    validated_groups = _validate_fuzzy_inputs(groups, alpha_sig)
    r = len(validated_groups)
    group_counts = {g: len(obs) for g, obs in validated_groups.items()}
    total_n = sum(group_counts.values())
    df_between = r - 1
    df_within = total_n - r
    df_total = total_n - 1
    f_crit = _f_ppf(1.0 - alpha_sig, df_between, df_within)

    # Extract lower and upper observations at level h
    lower_vals: Dict[str, List[float]] = {}
    upper_vals: Dict[str, List[float]] = {}
    for g, obs_list in validated_groups.items():
        lower_vals[g] = [(1.0 - h) * obs[0] + h * obs[1] for obs in obs_list]
        upper_vals[g] = [(1.0 - h) * obs[2] + h * obs[1] for obs in obs_list]

    def _calc_table(bound_name: str, vals: Dict[str, List[float]]) -> AlphaCutTable:
        grand_sum = sum(sum(v) for v in vals.values())
        grand_sum_sq = sum(sum(x ** 2 for x in v) for v in vals.values())
        sum_group_sq = sum((sum(v) ** 2) / group_counts[g] for g, v in vals.items())

        sstr = sum_group_sq - (grand_sum ** 2) / total_n
        sse = grand_sum_sq - sum_group_sq
        ssto = sstr + sse
        mstr = sstr / df_between
        mse = sse / df_within if sse > 1e-14 else 1e-14
        f_stat = mstr / mse
        p_val = 1.0 - _f_cdf(f_stat, df_between, df_within)

        return AlphaCutTable(
            bound_name=bound_name,
            h=h,
            sstr=float(sstr),
            sse=float(sse),
            ssto=float(ssto),
            df_between=df_between,
            df_within=df_within,
            df_total=df_total,
            mstr=float(mstr),
            mse=float(mse),
            f_stat=float(f_stat),
            p_value=float(p_val),
        )

    lower_table = _calc_table("lower", lower_vals)
    upper_table = _calc_table("upper", upper_vals)

    return AlphaCutANOVAResult(h=h, lower=lower_table, upper=upper_table, f_crit=float(f_crit))


def _compute_bound_coefficients(
    bound_per_group: Dict[str, List[float]],
    core_per_group: Dict[str, List[float]],
    group_counts: Dict[str, int],
    total_n: int,
    rho: float,
) -> QuadraticCoefficients:
    """
    Computes quadratic coefficients A, B, C for a fuzzy boundary (Lower or Upper)
    using the exact equations from Wu (2007) Page 14.
    """
    a_bound = {g: sum(vals) for g, vals in bound_per_group.items()}
    a_core = {g: sum(vals) for g, vals in core_per_group.items()}
    b_bound = sum(a_bound.values())
    b_core = sum(a_core.values())

    sum_diff_sq = sum(
        (b - c) ** 2
        for g in bound_per_group
        for b, c in zip(bound_per_group[g], core_per_group[g])
    )
    c1 = sum(b ** 2 for g in bound_per_group for b in bound_per_group[g])
    c2 = sum(b * c for g in bound_per_group for b, c in zip(bound_per_group[g], core_per_group[g]))

    # Quadratic coefficient A
    term_A1 = (1.0 + rho) * sum(
        ((a_bound[g] - a_core[g]) ** 2) / group_counts[g] for g in bound_per_group
    )
    term_A2 = ((b_bound - b_core) ** 2) / total_n
    term_A3 = rho * sum_diff_sq
    A = term_A1 - term_A2 - term_A3

    # Linear coefficient B
    term_B1 = (1.0 + rho) * sum(
        (a_bound[g] * (a_core[g] - a_bound[g])) / group_counts[g] for g in bound_per_group
    )
    term_B2 = (b_bound * (b_core - b_bound)) / total_n
    term_B3 = rho * (c2 - c1)
    B = 2.0 * (term_B1 - term_B2 - term_B3)

    # Constant coefficient C
    term_C1 = (1.0 + rho) * sum((a_bound[g] ** 2) / group_counts[g] for g in bound_per_group)
    term_C2 = (b_bound ** 2) / total_n
    term_C3 = rho * c1
    C = term_C1 - term_C2 - term_C3

    return QuadraticCoefficients(A=float(A), B=float(B), C=float(C))


def _find_sign_intervals(
    coeffs: QuadraticCoefficients, sign_cond: str, tol: float = 1e-9
) -> List[Tuple[float, float]]:
    """
    Finds the subintervals of [0.0, 1.0] where q(h) = A*h^2 + B*h + C satisfies:
      - 'LE': q(h) <= 0
      - 'GE': q(h) >= 0
    """
    A, B, C = coeffs.A, coeffs.B, coeffs.C

    def q(h: float) -> float:
        return A * h * h + B * h + C

    real_roots: List[float] = []
    if abs(A) < 1e-12:
        if abs(B) > 1e-12:
            r = -C / B
            if -tol <= r <= 1.0 + tol:
                real_roots.append(float(np.clip(r, 0.0, 1.0)))
    else:
        disc = B * B - 4.0 * A * C
        if disc >= -tol:
            sdisc = math.sqrt(max(0.0, disc))
            r1 = (-B - sdisc) / (2.0 * A)
            r2 = (-B + sdisc) / (2.0 * A)
            for r in (r1, r2):
                if -tol <= r <= 1.0 + tol:
                    real_roots.append(float(np.clip(r, 0.0, 1.0)))

    # Breakpoints in [0, 1]
    points = sorted(list(set([0.0] + real_roots + [1.0])))
    intervals: List[Tuple[float, float]] = []

    for i in range(len(points) - 1):
        p_start = points[i]
        p_end = points[i + 1]
        mid = 0.5 * (p_start + p_end)
        val = q(mid)
        if sign_cond == "LE" and (val <= tol or (abs(q(p_start)) <= tol and abs(q(p_end)) <= tol)):
            intervals.append((p_start, p_end))
        elif sign_cond == "GE" and (val >= -tol or (abs(q(p_start)) <= tol and abs(q(p_end)) <= tol)):
            intervals.append((p_start, p_end))

    return intervals


def _intersect_intervals(
    list1: List[Tuple[float, float]], list2: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """Computes the intersection of two lists of closed intervals."""
    result: List[Tuple[float, float]] = []
    for s1, e1 in list1:
        for s2, e2 in list2:
            start = max(s1, s2)
            end = min(e1, e2)
            if start < end - 1e-9:
                result.append((start, end))
            elif abs(start - end) <= 1e-9:
                result.append((start, start))
    return result


def _measure_and_sup(intervals: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Computes total length lambda(K) and supremum of an interval set."""
    if not intervals:
        return 0.0, 0.0
    length = sum(e - s for s, e in intervals)
    sup_val = max(e for s, e in intervals)
    return float(np.clip(length, 0.0, 1.0)), float(np.clip(sup_val, 0.0, 1.0))


def fuzzy_anova(
    groups: GroupInput,
    alpha_sig: float = 0.05,
    alpha_cut_level: float = 0.0,
    f_crit: Optional[float] = None,
) -> FuzzyANOVAResult:
    """
    Performs One-Way Fuzzy Analysis of Variance (ANOVA) on triangular fuzzy observations
    according to Wu (2007).

    Parameters
    ----------
    groups : Mapping[str, Sequence[Sequence[float]]] or Sequence[Sequence[Sequence[float]]]
        Triangular fuzzy observations formatted as (L, Core, U).
    alpha_sig : float, default=0.05
        Significance level for the F-test (must be in (0, 1)).
    alpha_cut_level : float, default=0.0
        The h-level cut to use for generating Upper/Lower bound ANOVA calculation tables
        (defaults to 0.0 for the full support intervals).
    f_crit : Optional[float], default=None
        Optional critical value F_(1-a; r-1, nT-r). If None, calculated automatically
        from Snedecor's F-distribution. Allows supplying textbook values (e.g. 4.76 in Wu 2007).

    Returns
    -------
    FuzzyANOVAResult
        Complete container holding hypothesis decision, optimistic/pessimistic degrees,
        ANOVA sums of squares, boundary quadratic coefficients, interval sets, and tables.
    """
    # 1. Validate inputs thoroughly
    validated_groups = _validate_fuzzy_inputs(groups, alpha_sig)

    # 2. Dimensions
    r = len(validated_groups)
    group_counts = {g: len(obs) for g, obs in validated_groups.items()}
    total_n = sum(group_counts.values())
    df_between = r - 1
    df_within = total_n - r

    # 3. Core values and classical ANOVA on Core (h = 1)
    core_per_group = {g: [obs[1] for obs in obs_list] for g, obs_list in validated_groups.items()}
    all_core_values = [val for group_vals in core_per_group.values() for val in group_vals]
    grand_core_mean = sum(all_core_values) / total_n

    sstr = sum(
        group_counts[g] * ((sum(core_per_group[g]) / group_counts[g] - grand_core_mean) ** 2)
        for g in validated_groups
    )
    sse = sum(
        sum((x - (sum(core_per_group[g]) / group_counts[g])) ** 2 for x in core_per_group[g])
        for g in validated_groups
    )

    # 4. Critical values and rho threshold
    if f_crit is None:
        computed_f_crit = _f_ppf(1.0 - alpha_sig, df_between, df_within)
    else:
        computed_f_crit = float(f_crit)
    rho = (float(df_between) / float(df_within)) * computed_f_crit

    if sse <= 1e-14:
        f_stat = float("inf") if sstr > 1e-14 else 0.0
        p_val = 0.0 if sstr > 1e-14 else 1.0
    else:
        f_stat = float((sstr / df_between) / (sse / df_within))
        p_val = float(1.0 - _f_cdf(f_stat, df_between, df_within))

    # 5. Extract lower and upper boundary observations
    lower_per_group = {g: [obs[0] for obs in obs_list] for g, obs_list in validated_groups.items()}
    upper_per_group = {g: [obs[2] for obs in obs_list] for g, obs_list in validated_groups.items()}

    # 6. Compute boundary quadratic coefficients AL, BL, CL and AU, BU, CU
    lower_coeffs = _compute_bound_coefficients(
        lower_per_group, core_per_group, group_counts, total_n, rho
    )
    upper_coeffs = _compute_bound_coefficients(
        upper_per_group, core_per_group, group_counts, total_n, rho
    )

    # 7. Compute interval sets K_{H0} and K_{H1} (Eq. 25 & 26)
    # K_{H0}: qL <= 0 and qU <= 0
    int_L_le = _find_sign_intervals(lower_coeffs, "LE")
    int_U_le = _find_sign_intervals(upper_coeffs, "LE")
    k_h0_intervals = _intersect_intervals(int_L_le, int_U_le)
    lambda_k_h0, sup_k_h0 = _measure_and_sup(k_h0_intervals)

    # K_{H1}: qL >= 0 and qU >= 0
    int_L_ge = _find_sign_intervals(lower_coeffs, "GE")
    int_U_ge = _find_sign_intervals(upper_coeffs, "GE")
    k_h1_intervals = _intersect_intervals(int_L_ge, int_U_ge)
    lambda_k_h1, sup_k_h1 = _measure_and_sup(k_h1_intervals)

    # 8. Decision Rule based on Wu (2007) Remark 4.1 and Decision Rules (I) & (II)
    if sstr > rho * sse:
        decision = "H1"
        # Decision Rule (II): h_{H1} = 1
        optimistic = lambda_k_h1
        pessimistic = sup_k_h0
    else:
        decision = "H0"
        # Decision Rule (I): h_{H0} = 1
        optimistic = lambda_k_h0
        pessimistic = sup_k_h1

    confidence = min(optimistic, 1.0 - pessimistic)

    # 9. Compute Upper & Lower Bound Calculation Tables (Wu Tables 2 & 3)
    alpha_cut_tables = alpha_cut_anova(validated_groups, h=alpha_cut_level, alpha_sig=alpha_sig)

    return FuzzyANOVAResult(
        decision=decision,
        optimistic_degree=float(optimistic),
        pessimistic_degree=float(pessimistic),
        confidence_degree=float(confidence),
        sstr=float(sstr),
        sse=float(sse),
        f_stat=float(f_stat),
        f_crit=float(computed_f_crit),
        p_value=float(p_val),
        rho=float(rho),
        df_between=df_between,
        df_within=df_within,
        lower_coeffs=lower_coeffs,
        upper_coeffs=upper_coeffs,
        alpha_sig=float(alpha_sig),
        k_h0_intervals=k_h0_intervals,
        k_h1_intervals=k_h1_intervals,
        lambda_k_h0=float(lambda_k_h0),
        lambda_k_h1=float(lambda_k_h1),
        sup_k_h0=float(sup_k_h0),
        sup_k_h1=float(sup_k_h1),
        alpha_cut_tables=alpha_cut_tables,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Command-line interface for fuzzy-anova.

    Usage:
        fuzzy-anova                     # Run Wu (2007) benchmark demonstration
        fuzzy-anova --demo              # Run Wu (2007) benchmark demonstration
        fuzzy-anova --version           # Display package version
        fuzzy-anova <data.json>         # Run Fuzzy ANOVA on a JSON data file
    """
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        prog="fuzzy-anova",
        description="One-Way Analysis of Variance (ANOVA) for Triangular Fuzzy Data (Wu, 2007)",
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the Wu (2007) benchmark demonstration problem.",
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="Optional path to a JSON file containing fuzzy observations (dict of groups).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level alpha for the F-test (default: 0.05).",
    )
    parser.add_argument(
        "--h",
        type=float,
        default=0.0,
        help="Alpha-cut level h for Lower/Upper bound ANOVA tables (default: 0.0).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output the ANOVA result as JSON instead of formatted text.",
    )

    args = parser.parse_args(argv)

    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            sys.stderr.write(f"Error reading JSON file '{args.file}': {e}\n")
            return 1

        result = fuzzy_anova(data, alpha_sig=args.alpha, alpha_cut_level=args.h)
        if args.output_json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(result.summary())
        return 0

    # Default / demo behavior
    wu_benchmark_data = {
        "Design_1": [(9, 11, 13), (14, 16, 18)],
        "Design_2": [(11, 15, 19), (10, 15, 21), (11, 13, 15)],
        "Design_3": [(15, 18, 21), (14, 17, 20), (17, 20, 23)],
        "Design_4": [(15, 19, 23), (21, 24, 27)],
    }
    result = fuzzy_anova(wu_benchmark_data, alpha_sig=args.alpha, alpha_cut_level=args.h)
    if args.output_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.summary())
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())