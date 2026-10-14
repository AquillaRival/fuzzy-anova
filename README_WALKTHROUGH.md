# Complete Line-by-Line Code Walkthrough: `fuzzy_anova.py`

This document explains **every line of code** in `fuzzy_anova.py` so that anyone — even without a programming background — can understand what each part does and why.

Reference Paper: *Hsien-Chung Wu (2007), "Analysis of variance for fuzzy data", International Journal of Systems Science, 38(3), 235-246.*

**Authors**: Harshit Trivedi & Dr. Masum Raj (Mathematics Professor)

---

# Table of Contents

1. [Section 1: Module Docstring & Imports (Lines 1–48)](#section-1-module-docstring--imports-lines-1-48)
2. [Section 2: Pure-Python F-Distribution Fallback (Lines 51–143)](#section-2-pure-python-f-distribution-fallback-lines-51-143)
3. [Section 3: Public API & Type Aliases (Lines 146–161)](#section-3-public-api--type-aliases-lines-146-161)
4. [Section 4: QuadraticCoefficients Class (Lines 164–204)](#section-4-quadraticcoefficients-class-lines-164-204)
5. [Section 5: AlphaCutTable & AlphaCutANOVAResult (Lines 207–254)](#section-5-alphacuttable--alphacutanovaresult-lines-207-254)
6. [Section 6: FuzzyANOVAResult Class (Lines 257–524)](#section-6-fuzzyanovaresult-class-lines-257-524)
7. [Section 7: Input Validation (Lines 527–612)](#section-7-input-validation-lines-527-612)
8. [Section 8: Alpha-Cut Method — get_alpha_cut (Lines 615–651)](#section-8-alpha-cut-method--get_alpha_cut-lines-615-651)
9. [Section 9: Alpha-Cut ANOVA Tables — alpha_cut_anova (Lines 654–727)](#section-9-alpha-cut-anova-tables--alpha_cut_anova-lines-654-727)
10. [Section 10: Quadratic Coefficient Computation (Lines 730–776)](#section-10-quadratic-coefficient-computation-lines-730-776)
11. [Section 11: Interval Sign Analysis (Lines 779–822)](#section-11-interval-sign-analysis-lines-779-822)
12. [Section 12: Interval Intersection & Measurement (Lines 825–847)](#section-12-interval-intersection--measurement-lines-825-847)
13. [Section 13: The Master Function — fuzzy_anova (Lines 850–982)](#section-13-the-master-function--fuzzy_anova-lines-850-982)
14. [Section 14: Main Block — Demo (Lines 985–995)](#section-14-main-block--demo-lines-985-995)
15. [Mathematical Symbols vs Code Variables](#mathematical-symbols-vs-code-variables)
16. [How to Run](#how-to-run)

---

# Section 1: Module Docstring & Imports (Lines 1–48)

## Lines 1–33: Module Docstring

```python
"""
Fuzzy ANOVA (Analysis of Variance for Triangular Fuzzy Data)
============================================================
...
"""
```

This is a Python docstring (documentation string). It does not execute — it is a comment block that describes what the entire file does. It explains:
- Each observation is a **Triangular Fuzzy Number (TFN)**: `(L, Core, U)` where `L ≤ Core ≤ U`.
- The module implements two key things:
  1. **Alpha-Cut Method**: Converts fuzzy numbers into real-valued intervals at each h-level.
  2. **Decision Rules with Optimistic/Pessimistic Degrees**: Uses quadratic equations to determine confidence in the hypothesis decision.

## Line 35: Future Annotations

```python
from __future__ import annotations
```

This is a Python compatibility line. It lets us write type hints like `List[float]` in a way that works on older Python versions (3.7+). It does not affect calculations.

## Lines 37–39: Core Library Imports

```python
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union
import math
```

- `dataclass, field` — Python tools to create structured data containers (like a form/template for results).
- `typing` — Provides type annotations so the code is self-documenting (e.g., `List[float]` means "a list of decimal numbers").
- `math` — Python's standard math library for `sqrt`, `log`, `lgamma`, `isnan`, `isinf`, etc.

## Line 40: NumPy Import

```python
import numpy as np
```

NumPy is a numerical library. We only use it for `np.clip(value, min, max)` — a function that clamps a number to stay within a range (e.g., clamp `1.0001` to `1.0`).

## Lines 42–48: Scipy Import with Fallback

```python
try:
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except Exception:
    _scipy_stats = None
    _HAS_SCIPY = False
```

**What this does**: Tries to import SciPy (a scientific computing library) for its F-distribution functions. If SciPy is not installed or is blocked by system policies (e.g., DLL restrictions on some Windows machines), it sets `_HAS_SCIPY = False` and the code uses its own built-in pure-Python implementation instead.

**Why**: This ensures the code **never crashes** due to missing or broken external libraries.

---

# Section 2: Pure-Python F-Distribution Fallback (Lines 51–143)

These three functions implement Snedecor's F-distribution from scratch using only basic math. They are used when SciPy is unavailable.

## Lines 51–87: `_betacf` — Continued Fraction for Incomplete Beta

```python
def _betacf(a: float, b: float, x: float, max_iter: int = 200, eps: float = 1e-14) -> float:
```

This implements **Lentz's continued fraction algorithm** to evaluate the regularized incomplete beta function. This is the mathematical engine behind the F-distribution CDF.

- **Lines 53–55**: Initialize helper variables `qab = a + b`, `qap = a + 1`, `qam = a - 1`.
- **Lines 56–61**: Set up the starting values. The `if abs(d) < 1e-30: d = 1e-30` prevents division by zero.
- **Lines 62–86**: The main loop iterates up to 200 times, alternating between "even" and "odd" steps of the continued fraction. Each iteration refines the approximation. The loop exits early (line 85) when the change `del_h` is close enough to 1.0 (meaning convergence).

## Lines 90–101: `_betainc` — Regularized Incomplete Beta Function

```python
def _betainc(a: float, b: float, x: float) -> float:
```

Computes $I_x(a, b)$, the regularized incomplete beta function. This is the core building block for computing the CDF of the F-distribution.

- **Line 92–95**: Edge cases — returns 0 if `x ≤ 0`, returns 1 if `x ≥ 1`.
- **Line 96**: Computes `log(Beta(a, b))` using the log-gamma function: $\ln B(a,b) = \ln\Gamma(a) + \ln\Gamma(b) - \ln\Gamma(a+b)$.
- **Line 97**: Computes the front factor: $bt = x^a (1-x)^b / B(a,b)$.
- **Lines 98–101**: Uses continued fraction from one direction or the other depending on which converges faster.

## Lines 104–114: `_f_cdf` — F-Distribution CDF

```python
def _f_cdf(f: float, df1: int, df2: int) -> float:
```

Computes $P(F \leq f)$ for Snedecor's F-distribution with `df1` and `df2` degrees of freedom.

- **Line 106–107**: If `f ≤ 0`, the probability is 0.
- **Lines 108–112**: If SciPy is available, use it (faster and more precise). If it fails, fall through.
- **Line 113**: Transform `f` into a beta variable: $x = \frac{df_1 \cdot f}{df_1 \cdot f + df_2}$.
- **Line 114**: Return `_betainc(df1/2, df2/2, x)` — the mathematical relationship between F-distribution and incomplete beta function.

## Lines 117–143: `_f_ppf` — F-Distribution Inverse CDF (Percent Point Function)

```python
def _f_ppf(p: float, df1: int, df2: int, tol: float = 1e-10) -> float:
```

Given a probability `p` (like 0.95), finds the F-value `f` such that $P(F \leq f) = p$. This is $F_{\text{crit}}$.

- **Lines 119–122**: Edge cases — if `p ≤ 0` return 0, if `p ≥ 1` return infinity.
- **Lines 123–127**: Try SciPy first if available.
- **Lines 128–133**: **Bracket finding** — starts with `[0, 1]` and doubles the upper bound until `CDF(high) ≥ p`. This ensures we have a range containing the answer.
- **Lines 134–142**: **Binary search (bisection method)** — repeatedly halves the interval. At each step, evaluates `CDF(mid)` and narrows to the half containing the answer. Runs for 100 iterations (giving ~30 decimal digits of precision).
- **Line 143**: Returns the midpoint as the final answer.

---

# Section 3: Public API & Type Aliases (Lines 146–161)

## Lines 146–154: `__all__`

```python
__all__ = [
    "fuzzy_anova",
    "get_alpha_cut",
    "alpha_cut_anova",
    "FuzzyANOVAResult",
    "AlphaCutANOVAResult",
    "AlphaCutTable",
    "QuadraticCoefficients",
]
```

This tells Python which names are "public" when someone does `from fuzzy_anova import *`. It's a convention for clean API design.

## Lines 156–161: Type Aliases

```python
FuzzyObservation = Tuple[float, float, float]
GroupInput = Union[
    Mapping[str, Sequence[Sequence[float]]],
    Sequence[Sequence[Sequence[float]]],
]
```

These define shorthand type names:
- `FuzzyObservation` = a triplet of floats `(L, Core, U)`.
- `GroupInput` = either a dictionary mapping group names to lists of triplets, or a plain list of lists of triplets.

---

# Section 4: QuadraticCoefficients Class (Lines 164–204)

```python
@dataclass(frozen=True)
class QuadraticCoefficients:
    A: float
    B: float
    C: float
```

This is a **data container** for the three coefficients of the quadratic equation $q(h) = A h^2 + B h + C$.

**Mathematical context**: In Wu (2007), the test whether `SSTR_h > ρ·SSE_h` at each h-level reduces to checking the sign of a quadratic in h. The coefficients A, B, C define that quadratic for either the lower bound or upper bound case.

### Key Methods:

- **`__iter__` (Lines 176–180)**: Allows `A, B, C = coeffs` — tuple unpacking.
- **`evaluate(h)` (Lines 182–184)**: Computes $A h^2 + B h + C$ at a specific h value.
- **`as_tuple()` (Lines 186–188)**: Returns `(A, B, C)` as a plain tuple.
- **`roots()` (Lines 190–204)**: Solves $A h^2 + B h + C = 0$ analytically:
  - **Line 193**: If `|A| < tolerance`, it's linear → root is `-C/B`.
  - **Line 197**: Compute discriminant $\Delta = B^2 - 4AC$.
  - **Line 198**: If $\Delta < 0$, no real roots → return empty list.
  - **Lines 200–204**: Otherwise, compute $r_{1,2} = \frac{-B \pm \sqrt{\Delta}}{2A}$ and return sorted.

---

# Section 5: AlphaCutTable & AlphaCutANOVAResult (Lines 207–254)

## Lines 207–239: `AlphaCutTable`

```python
@dataclass
class AlphaCutTable:
    bound_name: str   # "lower" or "upper"
    h: float          # The h-level
    sstr: float       # Treatment sum of squares
    sse: float        # Error sum of squares
    ssto: float       # Total sum of squares
    df_between: int   # Degrees of freedom between groups (r - 1)
    df_within: int    # Degrees of freedom within groups (nT - r)
    df_total: int     # Total degrees of freedom (nT - 1)
    mstr: float       # Mean square treatment = SSTR / df_between
    mse: float        # Mean square error = SSE / df_within
    f_stat: float     # F-statistic = MSTR / MSE
    p_value: float    # P-value of the F-statistic
```

This stores one complete ANOVA table — either the **Lower Bound Case** or the **Upper Bound Case** at a given h-level.

### `format_table()` (Lines 226–239):

Generates a nicely formatted text table:
```
ANOVA Table: Lower Bound Case (h-level = 0.00)
------------------------------------------------------------------------
Source of Variation                  SS     df           MS          F
------------------------------------------------------------------------
Between Treatments              82.2667      3      27.4222     4.5916
Error (within Treatments)       35.8333      6       5.9722
Total                          118.1000      9
------------------------------------------------------------------------
```

## Lines 242–254: `AlphaCutANOVAResult`

```python
@dataclass
class AlphaCutANOVAResult:
    h: float
    lower: AlphaCutTable
    upper: AlphaCutTable
    f_crit: float
```

A container that holds **both** the lower bound and upper bound ANOVA tables at a given h-level, plus the critical F-value.

- **`summary()` (Lines 252–254)**: Returns both tables formatted together.

---

# Section 6: FuzzyANOVAResult Class (Lines 257–524)

This is the **main result container** returned by `fuzzy_anova()`. It holds every computed value.

## Lines 311–332: Data Fields

| Field | Mathematical Meaning |
| :--- | :--- |
| `decision` | `"H0"` or `"H1"` — the hypothesis accepted |
| `optimistic_degree` | $\lambda(K_{\tilde{H}})$ — length of the interval set for the accepted hypothesis |
| `pessimistic_degree` | $\sup(K_{\text{opposing}})$ — supremum of the opposing set |
| `confidence_degree` | $\min(\text{optimistic}, 1 - \text{pessimistic})$ |
| `sstr` | $SSTR$ — Treatment sum of squares on core values |
| `sse` | $SSE$ — Error sum of squares on core values |
| `f_stat` | $F^* = \frac{MSTR}{MSE}$ — F-statistic on cores |
| `f_crit` | $F_{1-\alpha; r-1, n_T-r}$ — Critical F-value |
| `p_value` | P-value of $F^*$ |
| `rho` | $\rho = \frac{r-1}{n_T-r} \cdot F_{\text{crit}}$ — threshold multiplier |
| `df_between` | $r - 1$ |
| `df_within` | $n_T - r$ |
| `lower_coeffs` | $(A^L, B^L, C^L)$ — quadratic coefficients for lower bound |
| `upper_coeffs` | $(A^U, B^U, C^U)$ — quadratic coefficients for upper bound |
| `k_h0_intervals` | Intervals forming $K_{\tilde{H}_0}$ in $[0, 1]$ |
| `k_h1_intervals` | Intervals forming $K_{\tilde{H}_1}$ in $[0, 1]$ |
| `lambda_k_h0` | $\lambda(K_{\tilde{H}_0})$ — total length of $K_{\tilde{H}_0}$ |
| `lambda_k_h1` | $\lambda(K_{\tilde{H}_1})$ — total length of $K_{\tilde{H}_1}$ |
| `sup_k_h0` | $\sup(K_{\tilde{H}_0})$ — maximum h-value in $K_{\tilde{H}_0}$ |
| `sup_k_h1` | $\sup(K_{\tilde{H}_1})$ — maximum h-value in $K_{\tilde{H}_1}$ |
| `alpha_cut_tables` | The Lower/Upper bound ANOVA tables |

## Lines 334–369: Property Aliases

```python
@property
def SSTR(self) -> float:
    return self.sstr
```

These are backward-compatible shortcuts. For example, `result.SSTR` returns the same thing as `result.sstr`. Similarly for `SSE`, `F_crit`, `AL`, `BL`, `CL`, `AU`, `BU`, `CU`.

## Lines 371–374: `is_null_rejected`

```python
@property
def is_null_rejected(self) -> bool:
    return self.decision == "H1"
```

Returns `True` if the null hypothesis was rejected (i.e., the decision was H1).

## Lines 376–478: `summary()` — Human-Readable Output

This builds the formatted text report. Key sections:

- **Lines 381–393**: Compute and format the roots of the quadratic equations and the interval sets.
- **Lines 396–423**: Build the **conclusion text** matching Wu (2007) wording:
  - If H1: *"We conclude that we accept the alternative hypothesis H1..."*
  - If H0: *"We conclude that we accept the null hypothesis H0..."*
  - Also builds a **fact statement** with confidence assessment ("high"/"moderate", "strong"/"moderate").
- **Lines 425–478**: Assemble all sections into a formatted string:
  1. Header with decision, alpha, F_crit, rho.
  2. Core ANOVA quantities (SSTR, SSE, F-stat, P-value).
  3. Lower and Upper bound ANOVA tables (if computed).
  4. Quadratic equations and their roots.
  5. Interval sets $K_{\tilde{H}_0}$ and $K_{\tilde{H}_1}$ with their lengths and supremums.
  6. Degree of Belief Metrics (optimistic, pessimistic, confidence).
  7. Conclusion statement.

## Lines 480–514: `to_dict()` — JSON-Ready Serialization

Converts all results into a plain Python dictionary, suitable for JSON export or API responses.

## Lines 516–524: `__repr__` — Debug Representation

When you type `print(result)` without `.summary()`, this shows a compact one-line representation.

---

# Section 7: Input Validation (Lines 527–612)

```python
def _validate_fuzzy_inputs(groups, alpha_sig) -> Dict[str, List[Tuple[float, float, float]]]:
```

This function checks that all inputs are valid before any computation begins.

### Line-by-line:

- **Lines 541–544**: Check that `alpha_sig` is a valid number in `(0, 1)`. Raises `TypeError` if not a number, `ValueError` if out of range.
- **Lines 546–553**: Normalize input format:
  - If `groups` is a dictionary (e.g., `{"A": [...], "B": [...]}`), use it directly.
  - If `groups` is a list of lists, auto-name them `"Group_1"`, `"Group_2"`, etc.
  - Otherwise, raise `TypeError`.
- **Lines 555–557**: Check we have at least 2 groups (ANOVA needs groups to compare).
- **Lines 562–604**: Loop through every observation in every group:
  - **Line 563–564**: Ensure group name is a string.
  - **Lines 566–570**: Each group's data must be a sequence (list), not a string.
  - **Lines 572–573**: Each group must have at least 1 observation.
  - **Lines 577–580**: Each observation must be exactly 3 values (the triplet).
  - **Lines 582–587**: Convert all 3 values to floats. Raises `TypeError` if they can't be converted.
  - **Lines 589–593**: Reject `NaN` or `Inf` values.
  - **Lines 595–599**: **Enforce the triangular fuzzy condition**: $L \leq Core \leq U$.
  - **Line 601**: Store the validated triplet.
- **Lines 606–610**: Check that total observations $N >$ number of groups $k$ (otherwise $df_{\text{within}} \leq 0$, which is mathematically undefined).

---

# Section 8: Alpha-Cut Method — `get_alpha_cut` (Lines 615–651)

```python
def get_alpha_cut(groups, h) -> Dict[str, List[Tuple[float, float]]]:
```

**Mathematical basis** (Wu 2007, Equation 11): For a triangular fuzzy number $(L_{ij}, Core_{ij}, U_{ij})$, the **h-level cut** (also called alpha-cut) produces a closed interval:

$$[\tilde{Y}_{ij}]_h = \Big[(1 - h) \cdot L_{ij} + h \cdot Core_{ij}, \;\; (1 - h) \cdot U_{ij} + h \cdot Core_{ij}\Big]$$

### Line-by-line:

- **Lines 637–638**: Validate that `h` is in `[0, 1]`.
- **Line 640**: Validate the input data using `_validate_fuzzy_inputs`.
- **Lines 643–649**: For each observation $(L, Core, U)$:
  - **Line 646**: Lower bound at h: `(1 - h) * L + h * Core` → As h goes from 0→1, this moves from L toward Core.
  - **Line 647**: Upper bound at h: `(1 - h) * U + h * Core` → As h goes from 0→1, this moves from U toward Core.
  - At `h = 0`: interval is `[L, U]` (full support).
  - At `h = 1`: interval is `[Core, Core]` (just the peak).
  - At `h = 0.5`: interval is `[(L+Core)/2, (U+Core)/2]` (midway).

---

# Section 9: Alpha-Cut ANOVA Tables — `alpha_cut_anova` (Lines 654–727)

```python
def alpha_cut_anova(groups, h=0.0, alpha_sig=0.05) -> AlphaCutANOVAResult:
```

Performs **classical ANOVA** separately on the lower bounds and upper bounds of the fuzzy observations at a given h-level. This produces the Lower Bound ANOVA Table and the Upper Bound ANOVA Table.

### Line-by-line:

- **Lines 677–678**: Validate h is in `[0, 1]`.
- **Lines 680–687**: Setup — count groups, observations, degrees of freedom, compute F_crit.
- **Lines 689–694**: Extract lower and upper observations at level h:
  - **Line 693**: `lower_vals[g]` = list of `(1-h)*L + h*Core` for each observation.
  - **Line 694**: `upper_vals[g]` = list of `(1-h)*U + h*Core` for each observation.
- **Lines 696–722**: `_calc_table()` — inner function that computes one ANOVA table:
  - **Line 697**: `grand_sum` = $b = \sum_i \sum_j y_{ij}$ (sum of all observations).
  - **Line 698**: `grand_sum_sq` = $\sum_i \sum_j y_{ij}^2$ (sum of all squared observations).
  - **Line 699**: `sum_group_sq` = $\sum_i \frac{a_i^2}{n_i}$ where $a_i$ is the sum of group $i$.
  - **Line 701**: $SSTR = \sum_i \frac{a_i^2}{n_i} - \frac{b^2}{N}$ (between-group variation).
  - **Line 702**: $SSE = \sum_i \sum_j y_{ij}^2 - \sum_i \frac{a_i^2}{n_i}$ (within-group variation).
  - **Line 703**: $SSTO = SSTR + SSE$ (total variation).
  - **Line 704**: $MSTR = SSTR / (r - 1)$ (mean square treatment).
  - **Line 705**: $MSE = SSE / (n_T - r)$ (mean square error), with a safety check for zero SSE.
  - **Line 706**: $F = MSTR / MSE$ (F-statistic).
  - **Line 707**: P-value = $1 - F_{\text{CDF}}(F, df_1, df_2)$.
- **Lines 724–727**: Call `_calc_table` for both "lower" and "upper" bounds, package into result.

---

# Section 10: Quadratic Coefficient Computation (Lines 730–776)

```python
def _compute_bound_coefficients(bound_per_group, core_per_group, group_counts, total_n, rho):
```

**Mathematical basis**: The condition $SSTR_h \leq \rho \cdot SSE_h$ (or $\geq$) at each h-level, when expanded algebraically, becomes a quadratic in h:

$$q(h) = A h^2 + B h + C$$

This function computes the coefficients A, B, C for either the lower or upper bound case.

### Notation:
- $a_i^{bound}$ = sum of bound values (L or U) in group $i$ → code: `a_bound[g]`
- $a_i$ = sum of core values in group $i$ → code: `a_core[g]`
- $b^{bound}$ = grand total of bound values → code: `b_bound`
- $b$ = grand total of core values → code: `b_core`
- $c_1$ = $\sum_{i,j} (B_{ij})^2$ → code: `c1`
- $c_2$ = $\sum_{i,j} B_{ij} \cdot Core_{ij}$ → code: `c2`

### Line-by-line:

- **Lines 741–744**: Compute group sums `a_bound`, `a_core` and grand totals `b_bound`, `b_core`.
- **Lines 746–752**: Compute three intermediate sums:
  - `sum_diff_sq` = $\sum_{i,j} (B_{ij} - Core_{ij})^2$
  - `c1` = $\sum_{i,j} B_{ij}^2$
  - `c2` = $\sum_{i,j} B_{ij} \cdot Core_{ij}$
- **Lines 754–760**: **Coefficient A** (h² term):
  $$A = (1 + \rho) \sum_i \frac{(a_i^{bound} - a_i)^2}{n_i} - \frac{(b^{bound} - b)^2}{N} - \rho \sum_{i,j} (B_{ij} - Core_{ij})^2$$
- **Lines 762–768**: **Coefficient B** (h term):
  $$B = 2\left[(1+\rho)\sum_i \frac{a_i^{bound}(a_i - a_i^{bound})}{n_i} - \frac{b^{bound}(b - b^{bound})}{N} - \rho(c_2 - c_1)\right]$$
- **Lines 770–776**: **Coefficient C** (constant term):
  $$C = (1+\rho)\sum_i \frac{(a_i^{bound})^2}{n_i} - \frac{(b^{bound})^2}{N} - \rho \cdot c_1$$

---

# Section 11: Interval Sign Analysis (Lines 779–822)

```python
def _find_sign_intervals(coeffs, sign_cond, tol=1e-9) -> List[Tuple[float, float]]:
```

Finds the sub-intervals of $[0, 1]$ where $q(h) = Ah^2 + Bh + C$ is either $\leq 0$ (for $K_{\tilde{H}_0}$) or $\geq 0$ (for $K_{\tilde{H}_1}$).

### Algorithm:

1. **Lines 792–806: Find roots in [0, 1]**:
   - If `|A| < ε` (effectively linear): solve `-C/B` and keep if in `[0, 1]`.
   - Otherwise: compute discriminant $\Delta = B^2 - 4AC$.
     - If $\Delta \geq 0$: compute $r_1 = \frac{-B - \sqrt{\Delta}}{2A}$, $r_2 = \frac{-B + \sqrt{\Delta}}{2A}$.
     - Keep roots that fall within `[0, 1]`.

2. **Lines 808–809: Create breakpoints**: `[0.0] + roots + [1.0]`, sorted and deduplicated. These divide `[0, 1]` into sub-intervals.

3. **Lines 812–821: Test each sub-interval**: For each interval `[p_start, p_end]`:
   - Evaluate `q(midpoint)`.
   - If looking for `LE` (≤ 0) and the midpoint value is ≤ 0, include this interval.
   - If looking for `GE` (≥ 0) and the midpoint value is ≥ 0, include this interval.

**Why midpoint testing works**: A quadratic can only change sign at its roots. Between consecutive roots, its sign is constant. So checking one point in each sub-interval is sufficient.

---

# Section 12: Interval Intersection & Measurement (Lines 825–847)

## Lines 825–838: `_intersect_intervals`

```python
def _intersect_intervals(list1, list2) -> List[Tuple[float, float]]:
```

Computes the **set intersection** of two lists of intervals. For example:
- `list1 = [(0.0, 0.5), (0.7, 1.0)]`
- `list2 = [(0.3, 0.8)]`
- Result: `[(0.3, 0.5), (0.7, 0.8)]`

This is needed because $K_{\tilde{H}_0}$ requires BOTH $q_L(h) \leq 0$ AND $q_U(h) \leq 0$ simultaneously.

- **Lines 830–838**: For each pair of intervals from list1 and list2:
  - `start = max(s1, s2)`, `end = min(e1, e2)`.
  - If `start < end`, there's an overlap → add it to the result.

## Lines 841–847: `_measure_and_sup`

```python
def _measure_and_sup(intervals) -> Tuple[float, float]:
```

Computes two things about a set of intervals:
1. **Length** ($\lambda$): Total combined length = sum of `(end - start)` for each interval. This is the **optimistic degree**.
2. **Supremum** ($\sup$): The largest right endpoint. This is the **pessimistic degree** of the opposing hypothesis.

- **Lines 843–844**: If empty set, return `(0.0, 0.0)`.
- **Line 845**: `length` = total combined length of all intervals.
- **Line 846**: `sup_val` = maximum right endpoint.
- **Line 847**: Clamp both to `[0, 1]` and return.

---

# Section 13: The Master Function — `fuzzy_anova` (Lines 850–982)

```python
def fuzzy_anova(groups, alpha_sig=0.05, alpha_cut_level=0.0, f_crit=None) -> FuzzyANOVAResult:
```

This is the **main entry point**. It orchestrates the entire Fuzzy ANOVA computation.

### Parameters:
- `groups`: Your fuzzy data — dict or list of groups with `(L, Core, U)` triplets.
- `alpha_sig`: Significance level (default 0.05 = 95% confidence).
- `alpha_cut_level`: Which h-level to use for the ANOVA tables (default 0.0 = full support).
- `f_crit`: Optional override for $F_{\text{crit}}$ (for matching textbook values).

### Step-by-Step Execution:

#### Step 1 (Line 880): Validate Inputs
```python
validated_groups = _validate_fuzzy_inputs(groups, alpha_sig)
```
All data is checked and normalized.

#### Step 2 (Lines 882–887): Compute Dimensions
```python
r = len(validated_groups)                          # Number of groups (treatments)
group_counts = {g: len(obs) for ...}               # n_i for each group
total_n = sum(group_counts.values())               # Total observations N = Σn_i
df_between = r - 1                                 # Degrees of freedom between groups
df_within = total_n - r                            # Degrees of freedom within groups
```

#### Step 3 (Lines 889–901): Classical ANOVA on Core Values (h = 1)
```python
core_per_group = {g: [obs[1] for obs in obs_list] ...}   # Extract core values
grand_core_mean = sum(all_core_values) / total_n          # Grand mean ȳ..
```

**SSTR** (Line 894): $\sum_i n_i (\bar{y}_{i\cdot} - \bar{y}_{\cdot\cdot})^2$ — How much do group means differ from the overall mean?

**SSE** (Line 898): $\sum_i \sum_j (y_{ij} - \bar{y}_{i\cdot})^2$ — How much do individual observations vary within their groups?

#### Step 4 (Lines 903–915): Critical Values and F-test
```python
if f_crit is None:
    computed_f_crit = _f_ppf(1.0 - alpha_sig, df_between, df_within)   # Auto-calculate
else:
    computed_f_crit = float(f_crit)                                     # Use provided value
rho = (df_between / df_within) * computed_f_crit                        # ρ = (r-1)/(N-r) × F_crit
```

Then compute the F-statistic and p-value:
- **Lines 910–912**: Safety check — if SSE ≈ 0 (all observations identical within groups), set F = ∞ if SSTR > 0, else F = 0.
- **Lines 914–915**: Normal case — $F^* = \frac{SSTR/(r-1)}{SSE/(N-r)}$ and p-value from CDF.

#### Step 5 (Lines 917–919): Extract Boundary Observations
```python
lower_per_group = {g: [obs[0] for obs in obs_list] ...}   # All L values
upper_per_group = {g: [obs[2] for obs in obs_list] ...}   # All U values
```

#### Step 6 (Lines 921–927): Compute Quadratic Coefficients
```python
lower_coeffs = _compute_bound_coefficients(lower_per_group, core_per_group, ...)
upper_coeffs = _compute_bound_coefficients(upper_per_group, core_per_group, ...)
```

This gives us $(A^L, B^L, C^L)$ for the lower bound quadratic and $(A^U, B^U, C^U)$ for the upper bound quadratic.

#### Step 7 (Lines 929–940): Compute Interval Sets $K_{\tilde{H}_0}$ and $K_{\tilde{H}_1}$

```python
# K_{H0}: where BOTH qL(h) ≤ 0 AND qU(h) ≤ 0
int_L_le = _find_sign_intervals(lower_coeffs, "LE")     # Where qL ≤ 0
int_U_le = _find_sign_intervals(upper_coeffs, "LE")     # Where qU ≤ 0
k_h0_intervals = _intersect_intervals(int_L_le, int_U_le)  # Both simultaneously
lambda_k_h0, sup_k_h0 = _measure_and_sup(k_h0_intervals)   # Length and supremum

# K_{H1}: where BOTH qL(h) ≥ 0 AND qU(h) ≥ 0
int_L_ge = _find_sign_intervals(lower_coeffs, "GE")     # Where qL ≥ 0
int_U_ge = _find_sign_intervals(upper_coeffs, "GE")     # Where qU ≥ 0
k_h1_intervals = _intersect_intervals(int_L_ge, int_U_ge)  # Both simultaneously
lambda_k_h1, sup_k_h1 = _measure_and_sup(k_h1_intervals)   # Length and supremum
```

**Interpretation**: $K_{\tilde{H}_0}$ is the set of h-values where the data supports $H_0$ (equal means). $K_{\tilde{H}_1}$ is the set of h-values where the data supports $H_1$ (different means).

#### Step 8 (Lines 942–954): Decision Rules

```python
if sstr > rho * sse:
    decision = "H1"                      # Reject H0 → Accept H1
    optimistic = lambda_k_h1             # Length of K_{H1}
    pessimistic = sup_k_h0              # Supremum of K_{H0}
else:
    decision = "H0"                      # Accept H0
    optimistic = lambda_k_h0             # Length of K_{H0}
    pessimistic = sup_k_h1              # Supremum of K_{H1}

confidence = min(optimistic, 1.0 - pessimistic)
```

**Decision Rule (Wu 2007)**:
- The core decision at h=1: if $SSTR > \rho \cdot SSE$, reject $H_0$.
- **Optimistic degree** = how much of $[0, 1]$ supports our decision = $\lambda(K)$.
- **Pessimistic degree** = how far into $[0, 1]$ the opposing hypothesis reaches = $\sup(K_{\text{opposing}})$.
- **Confidence degree** = $\min(\text{optimistic}, 1 - \text{pessimistic})$.

#### Step 9 (Line 957): Generate ANOVA Tables

```python
alpha_cut_tables = alpha_cut_anova(validated_groups, h=alpha_cut_level, alpha_sig=alpha_sig)
```

Computes the Lower Bound and Upper Bound ANOVA tables at the specified h-level.

#### Lines 959–982: Package Everything Into Result

All computed values are assembled into a `FuzzyANOVAResult` object and returned.

---

# Section 14: Main Block — Demo (Lines 985–995)

```python
if __name__ == "__main__":
    wu_benchmark_data = {
        "Design_1": [(9, 11, 13), (14, 16, 18)],
        "Design_2": [(11, 15, 19), (10, 15, 21), (11, 13, 15)],
        "Design_3": [(15, 18, 21), (14, 17, 20), (17, 20, 23)],
        "Design_4": [(15, 19, 23), (21, 24, 27)],
    }
    res = fuzzy_anova(wu_benchmark_data, alpha_sig=0.05, alpha_cut_level=0.0)
    print(res.summary())
```

This block runs **only** when you execute `python fuzzy_anova.py` directly. It uses the exact dataset from Section 5 of Wu (2007) as a demonstration and prints the full summary.

---

# Mathematical Symbols vs Code Variables

| Math Symbol | Code Variable | Description |
| :--- | :--- | :--- |
| $r$ | `r` | Number of treatment groups |
| $n_i$ | `group_counts[g]` | Number of observations in group $i$ |
| $n_T$ or $N$ | `total_n` | Total number of observations |
| $Y_{ij}$ | `obs[1]` (core) | Core value of observation $j$ in group $i$ |
| $Y^L_{ij}$ | `obs[0]` | Lower bound of observation $j$ in group $i$ |
| $Y^U_{ij}$ | `obs[2]` | Upper bound of observation $j$ in group $i$ |
| $h$ | `h` | Alpha-cut level in $[0, 1]$ |
| $\alpha$ | `alpha_sig` | Significance level (e.g., 0.05) |
| $SSTR$ | `sstr` | Treatment (between-group) sum of squares |
| $SSE$ | `sse` | Error (within-group) sum of squares |
| $F^*$ | `f_stat` | F-statistic |
| $F_{\text{crit}}$ | `f_crit` / `computed_f_crit` | Critical F-value from F-distribution |
| $\rho$ | `rho` | $\frac{r-1}{n_T - r} \cdot F_{\text{crit}}$ |
| $a_i^L$ | `a_bound[g]` (lower) | Sum of lower bounds in group $i$ |
| $a_i^U$ | `a_bound[g]` (upper) | Sum of upper bounds in group $i$ |
| $a_i$ | `a_core[g]` | Sum of core values in group $i$ |
| $b^L$ | `b_bound` (lower) | Grand total of lower bounds |
| $b^U$ | `b_bound` (upper) | Grand total of upper bounds |
| $b$ | `b_core` | Grand total of core values |
| $A^L, B^L, C^L$ | `lower_coeffs.A, .B, .C` | Quadratic coefficients for lower bound |
| $A^U, B^U, C^U$ | `upper_coeffs.A, .B, .C` | Quadratic coefficients for upper bound |
| $K_{\tilde{H}_0}$ | `k_h0_intervals` | Set of h-values supporting $H_0$ |
| $K_{\tilde{H}_1}$ | `k_h1_intervals` | Set of h-values supporting $H_1$ |
| $\lambda(K)$ | `lambda_k_h0` / `lambda_k_h1` | Length (measure) of an interval set |
| $\sup(K)$ | `sup_k_h0` / `sup_k_h1` | Supremum of an interval set |

---

# How to Run

### Run the built-in demo (Wu 2007 benchmark):
```bash
python fuzzy_anova.py
```

### Run with your own data:
```bash
python test.py
```

### Run the automated test suite (23 tests):
```bash
python -m unittest test_fuzzy_anova.py -v
```

### Prerequisites:
- **Python 3.8+**
- **NumPy** (`pip install numpy`)
- SciPy is optional — the code works without it.
