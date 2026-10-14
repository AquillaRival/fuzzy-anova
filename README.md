# fuzzy-anova

**One-Way Analysis of Variance (ANOVA) for Triangular Fuzzy Data**

A complete Python implementation of the fuzzy ANOVA framework based on:

> Hsien-Chung Wu (2007), *"Analysis of variance for fuzzy data"*, International Journal of Systems Science, 38(3), 235-246.

[![PyPI version](https://img.shields.io/pypi/v/fuzzy-anova.svg)](https://pypi.org/project/fuzzy-anova/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Type Checked](https://img.shields.io/badge/typing-typed-brightgreen.svg)](py.typed)

*Developed by **Harshit Trivedi** and **Dr. Masum Raj** (Mathematics Professor).*

---

## Overview

In traditional ANOVA, observations are crisp real numbers. In many real-world experiments—such as sensory evaluation, customer satisfaction surveys, and engineering tolerances—observations are inherently imprecise and naturally expressed as **Triangular Fuzzy Numbers (TFN)**:

$$y_{ij} = (L_{ij}, \text{Core}_{ij}, U_{ij})$$

where:
- $L_{ij}$ is the lower bound of support (membership degree = 0)
- $\text{Core}_{ij}$ is the modal / core value (membership degree = 1)
- $U_{ij}$ is the upper bound of support (membership degree = 0)
- Condition: $L_{ij} \le \text{Core}_{ij} \le U_{ij}$

`fuzzy-anova` implements the complete Wu (2007) methodology, providing:
1. **Alpha-Cut ($h$-level set) Method**: Converts fuzzy numbers into real intervals at any membership degree $h \in [0, 1]$.
2. **Lower & Upper Bound ANOVA Tables**: Full ANOVA tables (SS, df, MS, F-statistic, P-value) calculated at any level-cut (Wu Table 2 & Table 3).
3. **Quadratic Boundary Analysis**: Computes exact quadratic coefficients $A \cdot h^2 + B \cdot h + C$ for both fuzzy boundaries.
4. **Interval Sets $K_{H0}$ and $K_{H1}$**: Determines the valid parameter ranges for accepting $H_0$ or $H_1$ via quadratic constraint optimization.
5. **Degree of Belief Metrics**:
   - **Optimistic Degree** $\lambda(K)$: Total interval measure / length of the supporting parameter set.
   - **Pessimistic Degree** $h_H$: Supremum of the opposing parameter set.
   - **Confidence Degree**: Overall balanced decision confidence $\min(\text{optimistic}, 1 - \text{pessimistic})$.
6. **Zero Mandatory Heavy Dependencies**: Fast pure-Python Snedecor's F-distribution fallback included, with automatic SciPy acceleration if SciPy is installed.

---

## Installation

```bash
pip install fuzzy-anova
```

### Optional Dependencies

For faster F-distribution calculation using SciPy:

```bash
pip install "fuzzy-anova[scipy]"
```

For development and running tests:

```bash
pip install "fuzzy-anova[dev]"
```

---

## Quickstart

```python
from fuzzy_anova import fuzzy_anova

# 1. Define groups with triangular fuzzy numbers (Lower, Core, Upper)
data = {
    "Design_1": [(9, 11, 13), (14, 16, 18)],
    "Design_2": [(11, 15, 19), (10, 15, 21), (11, 13, 15)],
    "Design_3": [(15, 18, 21), (14, 17, 20), (17, 20, 23)],
    "Design_4": [(15, 19, 23), (21, 24, 27)],
}

# 2. Run Fuzzy ANOVA at significance level alpha = 0.05
result = fuzzy_anova(data, alpha_sig=0.05)

# 3. Print the comprehensive formatted summary report
print(result.summary())
```

### Accessing Result Quantities

```python
# Decision and confidence metrics
print("Decision:           ", result.decision)            # 'H0' or 'H1'
print("Null Rejected:      ", result.is_null_rejected)     # True or False
print("Optimistic Degree:  ", result.optimistic_degree)   # e.g. 0.8808
print("Pessimistic Degree: ", result.pessimistic_degree)  # e.g. 0.0031
print("Confidence Degree:  ", result.confidence_degree)   # e.g. 0.8808

# Classical ANOVA core values (at h = 1)
print("SSTR:               ", result.sstr)
print("SSE:                ", result.sse)
print("F-Statistic:        ", result.f_stat)
print("Critical F-Value:   ", result.f_crit)
print("P-Value:            ", result.p_value)

# Boundary quadratic coefficients (A, B, C)
print("Lower Coeffs (A,B,C):", result.lower_coeffs.as_tuple())
print("Upper Coeffs (A,B,C):", result.upper_coeffs.as_tuple())

# Convert full results to a Python dictionary (ready for JSON serialization)
data_dict = result.to_dict()
```

---

## Alpha-Cut and Boundary Tables

You can evaluate explicit alpha-cuts or compute the lower/upper bound ANOVA tables at any specific cut level $h$:

```python
from fuzzy_anova import get_alpha_cut, alpha_cut_anova

# 1. Get explicit interval endpoints [lower, upper] at h = 0.5
cuts = get_alpha_cut(data, h=0.5)
print(cuts["Design_1"])  # [(10.0, 12.0), (15.0, 17.0)]

# 2. Generate Table 2 (Lower Bound) & Table 3 (Upper Bound) at h = 0.0 (full support)
tables = alpha_cut_anova(data, h=0.0, alpha_sig=0.05)
print(tables.summary())
```

---

## Command-Line Interface (CLI)

`fuzzy-anova` provides a built-in CLI:

```bash
# Run the benchmark demonstration problem from Wu (2007)
fuzzy-anova --demo

# View version
fuzzy-anova --version

# Run ANOVA on your own dataset stored in a JSON file
fuzzy-anova data.json --alpha 0.05 --json
```

---

## API Reference

### `fuzzy_anova(groups, alpha_sig=0.05, alpha_cut_level=0.0, f_crit=None) -> FuzzyANOVAResult`

Main function to perform One-Way Fuzzy ANOVA.

| Parameter | Type | Default | Description |
| :--- | :--- | :---: | :--- |
| `groups` | `Mapping` or `Sequence` | *required* | Dictionary of `{group_name: [(L, C, U), ...]}` or list of lists `[[(L, C, U), ...], ...]`. |
| `alpha_sig` | `float` | `0.05` | Statistical significance level $\alpha \in (0, 1)$. |
| `alpha_cut_level` | `float` | `0.0` | Cut level $h \in [0, 1]$ used for generating bound ANOVA tables. |
| `f_crit` | `float` or `None` | `None` | Optional critical value override (e.g. for matching textbook values). |

### `get_alpha_cut(groups, h: float) -> Dict[str, List[Tuple[float, float]]]`

Returns interval endpoints `(lower_h, upper_h)` for each observation at membership cut $h \in [0, 1]$.

### `alpha_cut_anova(groups, h=0.0, alpha_sig=0.05) -> AlphaCutANOVAResult`

Computes the classical ANOVA table for both lower and upper bounds at level $h$.

---

## Verification & Testing

The package includes an extensive test suite verifying:
- Numerical accuracy against Wu (2007) published benchmarks to 4 decimal places
- Degenerate crisp numbers ($L = \text{Core} = U$) collapsing to classical ANOVA
- Identical groups accepting $H_0$ with confidence 1.0
- Widely separated groups rejecting $H_0$ with confidence 1.0
- Unequal sample sizes across treatments
- Asymmetric and overlapping fuzzy numbers
- Comprehensive input validation (types, shapes, NaN/Inf, ordering violations)

To run the automated tests:

```bash
python test_fuzzy_anova.py
```

or with `pytest`:

```bash
pytest
```

---

---

## Authors & Maintainers

* **Harshit Trivedi** — *Software Architect & Core Implementation* ([GitHub](https://github.com/AquillaRival))
* **Dr. Masum Raj** — *Professor of Mathematics, Academic Mentorship & Theoretical Mathematical Verification*

---

## Acknowledgements

Special gratitude to **Dr. Masum Raj** for mathematics mentorship, guiding the research on fuzzy hypothesis testing, and validating the numerical proofs and formulations from Wu (2007).

We also thank the peers and students who assisted with reviewing documentation, testing edge cases, and validating the outputs.

---

## References

1. **Wu, Hsien-Chung (2007)**. *"Analysis of variance for fuzzy data"*, *International Journal of Systems Science*, 38(3), pp. 235–246. DOI: [10.1080/00207720601157997](https://doi.org/10.1080/00207720601157997).

---

## Citation

If you use `fuzzy-anova` in academic research, please cite both the foundational paper and this implementation:

```bibtex
@article{wu2007analysis,
  title   = {Analysis of variance for fuzzy data},
  author  = {Wu, Hsien-Chung},
  journal = {International Journal of Systems Science},
  volume  = {38},
  number  = {3},
  pages   = {235--246},
  year    = {2007},
  doi     = {10.1080/00207720601160822}
}

@software{fuzzy_anova_2026,
  author  = {Harshit Trivedi and Dr. Masum Raj},
  title   = {fuzzy-anova: One-Way Analysis of Variance (ANOVA) for Triangular Fuzzy Data},
  year    = {2026},
  url     = {https://github.com/AquillaRival/fuzzy-anova},
  version = {1.0.0}
}
```

---

## License

This project is licensed under the terms of the [MIT License](LICENSE).
Copyright (c) 2026 Harshit Trivedi, Dr. Masum Raj.
