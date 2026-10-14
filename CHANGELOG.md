# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-14

### Added
- Complete One-Way Fuzzy ANOVA implementation based on Wu (2007).
- Alpha-cut (h-level set) method for converting triangular fuzzy numbers to intervals.
- Lower Bound and Upper Bound ANOVA table generation at any h-level (Wu Table 2 & Table 3).
- Quadratic boundary coefficient derivation ($A\cdot h^2 + B\cdot h + C$).
- Interval sets $K_{H0}$ and $K_{H1}$ computation via quadratic constraint optimization.
- Optimistic degree ($\lambda$), pessimistic degree ($h_H$), and overall confidence degree metrics.
- High-performance pure-Python Snedecor's F-distribution fallback (zero mandatory dependency on SciPy).
- Optional acceleration with SciPy if installed.
- Comprehensive input validation for non-numeric types, NaN/Inf, inverted bounds ($L \le \text{Core} \le U$), and degrees of freedom.
- Command-line interface (`fuzzy-anova`) with `--demo`, `--version`, and `--json` support.
- Fully typed public API with `py.typed` (PEP 561 compliance).
- Automated test suite with 23 unit tests verifying numerical benchmark accuracy and edge cases.

### Credits & Authorship
- **Harshit Trivedi**: Software architecture, core Python implementation, numerical algorithms, test suite, and packaging.
- **Dr. Masum Raj**: Mathematics Professor, academic mentorship, theoretical formulation verification, and validation against Wu (2007).
