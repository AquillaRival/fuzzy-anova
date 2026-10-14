# Publishing `fuzzy-anova` to PyPI: Step-by-Step Guide

This guide explains how to publish the `fuzzy-anova` package to [PyPI (Python Package Index)](https://pypi.org/) and [TestPyPI](https://test.pypi.org/).

---

## 1. Prerequisites

### Tools
Ensure you have the latest build and upload tools installed:

```bash
python -m pip install --upgrade pip build twine
```

### PyPI Accounts
1. **Production PyPI**: Create an account at [https://pypi.org/account/register/](https://pypi.org/account/register/)
2. **TestPyPI (Recommended for dry-run)**: Create an account at [https://test.pypi.org/account/register/](https://test.pypi.org/account/register/)

### API Token Setup
PyPI requires an API token instead of your account password:
1. Go to **Account settings** $\to$ **API tokens** $\to$ **Add API token**.
2. Set token scope to "Entire account" (for a brand new package) or project-specific once published.
3. Copy the token starting with `pypi-...` (save it securely).

---

## 2. Verify and Build Locally

### A. Run all automated tests
```bash
python test_fuzzy_anova.py
```
*Expected: 23 passed tests with `OK`.*

### B. Clean previous build artifacts
```powershell
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue dist, build, *.egg-info
```

### C. Build the source distribution (sdist) and universal wheel
```bash
python -m build
```
This produces:
- `dist/fuzzy_anova-1.0.0.tar.gz` (Source archive)
- `dist/fuzzy_anova-1.0.0-py3-none-any.whl` (Binary wheel)

### D. Validate metadata with Twine
```bash
python -m twine check dist/*
```
*Expected output:*
```text
Checking dist/fuzzy_anova-1.0.0-py3-none-any.whl: PASSED
Checking dist/fuzzy_anova-1.0.0.tar.gz: PASSED
```

---

## 3. Test on TestPyPI First (Recommended)

TestPyPI lets you verify the rendering, descriptions, and download without touching the production registry.

### Upload to TestPyPI:
```bash
python -m twine upload --repository testpypi dist/*
```
When prompted:
- **Username**: `__token__`
- **Password**: Your TestPyPI API token (starts with `pypi-...`)

### Test Installation from TestPyPI:
In a new virtual environment or temporary terminal:
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ fuzzy-anova
```
*(The `--extra-index-url` ensures dependencies like `numpy` are fetched from main PyPI).*

Verify it works:
```bash
fuzzy-anova --version
fuzzy-anova --demo
```

---

## 4. Publish to Production PyPI

Once you have verified everything on TestPyPI:

```bash
python -m twine upload dist/*
```

When prompted:
- **Username**: `__token__`
- **Password**: Your production PyPI API token (starts with `pypi-...`)

Your package will immediately be live at:
**[https://pypi.org/project/fuzzy-anova/](https://pypi.org/project/fuzzy-anova/)**

---

## 5. Public Installation

Anyone can now install your package with:

```bash
pip install fuzzy-anova
```

Or with optional SciPy acceleration:
```bash
pip install "fuzzy-anova[scipy]"
```

---

## 6. Releasing Future Updates

When making updates in the future:
1. Increment the version number in both:
   - [`fuzzy_anova.py`](fuzzy_anova.py): `__version__ = "1.0.1"`
   - [`pyproject.toml`](pyproject.toml): `version = "1.0.1"`
2. Add notes to [`CHANGELOG.md`](CHANGELOG.md) under a new version heading.
3. Clean and rebuild:
   ```bash
   Remove-Item -Recurse -Force dist, build, *.egg-info
   python -m build
   python -m twine upload dist/*
   ```
