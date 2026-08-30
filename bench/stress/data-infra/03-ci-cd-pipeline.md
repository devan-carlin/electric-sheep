# CI/CD Pipeline

**Category:** Data & infrastructure
**Target:** GitHub Actions, matrix builds, caching, artifact management

---

## Prompt

Create a complete GitHub Actions CI/CD pipeline for a Python library with the following requirements:

**Repository structure:**
```
mylib/
├── src/mylib/
├── tests/
├── pyproject.toml
├── .github/workflows/
│   ├── ci.yml
│   ├── release.yml
│   └── nightly.yml
└── CHANGELOG.md
```

**Workflow 1: `ci.yml`** — Runs on every push and PR

- Matrix: Python 3.10, 3.11, 3.12, 3.13 × Ubuntu, macOS, Windows
- Steps:
  1. Cache `pip` dependencies (keyed on `pyproject.toml` hash)
  2. Install dependencies with `uv` or `pip`
  3. Run `ruff check` (linting)
  4. Run `mypy --strict` (type checking)
  5. Run `pytest -v --cov=mylib --cov-report=xml` (tests + coverage)
  6. Upload coverage to Codecov
  7. Upload test results as artifacts (JUnit XML)
- PR comments: Post a comment with test results summary and coverage delta
- Skip matrix jobs if only `.md` files changed (path filtering)

**Workflow 2: `release.yml`** — Manual trigger or tag push

- Steps:
  1. Validate version matches git tag
  2. Build distribution (`python -m build`)
  3. Publish to TestPyPI on `--test` flag, PyPI on real tags
  4. Create GitHub Release with changelog excerpt
  5. Sign artifacts with Sigstore (cosign)

**Workflow 3: `nightly.yml`** — Runs daily at 2 AM UTC

- Steps:
  1. Install with `--pre` to get latest pre-release dependencies
  2. Run full test suite against latest dependency versions
  3. Run additional compatibility tests (e.g., against next Python preview)
  4. Post results to a GitHub Discussion

**Requirements:**

- Use `actions/setup-python` with caching
- Use `concurrency` groups to cancel redundant runs on the same branch
- Use `env` for shared environment variables
- Use reusable workflows where appropriate (e.g., a shared `test.yml` called by both `ci.yml` and `nightly.yml`)
- Proper error handling with `continue-on-error` for non-critical steps
- Secrets management for PyPI token, Codecov token, Sigstore

**Constraints:**

- No third-party Actions beyond: actions/checkout, actions/setup-python, actions/upload-artifact, actions/download-artifact, actions/create-release, codecov/codecov-action
- Must work with `pyproject.toml` (no `setup.py`, no `requirements.txt`)
- Include a `.github/CODEOWNERS` file

Produce all workflow files with complete working code. No placeholders, no TODOs.
