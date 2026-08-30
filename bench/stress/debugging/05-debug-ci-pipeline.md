# Debug a Failing CI Pipeline

**Target Capability:** Reading CI logs, identifying root cause, fixing YAML configuration.

Tests whether a model can diagnose a broken GitHub Actions workflow from the error output.

---

## Prompt

```
The following GitHub Actions workflow is failing. Below is the workflow YAML and the relevant portion of the CI log.

Identify the root cause(s) of the failure and provide the corrected workflow YAML.

### Workflow (`.github/workflows/ci.yml`):

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Cache pip
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ matrix.python-version }}-${{ hashFiles('**/poetry.lock') }}

      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install poetry
          poetry install

      - name: Run tests
        run: |
          poetry run pytest tests/ --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          fail_ci_if_error: true

      - name: Build package
        run: |
          poetry build

      - name: Publish to Test PyPI
        if: github.ref == 'refs/heads/main'
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.TEST_PYPI_API_TOKEN }}
          repository-url: https://test.pypi.org/legacy/
```

### CI Log (failing job, Python 3.12):

```
Run poetry run pytest tests/ --cov=src --cov-report=xml
============================= test session starts ==============================
platform linux -- Python 3.12.4, pytest-8.2.2, pluggy-1.5.0
rootdir: /home/runner/work/myproject/myproject
configfile: pyproject.toml
plugins: cov-5.0.0, anyio-4.3.0
collected 47 items

tests/test_api.py::test_create_user PASSED                              [  2%]
tests/test_api.py::test_get_user PASSED                                 [  4%]
tests/test_api.py::test_delete_user PASSED                              [  6%]
tests/test_db.py::test_connection_pool PASSED                            [  8%]
tests/test_db.py::test_migration_up PASSED                               [ 10%]
tests/test_db.py::test_migration_down FAILED                             [ 12%]
tests/test_models.py::test_user_serialization PASSED                     [ 14%]
tests/test_models.py::test_user_validation PASSED                        [ 17%]
tests/test_models.py::test_user_validation_email FAILED                  [ 19%]
tests/test_auth.py::test_jwt_encode_decode PASSED                        [ 21%]
tests/test_auth.py::test_jwt_expiry PASSED                               [ 23%]
tests/test_auth.py::test_refresh_token_rotation FAILED                   [ 25%]

=================================== FAILURES ===================================
___________________________ test_migration_down ________________________________

    def test_migration_down():
        runner = CliRunner()
        result = runner.invoke(downgrade, ['-1'])
>       assert result.exit_code == 0
E       AssertionError: assert 2 == 0
E        +  where 2 = <Result SYSTEM_EXIT>.exit_code

tests/test_db.py:45: AssertionError
________________________ test_user_validation_email ____________________________

    def test_user_validation_email():
        user = User(email="user@domain", name="Test")
>       assert user.email == "user@domain.com"
E       AssertionError: assert 'user@domain' == 'user@domain.com'

tests/test_models.py:38: AssertionError
_______________________ test_refresh_token_rotation ____________________________

    def test_refresh_token_rotation():
        token1 = create_refresh_token(user_id="u1")
        token2 = refresh_token(token1)
        with pytest.raises(TokenExpiredError):
>           refresh_token(token1)  # Old token should be invalid
E           Failed: DID NOT RAISE <class 'auth.exceptions.TokenExpiredError'>

tests/test_auth.py:62: Failed

---------- coverage: platform linux, python 3.12.4 -----------
Coverage XML written to file coverage.xml
========================= 3 failed, 10 passed in 4.21s =========================
Error: Process completed with exit code 1.
```

Your response must include:
1. The root cause of each test failure (3 failures, explain each).
2. Whether the CI workflow YAML itself has any issues (separate from the test failures).
3. The corrected workflow YAML if any fixes are needed.
```
