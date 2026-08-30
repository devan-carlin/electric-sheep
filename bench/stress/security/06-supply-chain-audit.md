# Supply Chain Audit Tool

Build a Python CLI tool that audits a project's dependency lockfile for security risks.

**Requirements:**

The tool should accept a lockfile path and output a JSON report:

```bash
python audit.py --lockfile package-lock.json --output report.json
```

**Supported lockfile formats:**

1. `package-lock.json` (npm)
2. `poetry.lock` (Python, TOML format)
3. `Gemfile.lock` (Ruby)

**Checks to perform:**

1. **Known CVEs** — Cross-reference package names and versions against a bundled list of known vulnerabilities (include a small embedded database of ~20 real CVEs for testing).
2. **Typosquatting detection** — Flag packages with names similar to popular packages (e.g., `reqeusts` vs `requests`, `expresss` vs `express`). Use a hardcoded list of 30 well-known packages.
3. **Deprecated packages** — Flag packages marked as deprecated in the embedded database.
4. **Version pinning** — Warn if any dependency uses a loose range (`^`, `~`, `>=`) instead of an exact version.
5. **Duplicate packages** — Detect if the same package appears at multiple versions (potential prototype pollution vector).

**Output format:**

```json
{
  "lockfile": "package-lock.json",
  "total_packages": 142,
  "findings": [
    {
      "severity": "critical|high|medium|low",
      "type": "cve|typosquat|deprecated|loose-version|duplicate",
      "package": "expresss",
      "version": "4.17.1",
      "detail": "Possible typosquat of 'express'",
      "cve_id": null
    }
  ],
  "summary": {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3
  }
}
```

**Constraints:**

- Single file, standard library only (no `pip install` required).
- Must handle all three lockfile formats.
- The embedded CVE/typosquat database should be a Python dict at the top of the file (easy to extend).
- Exit code 0 if no critical/high findings, exit code 1 otherwise.
- Include `--quiet` flag (only print findings, no summary) and `--min-severity` flag.
