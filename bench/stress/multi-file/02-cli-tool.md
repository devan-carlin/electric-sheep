# CLI Tool with Subcommands

**Category:** Multi-file project
**Target:** Modular architecture, argument parsing, config management

---

## Prompt

Build a CLI tool called `logpipe` that parses and analyzes application log files. Structure:

```
logpipe/
├── logpipe/
│   ├── __init__.py
│   ├── __main__.py        # Entry point, argument parsing
│   ├── cli.py             # Click/Typer subcommand definitions
│   ├── parser.py          # Log line parsing (regex, structured)
│   ├── analyzer.py        # Statistics, aggregation
│   ├── formatter.py       # Output formatting (table, json, csv)
│   └── config.py          # Config file loading (~/.logpipe.yaml)
├── tests/
│   ├── __init__.py
│   ├── test_parser.py
│   └── test_analyzer.py
├── pyproject.toml
└── README.md
```

**Subcommands:**

- `logpipe parse <file>` — parse and display log entries
- `logpipe stats <file>` — show error rates, latency percentiles, top endpoints
- `logpipe filter <file> --level ERROR --after "2024-01-01"` — filter by criteria
- `logpipe tail <file> --lines 50` — show last N entries

**Requirements:**

- Support common log formats (Apache/Nginx combined, JSON structured, syslog)
- Config file for defaults (output format, date format, timezone)
- Colorized terminal output (red for errors, yellow for warnings)
- Progress bar for large files (>100MB)
- Unit tests with sample log data embedded in test fixtures
- `pyproject.toml` with proper metadata, dependencies, and entry points

**Constraints:**

- Use `typer` for CLI (not argparse)
- No external dependencies beyond typer, rich, pyyaml, pydantic
- Must handle malformed log lines gracefully (skip with warning)
- Tests must pass with `pytest`

Produce all files with complete working code. No placeholders, no TODOs.
