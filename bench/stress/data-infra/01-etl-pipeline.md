# ETL Pipeline

**Category:** Data & infrastructure
**Target:** Data transformation, error handling, incremental processing

---

## Prompt

Build a Python ETL pipeline that processes raw CSV export files from a SaaS analytics platform and loads cleaned data into a SQLite database for reporting.

**Structure:**

```
etl-pipeline/
├── etl/
│   ├── __init__.py
│   ├── main.py            # Entry point, CLI with argparse
│   ├── extract.py         # Read CSV files, handle encoding issues
│   ├── transform.py       # Clean, validate, enrich data
│   ├── load.py            # Upsert into SQLite with proper schema
│   ├── schema.py          # Database schema, migrations
│   └── config.py          # Settings, field mappings
├── tests/
│   ├── __init__.py
│   ├── test_extract.py
│   ├── test_transform.py
│   └── test_load.py
├── sample-data/
│   ├── events.csv         # Sample raw data (with intentional dirty rows)
│   └── users.csv          # Sample raw data
├── pyproject.toml
└── README.md
```

**Input CSV schema (events.csv):**

```
event_id,user_id,event_type,timestamp,properties,revenue,country,device
evt_001,user_42,purchase,2024-01-15 14:30:00,"{"plan":"pro"}",49.99,US,desktop
evt_002,user_17,page_view,2024-01-15 14:31:22,"{"page":"/pricing"}",0,GB,mobile
evt_003,,purchase,2024-01-15 14:32:00,"{"plan":"basic"}",19.99,,desktop
evt_004,user_42,signup,not-a-date,"{"referral":"twitter"}",0,US,desktop
evt_005,user_99,purchase,2024-01-15 14:35:00,"{invalid json",99.99,JP,tablet
```

**Requirements:**

- **Extract:** Read CSV files with robust error handling (bad encoding, missing fields, extra columns)
- **Transform:**
  - Skip rows with missing `user_id` (log warning)
  - Parse and validate timestamps (skip rows with invalid dates)
  - Parse `properties` JSON (store as text if invalid, log warning)
  - Normalize `country` to uppercase (empty → "UNKNOWN")
  - Convert `revenue` to integer cents (49.99 → 4999)
  - Add `processed_at` timestamp and `source_file` metadata
- **Load:**
  - SQLite database with proper schema (indexes on user_id, event_type, timestamp)
  - Upsert logic (if `event_id` exists, update the row)
  - Transactional inserts (batch of 1000 rows per transaction)
  - Track processing statistics (rows read, rows loaded, rows skipped, errors)
- **CLI:**
  - `python -m etl --input-dir ./data --output analytics.db --dry-run`
  - `python -m etl --input-dir ./data --output analytics.db --resume` (skip already-processed files)
  - Progress reporting to stdout

**Constraints:**

- Python 3.10+, stdlib only (no pandas, no sqlalchemy)
- Must handle: empty files, files with only headers, files with BOM encoding
- Tests must use the sample data with intentional dirty rows
- `--dry-run` should validate and report without writing to database

Produce all files with complete working code. No placeholders, no TODOs.
