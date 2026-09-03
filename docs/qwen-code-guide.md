# qwen-code Quick User Guide

Terminal AI coding agent (Qwen team, gemini-cli lineage) pointed at the local
vLLM server. Zero cloud. Installed 2026-09-01.

## What it is

- `qwen-code` is an interactive coding agent for the terminal. It reads your
  repo, edits files, runs commands, and answers questions.
- On this box it talks to the local vLLM endpoint (`qwen-256k` on :8000), so
  every token is generated on the 4x Arc B70. No API bill, no data leaves the
  tailnet.

## The two commands

| Command | What it does |
|---|---|
| `qwen-local` | Interactive session against local vLLM (default model `qwen-256k`) |
| `qwen-local -p "..."` | One-shot prompt, prints the answer, exits |

`qwen-local` is a wrapper that sets the local base URL + key and the model for
you. The raw command is `qwen` (with `--auth-type openai` and the env vars set).

## Start a session

```bash
cd /path/to/your/project
qwen-local
```

You get a prompt. Type a task in plain language, e.g.:

- `Find the bug in auth.py and fix it`
- `Add a unit test for the parse_config function`
- `Explain what this module does`
- `Refactor this function to be shorter`

Press Enter to send. It will read files, propose edits, and run commands.

## One-shot (no interactive session)

```bash
qwen-local -p "Summarize the README in 3 bullets"
qwen-local -p "Write a Python script that renames all .jpg files to .png"
```

Useful for scripting or quick questions. Add `--output-format json` to get
machine-readable output.

## Switching models

- Default is `qwen-256k` (the big 256K-context model).
- For a cheap/fast comparison, point at the llama.cpp gemma endpoint:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8089/v1 qwen-local -m gemma -p "your prompt"
```

- In an interactive session, use `/model` to switch mid-session.

## Useful in-session commands

Type these at the prompt (start with `/`):

| Command | What it does |
|---|---|
| `/help` | List all commands |
| `/model` | Show / switch the active model |
| `/clear` | Clear the conversation history |
| `/memory` | Manage project memory |
| `/init` | Initialize project context (writes a project memory file) |
| `/permissions` | Review / change tool permissions |
| `/mcp` | Manage MCP servers |
| `/extensions` | Manage extensions |
| `/review` | Run a code review |
| `/quit` | Exit the session |

## Approving actions / permissions

- By default qwen-code asks before running shell commands or editing files.
  In an interactive session you approve or deny each action as it comes up.
- In one-shot mode (`-p`) there is no one to approve, so any action not on
  the allow-list is silently denied. The local `~/.qwen/settings.json` already
  allow-lists the common tools so one-shot runs can edit files and run
  commands:

```json
"permissions": {
  "allow": ["read_file", "glob", "grep", "ls", "edit", "write_file", "run_shell_command"]
}
```

- The tool names are exact: `edit`, `write_file`, `read_file`, `glob`, `grep`,
  `ls`, and `run_shell_command` (not `shell`). If a new action type is denied,
  add its tool name to `permissions.allow`.
- Use `/permissions` in an interactive session to review or change these.

## Pointing it at a different endpoint

The wrapper reads two env vars (with local defaults). Override either:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8000/v1   # vLLM (default)
OPENAI_BASE_URL=http://127.0.0.1:8089/v1   # llama.cpp gemma
OPENAI_API_KEY=keyless-local               # any non-empty string when keyless
```

The model name must be the **bare** served name (`qwen-256k`, `gemma`), not a
prefixed form.

## Gotchas

- **Model name is bare.** `qwen-256k`, not `openai/qwen-256k`. The prefixed
  form makes vLLM return a 404.
- **If you see a 401 with an `aliyun.com` link**, the request went to the
  default cloud endpoint, not your local server. The `OPENAI_BASE_URL` env var
  is not set in that shell. Use `qwen-local` (which sets it) or export it.
- **One-shot mode denies unlisted actions.** With `-p` there is no one to
  approve a prompt, so any tool not in `permissions.allow` is silently denied
  (the model will say "permission was declined"). The local settings already
  allow `edit`, `write_file`, and `run_shell_command`. If a new action is
  denied, add its exact tool name to `permissions.allow` in
  `~/.qwen/settings.json`.
- **New shell can't find `qwen-local`?** Add this to `~/.bashrc`:
  `export PATH="$HOME/.local/bin:$PATH"`.
- **It's a reasoning model.** Long answers may take a while; the model thinks
  before responding. For quick tasks, the one-shot `-p` form is fastest.

## Where things live

| Thing | Path |
|---|---|
| Wrapper | `~/.local/bin/qwen-local` |
| CLI | `~/.local/bin/qwen` (v0.22.3) |
| Node runtime | `~/.local/node` (v24.20.0) |
| Settings | `~/.qwen/settings.json` |
| Backlog item | `electric-sheep/BACKLOG.md` (qwen-code section) |
| Setup notes | repo memory `qwen-code-setup.md` |

## Try this first

Pick a small scratch repo and give it a real task. This exact example is
verified to work end-to-end (fix + test + run the tests):

```bash
mkdir -p /tmp/qwen-test && cd /tmp/qwen-test
printf 'def add(a, b):\n    return a - b\n' > calc.py
qwen-local -p "calc.py has a bug (subtraction instead of addition). Fix it, add test_calc.py, then run 'python3 -m unittest -v' to verify."
```

It flips `-` to `+`, writes `test_calc.py`, runs the tests, and reports they
pass. That's the whole loop: describe the problem, it reads the code, edits it,
runs the verification, and reports back.