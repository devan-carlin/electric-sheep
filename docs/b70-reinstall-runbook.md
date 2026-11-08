# B70 Reinstall Runbook

Rebuild the Qwen3.8-Flash-Next serving stack on a fresh Ubuntu install, on the
4x Intel Arc Pro B70 rig. Written from the live state of the `ai-server` host on
2026-11-08.

This file lives in `electric-sheep/docs/`, which is pushed to GitHub. It survives
a disk wipe. Copy it off before wiping regardless.

## Scope

Target end state: serve `qwen-256k` on `:8000` via vLLM (XPU fork) on 4x B70,
serving the Intel AutoRound and heretic-2 FP8PLE checkpoints.

Out of scope: ComfyUI, the vn_pipeline ingest/watch services, the 27B config-B
split. Those are separate workloads.

## 1. What to copy off this host before wiping

Two things are irreplaceable and NOT on GitHub: the model weights and the PLE
table. Everything else is either in a git remote or rebuildable.

### 1a. Model weights (copy to the primary machine)

These are the only two models to keep. Both are custom quants with NO Hugging
Face download metadata. They cannot be re-downloaded. Copy them.

| Path (real, via `electric-sheep/models` symlink) | Size | Files |
|---|---|---|
| `/mnt/data/models/Intel-Qwen3.8-Flash-Next-W4A16-AutoRound` | 169 G | 29 |
| `/mnt/data/models/densohax-Qwen3.8-Flash-Next-heretic2-W4A16-FP8PLE` | 73 G | 334 |

Copy with rsync so it is resumable over the SSH link:

```bash
# run FROM the primary machine, pulling over ssh
rsync -aH --info=progress2 \
  dc@ai-server:/mnt/data/models/Intel-Qwen3.8-Flash-Next-W4A16-AutoRound \
  /path/to/models/
rsync -aH --info=progress2 \
  dc@ai-server:/mnt/data/models/densohax-Qwen3.8-Flash-Next-heretic2-W4A16-FP8PLE \
  /path/to/models/
```

### 1b. PLE table (needed by the heretic-2 FP8PLE launcher)

`/mnt/data/ple_cache/ple_table_qwen4exp.pt` is 96 G. The heretic-2 launcher
mmaps it and hard-fails if absent (`start-qwen-256k-heretic2-fp8ple-vllm.sh:104`).

Two options:
- Copy it (96 G, fastest restore), OR
- Rebuild it after reinstall with `electric-sheep/vllm/phase_b_ple_table_prep.py`
  (target-independent, but expensive compute).

If you copy it, keep the exact path `/mnt/data/ple_cache/ple_table_qwen4exp.pt`
or set `PLE_TABLE_PATH` to wherever you put it.

### 1c. Secrets and keys (plaintext, gitignored)

The live secret values are captured in plaintext in:

```
electric-sheep/docs/SECRETS.local.md
```

That file is gitignored (`docs/SECRETS.local.md` + `*.local.md` in `.gitignore`),
so it can never be committed or pushed. This repo is PUBLIC on GitHub. The secret
values are deliberately NOT inlined into this runbook body for that reason: this
file gets pushed. The values live only in the gitignored file. Copy
`SECRETS.local.md` off this host yourself and secure it.

Contents of `SECRETS.local.md`:

| Item | What it is | Re-add action |
|---|---|---|
| GitHub token | `gho_` OAuth, scopes `repo,user,workflow,gist,read:org` (full-account) | `gh auth login` with it, or mint a scoped fine-grained PAT instead |
| SSH private key | `~/.ssh/suze_smallserver` | Restore to `~/.ssh/`, `chmod 600`, or regenerate + re-add pub to GitHub |
| SSH public key | `~/.ssh/suze_smallserver.pub` | The counterpart pub key |
| `authorized_keys` | inbound SSH allow-list | Restore if inbound SSH access is wanted |
| `known_hosts` | host trust | Restore to avoid host-key prompts |
| ComfyUI env | `comfyui/scripts/.env` | Only if ComfyUI survives. Out of scope here |

No Hugging Face token file exists (`~/.cache/huggingface/huggingface--cache/token`
absent). The models were pulled without a stored token or are gated/custom.

### 1d. System units and sudoers (recreate after reinstall)

These are NOT in the repos. Recreate them:

- `/etc/systemd/system/arc-b70-power-caps.service` (applies 180 W GPU caps at
  boot). Recreated by `serve/install-power-caps-service.sh` (run with sudo). That
  installer also writes the `/etc/sudoers.d/dc-gpu-persist` NOPASSWD drop-in.
- `vn-ingest.service` + `vn-watch.service` (source in `vn_pipeline/systemd/`).
  Only if you keep the vn_pipeline workload. They reference `/home/dc/comfyui`
  and `/home/dc/vn_pipeline`.

### 1e. Do NOT bother copying

- `vllm-fresh-venv` (9 G). It was built from a `/tmp/vllm-verify` checkout that
  is already gone. Rebuild it from source (section 3). It is not portable.
- The other 25 model dirs under `/mnt/data/models`. Dropped per the prune.

## 2. Fresh OS prerequisites

The vLLM XPU stack needs the Intel GPU stack. None of this is in git.

1. Ubuntu Server (this host runs 26.04).
2. Intel GPU driver (the `xe` driver for Battlemage / Arc Pro B70, Xe2).
3. oneAPI 2026.1 base toolkit: compiler (`icpx`), DNNL, CCL. Installs to
   `/opt/intel/oneapi/` with `setvars.sh`. The build script and every launcher
   source `/opt/intel/oneapi/setvars.sh`. Without it nothing builds or serves.
4. Python 3.12 (the venv is `python3.12`).
5. Build tools: `git`, `cmake`, `ninja`, `setuptools`, `setuptools-rust`,
   `setuptools-scm`, `psutil`.
6. 4x B70 visible to Level Zero (`ze`/`sycl` device count = 4). The launchers
   hardcode `--tensor-parallel-size 4` and `ZE_AFFINITY_MASK=0,1,2,3`.

Verify before building:

```bash
source /opt/intel/oneapi/setvars.sh
icpx --version                 # compiler present
python3.12 --version           # 3.12.x
sycl-ls 2>/dev/null | grep -c GPU   # expect 4
```

## 3. Rebuild the vLLM XPU venv

The venv is a source build of the vLLM XPU fork plus three layers of local
patches. This is the long pole (kernel compile is 1h+ on a 12-core CPU).

### 3a. Clone the repos

```bash
cd ~
git clone https://github.com/devan-carlin/electric-sheep.git
git clone https://github.com/devan-carlin/arc-b70-vllm-optimized.git
git clone https://github.com/devan-carlin/neon-demon.git
```

### 3b. Build vLLM + kernels with the B70 patches

`arc-b70-vllm-optimized/scripts/build-vllm-xpu.sh` does the whole chain:
torch XPU, clone `vllm` + `vllm-xpu-kernels` from `vllm-project`, build the
SYCL/oneDNN/CUTLASS kernels, apply `moe-topk-16-32-64.patch` and
`mamba-xpu-ptr.patch`, build vLLM, restore the local (patched) kernels over the
pinned wheel, force `triton-xpu`, install the GGUF plugin.

```bash
cd ~/arc-b70-vllm-optimized
VENV_DIR=$HOME/vllm-fresh-venv BUILD_DIR=$HOME/vllm-src \
  bash scripts/build-vllm-xpu.sh
```

Note: the build script defaults `VENV_DIR` to `$HOME/vllm-xpu/.venv`. The live
launchers expect `$HOME/vllm-fresh-venv` (`QWEN256K_VENV` override). Pass
`VENV_DIR=$HOME/vllm-fresh-venv` so the launchers find it, or export
`QWEN256K_VENV` to wherever you build it.

### 3c. Apply the qwen4exp port (the 9 in-place venv edits)

The Qwen3.8-Flash-Next support is 9 files edited in-place inside the venv's
site-packages. They are NOT in the wheel RECORD and NOT in the vLLM git tree.
They are snapshotted in `arc-b70-vllm-optimized/qwen4exp/` with a checksum
manifest. A rebuild loses them. Restore them:

```bash
~/arc-b70-vllm-optimized/qwen4exp/restore.sh $HOME/vllm-fresh-venv
```

It copies the 17-file snapshot over site-packages and verifies every checksum.
The files it restores (from `MANIFEST.sha256`): `qwen4_exp.py`, `qwen4_exp_mtp.py`,
`qwen3_next.py`, `registry.py`, the GDN linear-attn layer, the WNA16 linear
scheme, the mixed-precision XPU linear kernel, `flash_attn.py`,
`llm_base_proposer.py`, `mamba_utils.py`, and the config/configs files.

### 3d. Apply the worker-affinity patch

```bash
python ~/arc-b70-vllm-optimized/patches/patch_vllm_worker_affinity_venv.py
```

Per-worker `ZE_AFFINITY_MASK` injection, anchored to `$HOME/vllm-fresh-venv`.
Off by default (`B70_WORKER_AFFINITY=0`), opt-in per launch. Needed for the
multi-GPU allreduce fix (cookbook `DUAL-B70-TP2.md`).

### 3e. Verify the venv

```bash
source $HOME/vllm-fresh-venv/bin/activate
python -c "import torch,vllm,vllm_xpu_kernels; \
  print('vllm',vllm.__version__); \
  print('xpu devices',torch.xpu.device_count()); \
  import vllm_xpu_kernels._C as c; \
  print('gemm ops',[o for o in dir(c) if 'gemm' in o])"
```

Expect `xpu devices 4` and a non-empty gemm op list.

## 4. Restore models and PLE table

```bash
# models: put them where the launchers look, and re-create the symlink
mkdir -p /mnt/data/models
# (rsync the two model dirs into /mnt/data/models/ here)
ln -s /mnt/data/models ~/electric-sheep/models

# PLE table: restore the copy, or rebuild it
mkdir -p /mnt/data/ple_cache
# cp /path/to/backup/ple_table_qwen4exp.pt /mnt/data/ple_cache/
# OR rebuild:
#   source $HOME/vllm-fresh-venv/bin/activate
#   python ~/electric-sheep/vllm/phase_b_ple_table_prep.py
```

The `/mnt/data` mount must exist at the exact path. Every launcher hardcodes
`/mnt/data/models/...` and `/mnt/data/ple_cache/...`. If the reinstall puts the
data disk elsewhere, either mount it at `/mnt/data` or export `QWEN256K_MODEL`
and `PLE_TABLE_PATH`.

## 5. Serve

```bash
# Intel AutoRound (the "intel" model)
bash ~/electric-sheep/serve/start-qwen-256k-intel-autoround-vllm.sh

# heretic-2 FP8PLE (needs the PLE table present)
bash ~/electric-sheep/serve/start-qwen-256k-heretic2-fp8ple-vllm.sh
```

Both serve alias `qwen-256k` on `:8000`, 256K ctx, fp8 KV, TP4. They are mutually
exclusive on the GPUs. `serve/switch-config.sh` toggles the rig between the
all-4 125B config and the ComfyUI+27B split config.

Smoke test:

```bash
curl -s localhost:8000/v1/models | jq .data[].id   # expect qwen-256k
```

## 6. Re-add secrets after install

1. `gh auth login` (paste the token, or mint a fresh scoped PAT).
2. Restore `~/.ssh/suze_smallserver` keypair, or generate a new one and add the
   pub to GitHub.
3. `sudo bash ~/electric-sheep/serve/install-power-caps-service.sh` (installs the
   GPU power-cap unit + the NOPASSWD sudoers drop-in).
4. If keeping vn_pipeline: install `vn_pipeline/systemd/vn-{ingest,watch}.service`.

## 7. What is NOT preserved by this runbook

- ComfyUI and its 449 G of models. Separate workload, out of scope.
- The other 25 model dirs (pruned). Re-downloadable from HF except the two kept.
- The `vllm-fresh-venv` binary. Rebuilt in section 3, not copied.
- Any `/home/dc` state not in a git remote or listed in section 1.

## Reference: the two kept models

| Model | HF repo id | Notes |
|---|---|---|
| `Intel-Qwen3.8-Flash-Next-W4A16-AutoRound` | (no HF download metadata on disk) | Intel AutoRound W4A16. 169 G, 29 files. Served by the intel-autoround launcher |
| `densohax-Qwen3.8-Flash-Next-heretic2-W4A16-FP8PLE` | (no HF download metadata on disk) | Abliterated heretic-2, W4A16 + FP8 PLE table. 73 G, 334 files. Needs `ple_table_qwen4exp.pt` |

Neither carries Hugging Face download metadata. Treat both as copy-only. If you
have the original HF repo ids or the AutoRound quant recipe, record them here.
