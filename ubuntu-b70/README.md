# Ubuntu B70 — Lenovo P620

**Hardware:** 4× Intel Arc Pro B70 (31.89 GiB each, `xe` driver)
**OS:** Ubuntu 26.04 LTS (Kernel 7.0.0-29)
**CPU:** AMD Threadripper PRO 3945WX (12C/24T)
**RAM:** 247 GiB
**Storage:** Samsung 990 PRO 2TB NVMe
**oneAPI:** 2026.1.0, Level Zero driver 26.22.38646.7

## vLLM

See [`vllm/`](./vllm/) for build scripts, model configs, patches, and deployment guides.

Deployment order:

1. [Install Prerequisites](./vllm/01-install-prerequisites.sh)
2. [Setup Project Directory](./vllm/02-setup-project-directory.sh)
3. [Build vLLM XPU](./vllm/03-build-vllm-xpu.sh)
4. [Download Models + Configs](./vllm/04-download-models.sh)
5. [MoE Patch (if using 35B-A3B)](./vllm/05-patch-vllm-moe-qzeros.sh)
6. [Model Configs](./vllm/06-model-configs.md)
7. [Server Config Baseline](./vllm/07-server-config-baseline.md)
8. [Deployment Guide](./vllm/08-vllm-deployment-guide.md)

## llama.cpp

See [`llama/`](./llama/) for build scripts, deployment guide, and model-specific configs.

Deployment order:

1. [Setup Project](./llama/01-setup-project.sh)
2. [Build SYCL](./llama/02-build-llama-cpp.sh)
3. [Deployment Guide](./llama/03-llama-cpp-deployment-guide.md)

## Utilities

See [`utilities/`](./utilities/) for cross-project tools (build-all, model download, GPU power limits, INT4 conversion).
