# M9 — Local-only Docker packaging (NOT a hosted deployment)

## Status

The source repository includes a CPU inference container and isolated local Docker Compose configuration. **An actual Linux CPU Docker image build and no-checkpoint HTTP API smoke passed in GitHub Actions** ([verified run](https://github.com/naim-munshi/ihsanvision-ai-industrial-anomaly-detection/actions/runs/39095740246)). A [later successful CPU run](https://github.com/naim-munshi/ihsanvision-ai-industrial-anomaly-detection/actions/runs/39100705980) also imported installed PyTorch/Anomalib, instantiated both PaDiM and PatchCore model classes without pretrained downloads, verified `pip check` and archived installed package versions. **Neither run loaded trained Bottle checkpoints or tested defect inference.** The owner's Mac build, private-weight Docker inference, cloud deployment and public security remain unverified. Apple Silicon/runtime compatibility must still be checked on target hardware.

The container is intended for **localhost experimentation only**. It is deliberately not exposed on a public interface. Real deployment requires a separate infrastructure/security review and HTTPS/gateway controls.

## Security and artifact boundaries

- The Docker image is built from source and intentionally contains **no** MVTec images, model checkpoints, experiment results, or private environment files.
- The Compose service binds **127.0.0.1:8080** on the Mac. Inside the isolated container, Uvicorn binds to 0.0.0.0:8080, which is necessary for a port mapping and does not itself expose the host port publicly.
- A **private, externally stored** benchmark directory is mounted **read-only** as /models; the container configuration points to both model checkpoints inside that tree.
- The service runs as UID 10001, drops Linux capabilities, enables no-new-privileges, uses a read-only root filesystem and keeps runtime Anomalib logs/tmp files in an in-memory /tmp directory.
- Compose requires IHSANVISION_API_KEY to be set. Do not use a demo/example key, do not commit it, and do not send it in chat.
- The container healthcheck now calls **GET /ready**: it checks configured nonempty local checkpoints, and if approval mode is on, verifies expected SHA-256 pins and rejects missing approved model pairs. **GET /health** remains a liveness-only endpoint. Neither endpoint deserializes a checkpoint, runs inference or proves availability under load.

## Prepare on macOS

After installing Docker Desktop and confirming the daemon responds, use a clean updated IhsanVision source checkout. The local archive folder should contain:

~~~
benchmark-bottle-seed42-2026-10-15/
  padim-bottle-seed42/model.ckpt
  patchcore-bottle-seed42/model.ckpt
~~~

From the source checkout:

~~~bash
cp .env.docker.example .env
~~~

**Edit .env privately** in your editor. Set IHSANVISION_MODELS_DIR to the absolute path to your private artifact directory and IHSANVISION_API_KEY to a new strong random secret. The .env file is ignored by Git and Docker build context. Never paste the secret in screenshots, terminal output or issues.

Then:

~~~bash
docker compose -f compose.yaml config --quiet
docker compose -f compose.yaml build
docker compose -f compose.yaml up -d
docker compose -f compose.yaml ps
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/ready
curl -fsS http://127.0.0.1:8080/models
~~~

**Build warning:** installing Anomalib/PyTorch may download large packages and may require architecture-specific wheels, RAM/disk space and a compatible Docker/CPU environment. An actual image build succeeded on GitHub's Linux CPU runner; an Apple Silicon/local Mac image build has NOT yet been demonstrated. On an Apple Silicon Mac the Linux container dependency stack may differ from the verified macOS virtual environment. Never disable certificate verification as a shortcut.

In Safari open http://127.0.0.1:8080/ and enter the private API key only into the password field to allow inference. Both local model paths should be discoverable if the bind mount was successful. Use an image you are permitted to inspect. Stop without deleting data using:

~~~bash
docker compose -f compose.yaml down
~~~

The host's external checkpoint directory is read-only to the container and is not removed by docker compose down. Do not run docker compose down --volumes unless you understand all mounts and named volumes.

## Automated Linux CPU build gate (no checkpoint weights)

A separate, heavier [GitHub Actions job](docker-cloud-build-evidence.md) has
successfully built the real CPU Docker image and started the API without
private weights. It verified live HTTP authentication, expected /ready 503
without checkpoints, dashboard routes, nonroot execution, loopback binding
and container confinement. This is genuine Linux container execution, **not**
checkpoint-backed inference and not full P8. Full real on-Mac/target-machine
inference/restart steps remain required below.

## Automated static Compose gate

The Python quality workflow now adds a `container-config` job that parses
`compose.yaml` with Docker Compose and checks localhost host binding, nonroot
UID, read-only root filesystem, dropped capabilities and read-only /models
mount. It uses dummy CI-only environment variables and **does not build the
PyTorch/Anomalib image or invoke model inference**. This static guard does not
satisfy the real target-machine Docker acceptance gates below.

## Verification gates before calling M9 complete

1. Docker daemon responds; compose config resolves the absolute model directory and key without printing any secret publicly.
2. Image builds on the target architecture and the readiness healthcheck becomes healthy; no model weights or input images appear in image layers. The readiness response explicitly marks inference_verified=false.
3. Both checkpoints are visible and **actual inference**, including heatmap, succeeds via the container on normal and defective images.
4. Container filesystem/UID/capability/read-only mount restrictions work without breaking runtime.
5. Failure and restart behavior, startup/cold-load, memory use, and graceful shutdown are documented with real outputs.
6. For a *remote* deployment, independently add gateway TLS, pre-upload authentication/size limits, a managed secrets store, distributed admission/quotas, sanitized logs and alerts, change management, environment-specific readiness probes and verified rollback procedures.

Source-code static unit tests are **not** equivalent to running Docker. Do not publish a production endpoint until all relevant security, licensing and operations gates pass.
