# Remote Training Runbook

**Goal:** Run a long, GPU-saturating training session on a rented H100-class box, monitored remotely, to produce a strong Kairnz model.

**Prerequisites & Assumptions:**
- You have rented an H100 (or equivalent) box on a cloud provider (AWS, Lambda Labs, Vast, etc.).
- The box runs a recent Ubuntu or compatible Linux.
- You have SSH access to the box.
- You have space on the box to store training outputs (typically 10-50 GB for a multi-day run).
- You have a copy of the Kairnz source repository on your machine and on the rented box.

---

## 1. Prerequisites: Install GPU Infrastructure on the Rented Box

Before starting the container, ensure the rented box can access GPUs:

### 1.1 NVIDIA Driver

Check if the NVIDIA driver is already installed:

```bash
nvidia-smi
```

If this command fails or reports no GPUs, install the driver. On Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y nvidia-driver-550  # or latest available
```

Reboot and verify:

```bash
sudo reboot
# After reboot:
nvidia-smi
```

You should see your GPU (e.g., H100, L40S, A100) listed with full memory visible (e.g., 80 GB for H100).

### 1.2 Docker & nvidia-container-toolkit

Install Docker:

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker
```

Install nvidia-container-toolkit (so `docker run --gpus all` works):

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Verify GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi
```

You should see the same GPU list as from your host `nvidia-smi` above.

---

## 2. Build and Launch the Training Container

### 2.1 Build the Image

Clone or sync the Kairnz repository to the rented box:

```bash
git clone https://github.com/yourusername/kairnz.git
cd kairnz
```

Build the Docker image:

```bash
docker build -t kairnz .
```

This will:
- Base on `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` (so CUDA and cuDNN are pre-installed).
- Install Rust and Python tooling.
- Compile the release binaries for `kairnz-selfplay`.
- Sync the Python training environment via `uv`.

The build takes 5-10 minutes on a modern box.

### 2.2 Launch the Container

Create a directory on the host to store training outputs (so they survive container restarts):

```bash
mkdir -p /data/kairnz-training
```

Launch the container with GPU and volume mount:

```bash
docker run --gpus all -v /data/kairnz-training:/kairnz/remote-run -it kairnz bash
```

Inside the container, you should now have:
- Full GPU access (verify with `nvidia-smi`).
- The Kairnz source code at `/kairnz`.
- The work directory at `/kairnz/remote-run` (mounted from `/data/kairnz-training` on the host).

---

## 3. Launch Training

Inside the container, start the training loop with the launch script:

```bash
scripts/run_remote.sh remote-run \
  --iterations 60 \
  --filters 128 \
  --blocks 10 \
  --selfplay-games 256 \
  --selfplay-sims 400 \
  --max-batch 256 \
  --leaves-per-step 8 \
  --threads <cores>
```

Replace `<cores>` with the number of CPU cores you want to dedicate to self-play (typically half the total core count on the box; for a 40-core box, use 20).

### Command Explanation

| Argument | Meaning | Notes |
|----------|---------|-------|
| `--iterations 60` | Train for 60 iterations (full loop cycles). | Increase for longer runs; each iteration takes 1-4 hours on H100. |
| `--filters 128` | Hidden layer size in the neural network. | Default; `256` is slower but may be stronger. |
| `--blocks 10` | Number of residual blocks in the network. | Default; `14` or `20` are larger. |
| `--selfplay-games 256` | Number of games per self-play iteration. | Affects batch sizes; higher is more stable. |
| `--selfplay-sims 400` | MCTS simulations per game. | Higher is slower but stronger. |
| `--max-batch 256` | Maximum neural-network batch size. | Tune to saturate GPU; 256-512 typical for H100. |
| `--leaves-per-step 8` | Virtual-loss batch width (Phase 2). | Batches evaluations across search leaves. |
| `--threads <cores>` | CPU threads for self-play workers. | Usually half the box's core count. |

### Auto-Resume

The script includes `--resume` automatically. If the run is interrupted (network loss, spot instance preemption, etc.), simply re-run the same command:

```bash
scripts/run_remote.sh remote-run \
  --iterations 60 \
  --filters 128 \
  --blocks 10 \
  --selfplay-games 256 \
  --selfplay-sims 400 \
  --max-batch 256 \
  --leaves-per-step 8 \
  --threads <cores>
```

The script will detect the existing `best.onnx` and `best.pt` in `remote-run/` and resume from the last completed iteration, preserving all progress. The iteration counter continues from the last entry in `remote-run/metrics.jsonl`.

### Running in the Background

To keep the training alive even if your SSH session drops, use a terminal multiplexer (e.g., `tmux` or `screen`):

```bash
# In the container, if tmux is available:
tmux new-session -d -s training bash
tmux send-keys -t training "scripts/run_remote.sh remote-run --iterations 60 ..." Enter

# To reattach later:
tmux attach-session -t training
```

Or simply keep your SSH connection open (it will run as long as SSH is connected).

---

## 4. Monitor Progress

### 4.1 Web Dashboard

The training script automatically starts a web dashboard on `127.0.0.1:8080` inside the container. To view it from your local machine, establish an SSH tunnel:

```bash
# On your local machine:
ssh -L 8080:127.0.0.1:8080 user@box-ip-address
```

Then open a browser and go to:

```
http://localhost:8080
```

The dashboard displays:
- Current iteration and training stage.
- Per-iteration table with metrics: `a_score`, cumulative promotions, etc.
- Chart of `a_score` and promotion trends over iterations.
- Live log output from both `loop.py` and the MCTS engine.

The dashboard updates every 1-2 seconds as new data arrives.

### 4.2 Logs

Inside the container, check logs in the `remote-run/` directory:

```bash
# Main training loop log:
tail -f remote-run/loop.log

# Dashboard server log:
tail -f remote-run/dashboard.log
```

### 4.3 GPU Monitoring

Inside the container:

```bash
nvidia-smi -l 1
```

Watch for `GPU-Util` to stay above 80% during self-play batched inference. If it's much lower, increase `--max-batch` or `--leaves-per-step`.

---

## 5. When Training is Complete

Once the run finishes (all iterations complete or you stop it gracefully), copy the best model back to your machine:

### 5.1 From Host to Local

If using the volume mount, the model is already on the host at `/data/kairnz-training/models/best.onnx`:

```bash
# On the host (rented box), or on your local machine via SCP:
scp user@box-ip:/data/kairnz-training/models/best.onnx ~/kairnz-models/
```

### 5.2 Measure Absolute Strength

On your local machine, run the strength harness to quantify how strong the model is vs. baselines:

```bash
cargo run --release -p kairnz-selfplay --bin strength -- \
  --model ~/kairnz-models/best.onnx \
  --games 100 \
  --simulations 800
```

This reports:
- Win/draw/loss record against Random, Greedy, and plain MCTS opponents.
- Win percentage and ELO-like score.

Example output:

```
Strength vs Random:         99W, 1D, 0L (99.5% win rate)
Strength vs Greedy:         87W, 8D, 5L (87.0% win rate)
Strength vs MCTS (800 sim):  62W, 30D, 8L (61.0% win rate)
```

---

## 6. Play with the Model in the Desktop App

Load the downloaded model in the Kairnz app with GPU acceleration:

```bash
# On your local machine (assumes GPU is available):
task app-gpu
```

In the app's AI opponent settings, point the model path to `~/kairnz-models/best.onnx`. The app will use GPU inference for instant AI moves.

(If GPU is not available locally, CPU inference is slower but still functional.)

---

## 7. Important Notes

### Determinism and Reproducibility

Batched self-play inference is not bit-for-bit reproducible. The order in which leaf evaluations are batched depends on thread timing and can vary across runs. This is an **accepted tradeoff** for the throughput gain (6-8x faster GPU utilization). The trained model is deterministic and plays the same; only the exact training trajectory and iteration timings differ.

If you need reproducible training for research (e.g., ablations), use `--leaves-per-step 1 --max-batch 1` to disable batching and fall back to sequential evaluation. This will be much slower (GPU underutilized) but fully deterministic.

### Spot Instances and Preemption

If using a spot instance, the container will receive a termination signal before the instance is reclaimed. Gracefully shut down:

```bash
# In the container, Ctrl+C to stop the training loop.
```

The script will save the current best model and metrics before exiting. On your next SSH session, re-run the same command to resume.

### Network Timeouts and Long Training Runs

For multi-day runs, use a terminal multiplexer (tmux/screen) or keep a persistent SSH session. If the SSH connection drops, the container continues running (Docker keeps it alive). Simply SSH back in, attach to tmux, and check progress.

### Storage

Monitor `/kairnz/remote-run/` size as training progresses. Checkpoint files and logs can accumulate:

```bash
du -sh remote-run/
```

Typical total: 10-50 GB for a 60-iteration run, depending on game counts and shard depth.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `docker run --gpus all` fails | Ensure nvidia-container-toolkit is installed and Docker is restarted. |
| GPU not detected inside container | Verify `nvidia-smi` works on the host first; then check `nvidia-container-toolkit` installation. |
| Dashboard not reachable | Confirm SSH tunnel is active (`ssh -L 8080:...`); check `remote-run/dashboard.log` for server errors. |
| Training is very slow | Check `nvidia-smi` output; GPU utilization should be >80%. If low, increase `--max-batch` or `--leaves-per-step`. |
| Out of disk space | Check `du -sh remote-run/`; delete old checkpoints if needed (keep only the latest `best.onnx` and `best.pt`). |
| Model load fails after training | Ensure `best.onnx` was correctly copied; verify file size is >50 MB (a tiny file suggests a corrupted save). |

---

## Next Steps

After validating the model strength, consider:

1. Archive the training run (logs, metrics, checkpoints) for reference.
2. Integrate the best model into the next app release.
3. Plan ablations or longer runs with different hyperparameters (net size, game count, simulation budget) if you want to explore further optimization.
