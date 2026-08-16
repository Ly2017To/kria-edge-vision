#!/bin/bash
# One-click start of the Vitis AI dev container for the YOLOv8n DPU project.
# Uses a persistent image with deps pre-installed so no reinstall is needed.
#
# Usage:
#   ./start_dev.sh            # start container in background (if not running)
#   ./start_dev.sh shell      # start + attach interactive bash
#   ./start_dev.sh stop       # stop the container
set -e

NAME=wod
BASE=xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.5.0.306
DEV_IMG=xilinx/vitis-ai-pytorch-cpu:wod-dev
WS=$HOME/kv260/yolov8-dpu/workspace
VAI_HOME=$HOME/Xilinx/Vitis-AI
PY=/opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin/python
PIP="pip install -i https://pypi.tuna.tsinghua.edu.cn/simple"

running() { docker ps --format '{{.Names}}' | grep -qx "$NAME"; }
exists()  { docker inspect "$1" >/dev/null 2>&1; }

# 1. ensure dev image exists (build once: install deps + commit)
if ! exists "$DEV_IMG"; then
  echo ">> Dev image $DEV_IMG not found, building it (first time only)..."
  [ -f "$WS/.confirm" ] || touch "$WS/.confirm"
  docker pull "$BASE"
  docker run -d --name wod-build --network=host -v /dev/shm:/dev/shm \
    -e USER=$(whoami) -e UID=$(id -u) -e GID=$(id -g) \
    -v "$VAI_HOME":/vitis_ai_home -v "$WS":/workspace -w /workspace \
    "$BASE" sleep infinity
  docker exec wod-build bash -c "export PATH=/opt/vitis_ai/conda/envs/vitis-ai-pytorch/bin:\$PATH && $PIP ultralytics && $PIP 'numpy==1.24.2' 'scipy==1.9.3' ninja"
  docker commit wod-build "$DEV_IMG"
  docker rm -f wod-build >/dev/null
  echo ">> Dev image saved as $DEV_IMG"
fi

# 2. stop requested
if [ "$1" = "stop" ]; then
  if running; then docker stop "$NAME"; echo ">> Container '$NAME' stopped."; else echo ">> Container '$NAME' not running."; fi
  exit 0
fi

# 3. ensure container running
if ! running; then
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d --name "$NAME" --network=host -v /dev/shm:/dev/shm \
    -e USER=$(whoami) -e UID=$(id -u) -e GID=$(id -g) \
    -v "$VAI_HOME":/vitis_ai_home -v "$WS":/workspace -w /workspace \
    "$DEV_IMG" sleep infinity
  echo ">> Container '$NAME' started from $DEV_IMG"
else
  echo ">> Container '$NAME' already running."
fi

# 4. optional interactive shell
if [ "$1" = "shell" ]; then
  exec docker exec -it "$NAME" bash
fi

echo ""
echo "Container '$NAME' ready. Quick commands:"
echo "  docker exec $NAME $PY --version                    # python in vitis-ai-pytorch env"
echo "  docker exec -it $NAME bash                         # interactive shell (then conda activate vitis-ai-pytorch)"
echo "  ./start_dev.sh stop                                # stop container"
