#!/bin/bash
# Manual start of the Kria Edge Vision web app (systemd recommended instead).
cd "$(dirname "$0")" || exit 1
pkill -f "[a]pp.py --port" 2>/dev/null
sleep 1
export LD_PRELOAD=/usr/local/lib/libxcl_stub.so
setsid nohup python3 app.py --port 8080 < /dev/null > web.log 2>&1 &
echo "PID=$!"
