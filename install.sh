#!/usr/bin/env bash
# Install iris-listen (Qwen3-ASR-1.7B) as a user service. Safe to run again.
#
#   ./install.sh
#   LISTEN_HOSTS="127.0.0.1 <vpn address>" ./install.sh   # also listen on the VPN
#
# Needs Linux, an NVIDIA GPU with ~5 GB free, uv and python3. Takes ~10 GB of
# disk: the venv here (.venv, torch with its CUDA) and the model (3.9 GB, in
# LISTEN_MODEL_DIR). Every request needs the bearer token in LISTEN_TOKEN_FILE;
# the first install makes one.
set -euo pipefail
PORT=${LISTEN_PORT:-7861}
HOST_ARGS=$(printf -- '--host %s ' ${LISTEN_HOSTS:-127.0.0.1})
MODEL=${LISTEN_MODEL_DIR:-$HOME/.local/share/iris-listen/models/qwen3-asr-1.7b}
TOKEN_FILE=${LISTEN_TOKEN_FILE:-$HOME/.config/iris-listen/token}
HERE="$(dirname "$(readlink -f "$0")")"
cd "$HERE"

if [ ! -s "$TOKEN_FILE" ]; then
  mkdir -p "$(dirname "$TOKEN_FILE")"
  (umask 077 && python3 -c 'import secrets; print(secrets.token_urlsafe(32))' > "$TOKEN_FILE")
  echo "Made a token in $TOKEN_FILE"
fi

echo "Setting up the Python environment ..."
[ -d .venv ] || uv venv -q --python 3.12 .venv
uv pip install -q --python .venv/bin/python -r requirements.txt

echo "Fetching Qwen/Qwen3-ASR-1.7B-hf into $MODEL (3.9 GB the first time) ..."
mkdir -p "$MODEL"
.venv/bin/python - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-ASR-1.7B-hf", allow_patterns=["*.json", "*.jinja", "*.safetensors"],
                  local_dir=sys.argv[1])
PY

mkdir -p ~/.config/systemd/user
sed -e "s|HERE_DIR|$HERE|g" -e "s|LISTEN_PORT|$PORT|g" -e "s|HOST_ARGS|${HOST_ARGS% }|g" \
    -e "s|MODEL_DIR|$MODEL|g" -e "s|TOKEN_PATH|$TOKEN_FILE|g" \
    "$HERE/iris-listen.service" > ~/.config/systemd/user/iris-listen.service
systemctl --user daemon-reload
systemctl --user enable --now iris-listen.service
systemctl --user restart iris-listen.service
echo "Starting: ~30 s (loading and compiling the model). Logs: journalctl --user -u iris-listen -f"
echo "Callers send: Authorization: Bearer \$(cat $TOKEN_FILE)  to http://127.0.0.1:$PORT"
