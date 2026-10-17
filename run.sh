#!/usr/bin/env bash
# Start the test service so a phone can use its camera with it.
#
#   ./run.sh          Same Wi-Fi: https://<this-pc-ip>:8000 (self-signed cert, accept the warning once)
#   ./run.sh tunnel   Any network: prints a public https://....trycloudflare.com link (no cert warning)
#
# Phones only allow camera access on HTTPS pages, which is why both modes use HTTPS.
set -e
cd "$(dirname "$0")"

if [ ! -d weights/onnx ]; then
  echo "Models missing: run ./fetch_weights.sh first." >&2
  exit 1
fi

if [ "$1" = "tunnel" ]; then
  CF=$(command -v cloudflared || true)
  if [ -z "$CF" ]; then
    CF=./bin/cloudflared
    if [ ! -x "$CF" ]; then
      case "$(uname -s)-$(uname -m)" in
        Linux-x86_64) ASSET=cloudflared-linux-amd64 ;;
        Linux-aarch64|Linux-arm64) ASSET=cloudflared-linux-arm64 ;;
        *) echo "Install cloudflared first (macOS: brew install cloudflared)." >&2; exit 1 ;;
      esac
      mkdir -p bin
      curl -fsSL -o "$CF" "https://github.com/cloudflare/cloudflared/releases/latest/download/$ASSET"
      chmod +x "$CF"
    fi
  fi
  uv run uvicorn app:app --host 127.0.0.1 --port 8000 &
  SERVER=$!
  trap 'kill $SERVER' EXIT
  echo "Starting tunnel... open the https://....trycloudflare.com link it prints on your phone."
  echo "Anyone with that link can use the service while this runs; press Ctrl+C to stop."
  "$CF" tunnel --no-autoupdate --url http://127.0.0.1:8000
  exit
fi

IP=$(uv run python -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('1.1.1.1', 80)); print(s.getsockname()[0])")
if [ ! -f cert.pem ]; then
  openssl req -x509 -newkey rsa:2048 -nodes -days 365 -keyout key.pem -out cert.pem \
    -subj "/CN=plate-test" -addext "subjectAltName=IP:$IP,DNS:localhost" 2>/dev/null
fi
echo "Open https://$IP:8000 on your phone (same Wi-Fi). Tap Advanced -> Proceed on the certificate warning."
exec uv run uvicorn app:app --host 0.0.0.0 --port 8000 --ssl-keyfile key.pem --ssl-certfile cert.pem
