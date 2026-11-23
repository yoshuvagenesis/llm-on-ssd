#!/bin/sh
# Start the chat server in the background. It keeps running and starts again by itself
# after a reboot. Run from the repository folder after building the image (docker build).
#
#   MODELS=/volume1/moe sh docker/serve.sh
#
# Settings (environment variables):
#   MODELS    folder with the pack and its tokenizer          (default /volume1/moe)
#   PACK      pack file name                                 (default qwen3-q4b-d4.moe)
#   TOKENIZER the model's tokenizer.json, in MODELS           (default tokenizer.json;
#             for Qwen3.6 use its own file, e.g. tokenizer-qwen3.6.json)
#   MIRROR    optional folder on a second SSD with a copy of the same pack
#   PORT      port for the web page and API                  (default 8080)
#   PASSWORD  optional password for the page and API
#   RAM_GB    hard RAM cap for the engine                    (default 7)
# Extra arguments go to the engine, e.g.  sh docker/serve.sh --max-context 8192
MODELS=${MODELS:-/volume1/moe}
PACK=${PACK:-qwen3-q4b-d4.moe}
TOKENIZER=${TOKENIZER:-tokenizer.json}
PORT=${PORT:-8080}
RAM_GB=${RAM_GB:-7}

if [ ! -f "$MODELS/$PACK" ] || [ ! -f "$MODELS/$TOKENIZER" ]; then
  echo "error: $MODELS needs $PACK and $TOKENIZER (the model's own tokenizer.json, from its Hugging Face folder)"
  exit 1
fi
set -- --pack "/models/$PACK" --tokenizer "/models/$TOKENIZER" --serve --ram-gb "$RAM_GB" "$@"
mirror_mount=""
if [ -n "$MIRROR" ]; then
  mirror_mount="-v $MIRROR:/models2"
  set -- "$@" --mirror "/models2/$PACK"
fi
[ -n "$PASSWORD" ] && set -- "$@" --password "$PASSWORD"

docker rm -f llm-on-ssd >/dev/null 2>&1
docker run -d --name llm-on-ssd --restart unless-stopped \
  --memory "${RAM_GB}g" --memory-swap "${RAM_GB}g" -p "$PORT:8080" \
  -v "$MODELS":/models $mirror_mount --entrypoint moe-engine moe-bench "$@" >/dev/null || exit 1

echo "Chat server started. It needs about a minute to load, then open:"
echo "  http://<this machine's IP address>:$PORT"
echo "Logs: docker logs -f llm-on-ssd     Stop: docker rm -f llm-on-ssd"
