#!/bin/bash
# vbench.sh PORT LABEL [NUM_PROMPTS] - vLLM's serving benchmark (`vllm bench serve`, vLLM 0.30 from the vllm-a image,
# no GPU) against the ds41 server: MT-Bench's 80 first turns (philschmid/mt-bench: vLLM's speculative-decoding set,
# 256 tokens each), one request at a time (the server runs one), thinking off, the service's sampling (temperature
# 1.0, top-p 0.95).  The client's chat request function is patched (patch_vllm_chat.py: a chunk counts as a token only
# when it carries text - the server's role chunk comes before the prompt is read; SSE reads unstripped - keep-alive
# comments swallowed the first words).  No vLLM warm-ups: they repeat the first prompt, and the engine answers a
# repeated prompt from its snapshot without reading it (the first request's TTFT would be ~10 ms); tspeed.sh warms the
# server with other requests.  -> out/mtbench-LABEL.json
set -u
S=/opt/ai/dsv41/bench/speed
P=$1; L=$2; NP=${3:-80}
TOK=/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/1bae0e1088993b9b82de08f59de5be5a34e42170
[ -f $S/endpoint_request_func.py ] || python3 -I $S/patch_vllm_chat.py $S/endpoint_request_func.orig.py $S/endpoint_request_func.py
OPENAI_API_KEY=$(grep '^VLLM_API_KEY' /opt/docker/vllm/.env | cut -d= -f2- | tr -d '[:space:]') \
docker run --rm --network host --cpuset-cpus 15 -e OPENAI_API_KEY -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  -v $S/hf:/hf -v $S/out:/out -v $S/endpoint_request_func.py:/opt/venv/lib/python3.12/site-packages/vllm/benchmarks/lib/endpoint_request_func.py:ro \
  --entrypoint vllm vllm/vllm-openai-xpu:v0.30.0 bench serve \
  --backend openai-chat --base-url http://172.23.0.1:$P --endpoint /v1/chat/completions \
  --model deepseek-v4.1-flash-1017 --tokenizer $TOK \
  --dataset-name hf --dataset-path philschmid/mt-bench --hf-split train --no-stream --skip-chat-template \
  --num-prompts $NP --max-concurrency 1 --request-rate inf --num-warmups 0 \
  --temperature 1.0 --top-p 0.95 --extra-body '{"reasoning_effort": "none"}' \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
  --disable-tqdm --save-result --save-detailed --result-dir /out --result-filename mtbench-$L.json 2>&1 \
  | grep -a -v -E "^\s*$|zes_count|_xpu_getDeviceCount"
