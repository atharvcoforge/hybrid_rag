MODEL=models/qwen2.5-3b-instruct-q4_k_m.gguf
URL=https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf
LLAMA_URL=https://github.com/ggml-org/llama.cpp/releases/download/b11093/llama-b11093-bin-macos-arm64.tar.gz
LLAMA=tools/llama-b11093/llama-server

.PHONY: model up test

model:
	mkdir -p models
	test -f $(MODEL) || curl -L --fail --retry 3 -o $(MODEL) $(URL)

$(LLAMA):
	mkdir -p tools
	curl -L --fail --retry 3 -o tools/llama.tar.gz $(LLAMA_URL)
	tar -xzf tools/llama.tar.gz -C tools

up: model $(LLAMA)
	@curl -sf http://127.0.0.1:8081/health >/dev/null || $(LLAMA) -m $(MODEL) --port 8081 -ngl 99 --host 127.0.0.1 >/tmp/llama-server.log 2>&1 &
	docker compose up --build

test:
	python -m pytest

eval:
	uv run rag eval --index index
