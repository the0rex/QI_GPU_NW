# Makefile for Alignment Pipeline Docker Management

.PHONY: help build build-cpu build-gpu run-cpu run-gpu test push clean

# Default target
help:
	@echo "Alignment Pipeline Docker Management"
	@echo ""
	@echo "Targets:"
	@echo "  build-cpu      Build CPU-only Docker image"
	@echo "  build-gpu      Build GPU-accelerated Docker image"
	@echo "  run-cpu        Run CPU-only container"
	@echo "  run-gpu        Run GPU-accelerated container"
	@echo "  compose-up     Start all services with docker-compose"
	@echo "  test-cpu       Run tests in CPU container"
	@echo "  test-gpu       Run tests in GPU container"
	@echo "  push           Push images to registry"
	@echo "  clean          Remove Docker images and containers"
	@echo "  logs           View container logs"
	@echo "  shell          Open shell in container"

# Build targets
build-cpu:
	docker build \
		--build-arg USE_CUDA=false \
		-t alignment-pipeline:cpu \
		-t alignment-pipeline:latest \
		.

build-gpu:
	docker build \
		--build-arg USE_CUDA=true \
		--build-arg CUDA_VERSION=12.1 \
		-t alignment-pipeline:gpu \
		-t alignment-pipeline:cuda-12.1 \
		.

build-dev:
	docker build \
		-f Dockerfile.dev \
		-t alignment-pipeline:dev \
		.

# Run targets
run-cpu: build-cpu
	docker run --rm -it \
		-v $(PWD)/data:/home/pipeline/data:ro \
		-v $(PWD)/results:/home/pipeline/results \
		-v $(PWD)/config:/home/pipeline/config:ro \
		--memory="8g" \
		--cpus="4" \
		alignment-pipeline:cpu \
		$(ARGS)

run-gpu: build-gpu
	docker run --rm -it \
		--gpus all \
		-v $(PWD)/data:/home/pipeline/data:ro \
		-v $(PWD)/results-gpu:/home/pipeline/results \
		-v $(PWD)/config:/home/pipeline/config:ro \
		-e NVIDIA_VISIBLE_DEVICES=all \
		alignment-pipeline:gpu \
		$(ARGS)

# Docker Compose
compose-up:
	docker-compose up -d

compose-down:
	docker-compose down

compose-logs:
	docker-compose logs -f

# Test targets
test-cpu: build-cpu
	docker run --rm \
		-v $(PWD)/tests:/home/pipeline/app/tests:ro \
		alignment-pipeline:cpu \
		pytest /home/pipeline/app/tests -v

test-gpu: build-gpu
	docker run --rm \
		--gpus all \
		-v $(PWD)/tests:/home/pipeline/app/tests:ro \
		alignment-pipeline:gpu \
		pytest /home/pipeline/app/tests -v

# Development
dev-shell:
	docker run --rm -it \
		-v $(PWD)/src:/home/pipeline/app/src \
		-v $(PWD)/tests:/home/pipeline/app/tests \
		-v $(PWD)/data:/home/pipeline/data:ro \
		-v $(PWD)/config:/home/pipeline/config:ro \
		--entrypoint bash \
		alignment-pipeline:cpu

# Utility targets
push:
	docker tag alignment-pipeline:cpu your-registry/alignment-pipeline:cpu
	docker tag alignment-pipeline:gpu your-registry/alignment-pipeline:gpu
	docker push your-registry/alignment-pipeline:cpu
	docker push your-registry/alignment-pipeline:gpu

clean:
	docker system prune -f
	docker rmi alignment-pipeline:cpu alignment-pipeline:gpu alignment-pipeline:dev || true

logs:
	docker logs -f alignment-pipeline-cpu

# Example usage with arguments
example-run:
	@echo "Example: make run-cpu ARGS='--fasta1=data/seq1.fa --fasta2=data/seq2.fa --verbose'"
	@echo "Example: make run-gpu ARGS='--config=config/pipeline_config_gpu.yaml'"