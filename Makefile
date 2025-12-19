# ============================
# Project settings
# ============================
PROJECT_NAME=road-mark-detection
PYTHON=py
PIP=$(PYTHON) -m pip
DOCKER_COMPOSE=docker-compose

# ============================
# MLflow
# ============================
MLFLOW_URI=http://localhost:5000

# ============================
# Default
# ============================
help:
	@echo "Available commands:"
	@echo "  make setup            - Install python dependencies"
	@echo "  make infra-up         - Start all infrastructure services"
	@echo "  make infra-down       - Stop all infrastructure services"
	@echo "  make infra-restart    - Restart infrastructure"
	@echo "  make logs             - View docker logs"
	@echo "  make mlflow           - Open MLflow UI"
	@echo "  make train-baseline   - Run baseline training"
	@echo "  make train-cnn        - Run CNN training"
	@echo "  make clean            - Remove cache & temp files"

# ============================
# Python environment
# ============================
setup:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# ============================
# Docker Infrastructure
# ============================
start:
	$(DOCKER_COMPOSE) up -d --build

stop:
	$(DOCKER_COMPOSE) down

restart:
	$(DOCKER_COMPOSE) down
	$(DOCKER_COMPOSE) up -d --build

logs:
	$(DOCKER_COMPOSE) logs -f

# ============================
# MLflow
# ============================
mlflow:
	@echo "Open MLflow UI at $(MLFLOW_URI)"

# ============================
# Training
# ============================
train-yolo:
	$(PYTHON) scripts/road_mark_detection.py

train-cnn:
	$(PYTHON) scripts/train_cnn.py

# ============================
# Cleanup
# ============================
clean:
	rm -rf __pycache__ */__pycache__
	rm -rf mlruns
	rm -rf models/*.pt
