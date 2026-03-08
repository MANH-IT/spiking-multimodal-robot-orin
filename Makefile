# Makefile cho Multi-Modal Robot AI Project
# Sử dụng: make <target>

.PHONY: help install install-dev test lint format clean

help: ## Hiển thị help message
	@echo "Multi-Modal Robot AI - Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Cài đặt dependencies cơ bản
	python -m pip install -r requirements.txt
	python -m pip install -e .

install-safe: ## Cài đặt dependencies (workaround cho lỗi pip)
	python scripts/install_dependencies.py
	python -m pip install -e .

install-dev: ## Cài đặt dependencies cho development
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

test: ## Chạy tests
	pytest

test-cov: ## Chạy tests với coverage
	pytest --cov=src --cov=vision_system --cov-report=html --cov-report=term

lint: ## Chạy linters
	flake8 src vision_system scripts
	mypy src vision_system --ignore-missing-imports

format: ## Format code
	black .
	isort .

format-check: ## Kiểm tra format (không sửa)
	black --check .
	isort --check .

type-check: ## Type checking
	mypy src vision_system --ignore-missing-imports

quality: format lint type-check ## Chạy tất cả quality checks

clean: ## Xóa cache và build files
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -r {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/

clean-data: ## Xóa dữ liệu đã xử lý (cẩn thận!)
	@echo "⚠️  Cảnh báo: Lệnh này sẽ xóa dữ liệu trong data/01_interim và data/02_processed"
	@read -p "Bạn có chắc chắn? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data/01_interim/*; \
		rm -rf data/02_processed/*; \
		echo "✅ Đã xóa dữ liệu đã xử lý"; \
	else \
		echo "❌ Đã hủy"; \
	fi

train: ## Chạy training
	python vision_system/training/depth_aware_train.py

demo: ## Chạy demo inference
	python vision_system/inference/depth_aware_demo.py

benchmark: ## Chạy benchmark
	python vision_system/inference/benchmark_depth_aware.py

hilo-inspect: ## Kiểm tra HILO dataset
	python scripts/data_collection/hilo_inspect.py

hilo-interim: ## Tạo RGB-D aligned
	python scripts/data_collection/hilo_make_interim.py

hilo-annotations: ## Tạo annotations 3D
	python scripts/data_collection/hilo_make_annotations.py

hilo-yolo: ## Convert sang YOLO format
	python scripts/data_collection/hilo_to_yolo.py

hf-login: ## Login Hugging Face (helper script)
	python login_huggingface.py

download-text-seg: ## Tải text-segmentation dataset từ Hugging Face
	python scripts/data_collection/download_text_segmentation.py

docs: ## Build documentation (nếu có Sphinx)
	@echo "📚 Building documentation..."
	@if [ -d "docs" ]; then \
		cd docs && make html; \
	else \
		echo "⚠️  Thư mục docs/ chưa có Sphinx config"; \
	fi

.DEFAULT_GOAL := help
