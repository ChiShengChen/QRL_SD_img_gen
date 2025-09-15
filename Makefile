.PHONY: help env data train sample test clean fmt lint

# 預設目標
help:
	@echo "QRL Image Synthesis - 可用指令："
	@echo "  make env     - 建立 conda 環境"
	@echo "  make data    - 下載 CIFAR-10 與預訓練模型"
	@echo "  make train   - 訓練量子策略（預設配置）"
	@echo "  make sample  - 從最新 checkpoint 生成圖像"
	@echo "  make test    - 運行測試"
	@echo "  make clean   - 清理生成檔案"
	@echo "  make fmt     - 格式化程式碼"
	@echo "  make lint    - 檢查程式碼風格"

# 建立 conda 環境
env:
	@echo "建立 conda 環境..."
	conda env create -f environment.yml
	@echo "環境建立完成！請執行：conda activate qrl_image_synthesis"

# 下載資料與預訓練模型
data:
	@echo "下載 CIFAR-10 資料集..."
	python scripts/download_cifar10.py
	@echo "資料下載完成！"

# 訓練量子策略
train:
	@echo "開始訓練量子策略..."
	python scripts/train_qrl.py dataset=cifar10 control=stageA_cfg algo=ppo model=unet_ddim
	@echo "訓練完成！"

# 生成圖像
sample:
	@echo "從最新 checkpoint 生成圖像..."
	@if [ -f "runs/latest/ckpt_best.pt" ]; then \
		python scripts/sample_qrl.py --ckpt runs/latest/ckpt_best.pt --class 3 --num 16; \
	else \
		echo "找不到 checkpoint，請先執行 make train"; \
	fi

# 運行測試
test:
	@echo "運行測試..."
	pytest tests/ -v

# 清理生成檔案
clean:
	@echo "清理生成檔案..."
	rm -rf runs/
	rm -rf assets/samples/*
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	@echo "清理完成！"

# 格式化程式碼
fmt:
	@echo "格式化程式碼..."
	black qrl/ scripts/ tests/
	isort qrl/ scripts/ tests/

# 檢查程式碼風格
lint:
	@echo "檢查程式碼風格..."
	flake8 qrl/ scripts/ tests/
	black --check qrl/ scripts/ tests/
	isort --check-only qrl/ scripts/ tests/

# 安裝開發依賴
dev-deps:
	@echo "安裝開發依賴..."
	pip install black isort flake8 pytest pytest-xdist

# 完整建置流程
build: env data
	@echo "完整建置完成！"
	@echo "下一步：conda activate qrl_image_synthesis && make train"

# 快速測試（不包含訓練）
quick-test: env data test
	@echo "快速測試完成！"

# 顯示專案狀態
status:
	@echo "專案狀態："
	@echo "  - 環境：$$(if conda env list | grep -q qrl_image_synthesis; then echo "已建立"; else echo "未建立"; fi)"
	@echo "  - 資料：$$(if [ -d "data/cifar10" ]; then echo "已下載"; else echo "未下載"; fi)"
	@echo "  - 模型：$$(if [ -f "runs/latest/ckpt_best.pt" ]; then echo "已訓練"; else echo "未訓練"; fi)"
	@echo "  - 樣本：$$(ls -1 assets/samples/*.png 2>/dev/null | wc -l) 張圖像"
