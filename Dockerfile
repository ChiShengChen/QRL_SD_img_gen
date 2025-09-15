# 使用 NVIDIA CUDA 12.1 基礎鏡像
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# 設置環境變數
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

# 設置 PATH
ENV PATH="/opt/conda/bin:$PATH"

# 複製環境配置文件
COPY environment.yml /tmp/environment.yml

# 創建 conda 環境
RUN conda env create -f /tmp/environment.yml

# 激活環境
SHELL ["conda", "run", "-n", "qrl_image_synthesis", "/bin/bash", "-c"]

# 複製專案檔案
WORKDIR /workspace
COPY . .

# 安裝專案依賴
RUN pip install -e .

# 創建入口腳本
RUN echo '#!/bin/bash\n\
source /opt/conda/etc/profile.d/conda.sh\n\
conda activate qrl_image_synthesis\n\
exec "$@"' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

# 設置工作目錄
WORKDIR /workspace

# 暴露端口（如果需要）
EXPOSE 8888

# 設置入口點
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
