"""日誌工具模組，提供統一的日誌記錄功能。"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json
import csv


class QRLLogger:
    """QRL 專用的日誌記錄器。"""
    
    def __init__(
        self,
        name: str = "qrl",
        level: int = logging.INFO,
        log_dir: Optional[str] = None,
        console_output: bool = True,
        file_output: bool = True,
        tensorboard: bool = False,
    ):
        """
        初始化日誌記錄器。
        
        Args:
            name: 日誌記錄器名稱
            level: 日誌級別
            log_dir: 日誌目錄
            console_output: 是否輸出到控制台
            file_output: 是否輸出到檔案
            tensorboard: 是否啟用 TensorBoard
        """
        self.name = name
        self.level = level
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.console_output = console_output
        self.file_output = file_output
        self.tensorboard = tensorboard
        
        # 創建日誌目錄
        if self.file_output:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化日誌記錄器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()  # 清除現有處理器
        
        # 設置格式
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # 控制台輸出
        if self.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # 檔案輸出
        if self.file_output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.log_dir / f"{name}_{timestamp}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # TensorBoard 支持
        if self.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.log_dir / "tensorboard")
            except ImportError:
                print("警告：TensorBoard 未安裝，將禁用 TensorBoard 日誌")
                self.tensorboard = False
        
        # 記錄初始化信息
        self.info(f"日誌記錄器 '{name}' 已初始化")
        if self.file_output:
            self.info(f"日誌檔案：{log_file}")
    
    def debug(self, message: str) -> None:
        """記錄 DEBUG 級別日誌。"""
        self.logger.debug(message)
    
    def info(self, message: str) -> None:
        """記錄 INFO 級別日誌。"""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """記錄 WARNING 級別日誌。"""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """記錄 ERROR 級別日誌。"""
        self.logger.error(message)
    
    def critical(self, message: str) -> None:
        """記錄 CRITICAL 級別日誌。"""
        self.logger.critical(message)
    
    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """記錄指標到 TensorBoard。"""
        if self.tensorboard and hasattr(self, 'writer'):
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, step)
                elif isinstance(value, str):
                    self.writer.add_text(key, value, step)
    
    def log_hyperparameters(self, hparams: Dict[str, Any]) -> None:
        """記錄超參數到 TensorBoard。"""
        if self.tensorboard and hasattr(self, 'writer'):
            self.writer.add_hparams(hparams, {})
    
    def close(self) -> None:
        """關閉日誌記錄器。"""
        if self.tensorboard and hasattr(self, 'writer'):
            self.writer.close()
        self.info("日誌記錄器已關閉")


class MetricsLogger:
    """指標記錄器，用於記錄訓練過程中的各種指標。"""
    
    def __init__(self, log_dir: str, filename: str = "metrics.csv"):
        """
        初始化指標記錄器。
        
        Args:
            log_dir: 日誌目錄
            filename: CSV 檔案名稱
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.filename = self.log_dir / filename
        self.metrics_history = []
        self.fieldnames = None
    
    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """
        記錄指標。
        
        Args:
            metrics: 指標字典
            step: 當前步數
        """
        # 添加步數
        metrics_with_step = {"step": step, **metrics}
        
        # 更新欄位名稱
        if self.fieldnames is None:
            self.fieldnames = list(metrics_with_step.keys())
        
        # 添加到歷史記錄
        self.metrics_history.append(metrics_with_step)
        
        # 寫入 CSV 檔案
        with open(self.filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.metrics_history)
    
    def get_metrics(self, key: str) -> list:
        """獲取特定指標的歷史記錄。"""
        return [m.get(key, None) for m in self.metrics_history]


# 全局日誌記錄器實例
_logger = None


def get_logger(name: str = "qrl", **kwargs) -> QRLLogger:
    """獲取全局日誌記錄器實例。"""
    global _logger
    if _logger is None:
        _logger = QRLLogger(name, **kwargs)
    return _logger


def setup_logging(
    name: str = "qrl",
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
    **kwargs
) -> QRLLogger:
    """設置日誌記錄。"""
    global _logger
    _logger = QRLLogger(name, level, log_dir, **kwargs)
    return _logger
