import logging
import os
from datetime import datetime

def setup_logger(log_name, log_dir='./results/logs'):
    """
    配置日志功能，同时输出到终端和文件
    """
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)

    # 以当前时间作为日志文件名的一部分，避免覆盖
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(log_dir, f"{log_name}_{current_time}.log")

    # 创建 Logger
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    
    # 避免重复添加 Handler
    if not logger.handlers:
        # 配置文件 Handler (保存到硬盘)
        file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # 配置终端 Handler (打印到屏幕)
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(message)s') # 终端为了美观，尽量保持原样
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger