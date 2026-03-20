import os
import math
from datetime import datetime
import hashlib
from pathlib import Path
import torch

# ============================================================
# 日志目录生成函数：使用日期-时间并检查重复
# ============================================================
def get_unique_log_dir(base_dir="logs", prefix="exp"):
    """
    生成唯一的日志文件夹路径。
    使用日期-时间作为基础名称，如果存在则添加随机哈希直到找到不重复的名称。
    
    Args:
        base_dir: 日志文件夹的根目录 (默认: "logs")
        prefix: 日志文件夹的前缀 (默认: "exp")
    
    Returns:
        str: 唯一的日志文件夹路径
    """
    # 确保根目录存在
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    
    # 生成日期-时间格式的文件夹名称 (格式: exp_2025-03-19_14-30-45)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"{base_dir}/{prefix}_{timestamp}"
    
    # 如果不存在，直接返回
    if not Path(log_dir).exists():
        return log_dir
    
    # 如果存在，添加随机哈希直到找到唯一的名称
    attempt = 0
    while Path(log_dir).exists():
        # 生成随机哈希 (使用尝试次数和当前时间)
        hash_input = f"{timestamp}_{attempt}".encode()
        random_hash = hashlib.sha256(hash_input).hexdigest()[:8]
        log_dir = f"{base_dir}/{prefix}_{timestamp}_{random_hash}"
        attempt += 1
    
    return log_dir



# ============================================================
# 高效张量函数：利用四元数进行逆向旋转 (全局->局部坐标系)
# ============================================================
@torch.jit.script
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    使用四元数对向量进行逆向旋转 (从世界坐标系转换到机身局部坐标系)
    q: 形状为 (N, 4) 的四元数张量，格式必须是 [x, y, z, w]
    v: 形状为 (N, 3) 的三维向量张量
    """
    q_w = q[:, 3:4]
    q_vec = q[:, 0:3]
    
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * torch.sum(q_vec * v, dim=-1, keepdim=True) * 2.0
    
    return a - b + c
