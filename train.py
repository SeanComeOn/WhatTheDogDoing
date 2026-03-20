import yaml

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.env import VecEnv

import os
import math
import torch
import yaml
from tensordict import TensorDict

from utils import get_unique_log_dir
from env import WhatTheDogDoingEnv

# 1) Create your environment (usually provided by environment libraries such as Isaac Lab)
# env = make_env()


if __name__ == "__main__":
    # 2) Load a YAML configuration and extract the configuration dictionary expected by RSL-RL
    with open("simple_cfg.yaml", "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
    train_cfg = full_cfg["runner"]

    # 3) Generate unique log directory based on date-time
    log_dir = get_unique_log_dir(base_dir="logs", prefix="exp")
    print(f"\n=== 实验日志文件夹: {log_dir} ===")


    # 4) Build the runner
    runner = OnPolicyRunner(
        env=WhatTheDogDoingEnv(train_cfg, num_envs=train_cfg["num_envs"], num_actions=train_cfg["num_actions"]),
        train_cfg=train_cfg,
        log_dir=log_dir, # Directory for saving checkpoints and logs
        device="cuda", # Device to run the training on
    )

    # ============================================================
    # 【新增】加载预训练权重 (Resume / Fine-tuning)
    # ============================================================
    # resume_path = "../../7.code_refactor/logs/exp_2026-03-19_20-33-29/model_1100.pt"
    
    # if os.path.exists(resume_path):
    #     print(f"\n[*] 成功找到并加载预训练权重: {resume_path}")
    #     runner.load(resume_path)
    # else:
    #     raise FileNotFoundError(f"找不到权重: {resume_path}")

    # 5) Start training
    runner.learn(num_learning_iterations=1500) # Specify the number of desired iterations

    # 6) Export the trained policy for deployment
    runner.export_policy_to_jit(f"{log_dir}/exported", filename="policy.pt")
    runner.export_policy_to_onnx(f"{log_dir}/exported", filename="policy.onnx")