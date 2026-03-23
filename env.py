from tensordict import TensorDict
from rsl_rl.env import VecEnv
import warp as wp
import mujoco
import mujoco_warp as mjw
import torch
from utils import quat_rotate_inverse

class WhatTheDogDoingEnv(VecEnv):
    def __init__(self, config, num_envs=128, num_actions=12):
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.max_episode_length = 1000
        self.device = "cuda"
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device) # https://deepwiki.com/search/episodelengthbuf_b816b810-6d11-4b8d-8bc0-ea5224a68dad?mode=fast
        self.cfg = config

        # load from go2/scene.xml
        mjm = mujoco.MjModel.from_xml_path("go2/scene_plane.xml")
        self.m = mjw.put_model(mjm)
        self.d = mjw.make_data(mjm, nworld=num_envs)
        
        # ============================================================
        # 观测缩放系数 (极大加速神经网络收敛)
        # ============================================================
        self.obs_scales = {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05
        }
        
        # ============================================================
        # 动态读取默认站立姿态 (告别硬编码)
        # ============================================================
        # 1. 通过名字找到 XML 中名为 "home" 的 keyframe 的 ID
        key_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id < 0:
            raise ValueError("在 XML 中找不到名为 'home' 的 keyframe！")

        # 2. 从模型中直接提取这个 keyframe 对应的 19 维 qpos
        home_qpos = mjm.key_qpos[key_id]

        # 3. 提取机身默认高度 (索引 2 是 Z 轴坐标)
        self.default_base_height = home_qpos[2]

        # 4. 提取 12 个电机的默认角度 (索引 7 到 19)
        self.default_dof_pos = torch.tensor(
            home_qpos[7:19], 
            dtype=torch.float, 
            device=self.device
        ).repeat(self.num_envs, 1)
        
        # 5. 提取默认的机身四元数 (索引 3 到 7，MuJoCo 格式 [w, x, y, z])
        self.default_base_quat = torch.tensor(
            home_qpos[3:7],
            dtype=torch.float,
            device=self.device
        )
        
        # ============================================================
        # 状态缓存 (供网络感知指令和延迟)
        # ============================================================
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)  # [v_x, v_y, yaw]
        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        
        # ============================================================
        # 历史观测缓冲区 (Frame Stacking)
        # ============================================================
        # 从配置动态读取，而非硬编码
        self.history_length = self.cfg.get("history_length", 5)
        self.single_obs_dim = self.cfg.get("single_obs_dim", 45)
        
        # 初始化缓冲区 [batch_size, history_length, single_obs_dim]
        self.obs_history_buf = torch.zeros(
            self.num_envs, self.history_length, self.single_obs_dim, 
            dtype=torch.float, device=self.device
        )
        
        print(f"[*] 环境观测配置:")
        print(f"    - 历史帧数: {self.history_length}")
        print(f"    - 单帧维度: {self.single_obs_dim}")
        print(f"    - 总观测维度: {self.history_length * self.single_obs_dim}")
        
        # 用于缓存的中间变量
        self.base_quat = None
        self.base_lin_vel = None
        self.base_ang_vel = None
        
        # ============================================================
        # 控制参数 (物理推演与动作映射)
        # ============================================================
        # 降采样：假设网络 50Hz，物理引擎 500Hz，每步推演 10 次
        self.decimation = 10 
        
        # 动作缩放：网络输出 1.0 时，实际角度变化 0.25 rad (约14度)
        self.action_scale = 0.25 

        # 【新增】指令随机生成区间配置 [min, max]
        self.command_ranges = {
            "lin_vel_x": [-1.0, 1.0],   # 前向速度范围 [m/s] (例如：后退到全速前进)
            "lin_vel_y": [-0.5, 0.5],   # 侧向速度范围 [m/s] (左移到右移)
            "ang_vel_yaw": [-1.0, 1.0], # 偏航角速度范围 [rad/s] (左转到右转)
        }


    def get_observations(self):
        """
        提取观测张量。严格区分传感器数据 (Policy) 和 特权数据 (Privileged)。
        
        返回维度说明：
        - policy: 45 维 (3 ang_vel + 3 proj_gravity + 3 commands + 12 dof_pos + 12 dof_vel + 12 actions)
        - privileged: 48 维 (3 lin_vel + 45 policy)
        """
        # ================= 0. 获取底层原始张量 =================
        qpos = wp.to_torch(self.d.qpos).to(self.device)  # [128, 19]
        qvel = wp.to_torch(self.d.qvel).to(self.device)  # [128, 18]

        # ================= 1. 模拟 IMU 传感器 =================
        # 提取机身四元数并转换格式 (MuJoCo的 [w,x,y,z] 转为 PyTorch的 [x,y,z,w])
        base_quat_mujoco = qpos[:, 3:7]
        self.base_quat = base_quat_mujoco[:, [1, 2, 3, 0]]
        
        # 1.1 投影重力 (完美替代 IMU 加速度计的姿态感知)
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        obs_proj_gravity = quat_rotate_inverse(self.base_quat, gravity_vec)
        
        # 1.2 局部角速度 (替代 IMU 陀螺仪)
        global_ang_vel = qvel[:, 3:6]
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, global_ang_vel)
        obs_ang_vel = self.base_ang_vel * self.obs_scales["ang_vel"]

        # ================= 2. 模拟电机编码器 =================
        qpos_joints = qpos[:, 7:19]
        qvel_joints = qvel[:, 6:18]
        
        # 计算关节角度偏差与转速
        obs_dof_pos = (qpos_joints - self.default_dof_pos) * self.obs_scales["dof_pos"]
        obs_dof_vel = qvel_joints * self.obs_scales["dof_vel"]

        # ================= 3. 内部状态信息 =================
        obs_commands = self.commands * self.obs_scales["lin_vel"]  # 目标指令
        obs_actions = self.actions  # 上一步动作

        # ================= 4. 打包当前单步 Policy 观测 =================
        # 维度合计: 3(角速度) + 3(重力) + 3(指令) + 12(角度) + 12(转速) + 12(动作) = 45 维
        current_obs_policy = torch.cat((
            obs_ang_vel,
            obs_proj_gravity,
            obs_commands,
            obs_dof_pos,
            obs_dof_vel,
            obs_actions
        ), dim=-1)  # 形状: [num_envs, 45]

        # ================= 【核心修改】更新历史缓冲区 =================
        # 利用 torch.roll 将历史数据往前挪一格（丢弃最老的一帧），并将最新帧放入末尾
        self.obs_history_buf = torch.roll(self.obs_history_buf, shifts=-1, dims=1)
        self.obs_history_buf[:, -1, :] = current_obs_policy

        # 展平历史缓冲区 -> 形状: [num_envs, 225]
        obs_policy_flat = self.obs_history_buf.view(self.num_envs, -1)

        # ================= 5. 获取特权信息 (上帝视角) =================
        # 真实局部线速度 (只有 Critic 能看)
        global_lin_vel = qvel[:, 0:3]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, global_lin_vel)
        obs_lin_vel = self.base_lin_vel * self.obs_scales["lin_vel"]
        
        # 【修改】：不再手动拼接 policy，直接返回纯净的特权信息
        obs_privileged = obs_lin_vel  # 形状: [num_envs, 3]

        # ================= 6. 返回给 RSL-RL =================
        return TensorDict({
            "policy": obs_policy_flat,
            "privileged": obs_privileged
        })

    def reset(self):
        """Reset all environments and return initial observations."""
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(env_ids)
        return self.get_observations()

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """执行动作，步进物理世界，并返回强化学习所需的所有反馈
        
        Returns:
            observations: TensorDict with observation groups
            rewards: Rewards tensor shape (num_envs,)
            dones: Done flags shape (num_envs,)
            extras: Extra information dict
        """
        # ================= 1. 动作记录与映射 =================
        # 【修改】给动作套上紧箍咒，强制限制在 [-1.0, 1.0] 之间
        clipped_actions = torch.clamp(actions, min=-1.0, max=1.0)
        
        # 记录被裁剪后的安全动作
        self.actions = clipped_actions.clone()
        
        # 将网络输出的 [-1, 1] 映射为目标关节角度 (绝对位置)
        # target_dof_pos 形状为 [128, 12]
        target_dof_pos = self.default_dof_pos + self.actions * self.action_scale
        
        # ================= 2. 高频物理推演 (Decimation) =================
        for _ in range(self.decimation):
            # 将目标角度直接写入 mujoco 的 ctrl 张量
            # (因为 xml 里用了 <position> 执行器，MuJoCo 会自动用 PD 算出 torque)
            wp.copy(self.d.ctrl, wp.from_torch(target_dof_pos))
            
            # 物理引擎步进
            mjw.step(self.m, self.d)
            
        # 更新存活时间
        self.episode_length_buf += 1

        # ================= 3. 获取最新观测 =================
        # 这会调用 get_observations，并更新内部的 base_lin_vel 等状态
        obs = self.get_observations()

        # ================= 【新增】NaN/Inf 终极防御机制 =================
        # 分别检查 policy 和 privileged 中是否有 NaN 或 Inf
        nan_policy = torch.isnan(obs["policy"]).any(dim=1) | torch.isinf(obs["policy"]).any(dim=1)
        nan_priv = torch.isnan(obs["privileged"]).any(dim=1) | torch.isinf(obs["privileged"]).any(dim=1)

        # 只要任何一个有脏数据，该环境判定为物理崩溃
        nan_mask = nan_policy | nan_priv
        has_nan = nan_mask.any()

        if has_nan:
            # 物理层面的隔离：把脏数据强制清零，防止毒害 Actor 和 Critic
            obs["policy"][nan_mask] = 0.0
            obs["privileged"][nan_mask] = 0.0
            print(f"[警告] 检测到 {nan_mask.sum().item()} 个环境发生物理爆炸 (NaN/Inf)！已隔离并清零观测。")

        # ================= 4. 结算奖励与终止条件 =================
        # 判断狗子是不是死了 (摔倒或超时)
        dones, time_outs = self.check_terminations()

        # 【新增】最高指令：强制让发生 NaN 的环境立刻 done
        if has_nan:
            dones = dones | nan_mask
        
        # 计算当前步的奖励 (依赖于刚刚 get_observations 算出的最新状态)
        rewards = self.compute_rewards()

        # 【新增】清洗奖励：防止计算出的 NaN reward 传给网络
        if has_nan:
            rewards[nan_mask] = 0.0

        # ================= 5. 局部重置 (The "Magic" Part) =================
        # 找出哪些环境死掉了 (done = True 的索引)
        env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        
        if len(env_ids) > 0:
            # 只把死掉的狗重新扶起来，其他狗继续跑
            self.reset_idx(env_ids)
            
            # 【极其关键】重置后，死掉的狗的状态突变了，必须重新获取一次观测！
            # 否则传给 RSL-RL 的将是狗子四脚朝天的遗容，网络会非常困惑
            obs = self.get_observations()

            # 【新增】双重保险：重置后再次清洗潜在的 NaN / Inf
            obs["policy"] = torch.nan_to_num(obs["policy"], nan=0.0, posinf=0.0, neginf=0.0)
            obs["privileged"] = torch.nan_to_num(obs["privileged"], nan=0.0, posinf=0.0, neginf=0.0)

        # ================= 6. 返回给 RSL-RL =================
        extras = {
            "time_outs": time_outs,
            "log": {} # 以后可以把各种奖励分项放进这里，在 tensorboard 里看
        }
        
        return obs, rewards, dones, extras

    def compute_rewards(self):
        """
        计算奖励函数。
        当前目标：鼓励狗子以最小的动作幅度，全向跟踪线速度和角速度指令。
        """
        # ================= 1. 获取真实速度与目标速度 =================
        # self.base_lin_vel 包含局部坐标系的 [v_x, v_y, v_z]
        actual_vx = self.base_lin_vel[:, 0]
        actual_vy = self.base_lin_vel[:, 1]

        # self.base_ang_vel 包含局部坐标系的 [omega_roll, omega_pitch, omega_yaw]
        actual_yaw = self.base_ang_vel[:, 2]

        # 提取目标指令
        target_vx = self.commands[:, 0]
        target_vy = self.commands[:, 1]
        target_yaw = self.commands[:, 2]

        # ================= 2. 计算跟随奖励 =================
        # 计算线速度误差的平方和 (x 和 y 结合计算)
        lin_vel_error_sq = torch.square(actual_vx - target_vx) + torch.square(actual_vy - target_vy)
        # 映射为 0~1 的线速度奖励
        reward_tracking_lin_vel = torch.exp(-lin_vel_error_sq / 0.25)

        # 计算角速度误差平方
        yaw_error_sq = torch.square(actual_yaw - target_yaw)
        # 映射为 0~1 的角速度奖励
        reward_tracking_yaw = torch.exp(-yaw_error_sq / 0.25)

        # ================= 3. 动作幅度惩罚 (防止抽搐) =================
        # 惩罚网络输出动作的平方和，鼓励它用最接近默认站立姿态的动作来完成任务
        # self.actions 形状是 [128, 12]，沿最后一个维度求和
        penalty_actions = torch.sum(torch.square(self.actions), dim=1)

        # ================= 4. 计算总奖励 =================
        # 为不同维度分配权重：线速度是大头，角速度辅助，动作惩罚是微小的负分
        # 如果狗子为了转弯而不往前走，可以适当调高 lin_vel 的权重
        weight_lin_vel = 1.0
        weight_yaw = 0.5
        weight_action_penalty = -0.01

        total_reward = (
            weight_lin_vel * reward_tracking_lin_vel
            + weight_yaw * reward_tracking_yaw
            + weight_action_penalty * penalty_actions
        )
        
        return total_reward

    def check_terminations(self):
        """判断环境是否终止
        
        Returns:
            dones: 整体终止标志 (超时 或 摔倒)
            time_outs: 仅由于超时的标志
        """
        # 条件 1：超时 (达到最大步数)
        time_outs = self.episode_length_buf >= self.max_episode_length
        
        # 条件 2：摔倒判定 (机身高度过低，或者发生翻转)
        # qpos 的前 3 维是全局 [x, y, z]。取出 z (高度)
        qpos = wp.to_torch(self.d.qpos).to(self.device)
        base_z = qpos[:, 2]
        
        # 提取投影重力的 z 分量 (正常站立时接近 -1.0)
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        gravity_proj = quat_rotate_inverse(self.base_quat, gravity_vec)
        gravity_z = gravity_proj[:, 2]
        
        # 如果高度低于 0.15 米，或者身体倾斜超过 90 度 (gravity_z > 0)，视为摔倒
        crashed = (base_z < 0.15) | (gravity_z > 0.0)
        
        # 只要满足其一，环境就 done
        dones = time_outs | crashed
        return dones, time_outs

    def reset_idx(self, env_ids):
        """将指定环境的狗子恢复到初始状态（修复物理爆炸版）"""
        if len(env_ids) == 0:
            return

        # 1. 重置计时器
        self.episode_length_buf[env_ids] = 0
        
        # 2. 获取底层内存映射
        qpos = wp.to_torch(self.d.qpos)
        qvel = wp.to_torch(self.d.qvel)
        
        # 3. 恢复 qpos (位置与姿态)
        # 【关键修复】：绝对不能“保持在原地”。必须将 X, Y 归零，清洗掉所有潜在的 NaN 和 Inf！
        qpos[env_ids, 0] = 0.0  
        qpos[env_ids, 1] = 0.0  
        qpos[env_ids, 2] = self.default_base_height
        
        # 恢复默认的机身四元数
        qpos[env_ids, 3:7] = self.default_base_quat
        
        # 12 个关节恢复为动态读取的默认角度
        qpos[env_ids, 7:19] = self.default_dof_pos[env_ids]
        
        # 4. 恢复 qvel (速度全部清零)
        qvel[env_ids, :] = 0.0
        
        # 5. 重置历史状态
        self.actions[env_ids] = 0.0
        if hasattr(self, '_prev_actions'):
            self._prev_actions[env_ids] = 0.0
        
        # 【新增】：彻底清空这些死亡狗子的历史观测
        self.obs_history_buf[env_ids] = 0.0
            
        # 【可选修复】：如果你的 MuJoCo/Warp 版本允许，清空加速度缓存以消除“幽灵力”
        if hasattr(self.d, 'qacc'):
            qacc = wp.to_torch(self.d.qacc)
            qacc[env_ids, :] = 0.0
        
        # 5. 重置历史状态
        self.actions[env_ids] = 0.0
        if hasattr(self, '_prev_actions'):
            self._prev_actions[env_ids] = 0.0
            
        # ==========================================================
        # 【修复2】：彻底清洗 MuJoCo 底层动力学缓存 (消除 NaN 幽灵力)
        # ==========================================================
        caches_to_clear = [
            'qacc',            # 加速度缓存
            'qacc_warmstart',  # 约束求解器热启动缓存 (极易残留 NaN)
            'qfrc_applied',    # 施加的广义力
            'qfrc_bias',       # 偏置力 (科里奥利力/重力)
            'ctrl',            # 控制器输入
            'actuator_force'   # 执行器输出的力
        ]
        
        for cache_name in caches_to_clear:
            if hasattr(self.d, cache_name):
                # 提取底层的 Warp 数组并清零对应 env 的数据
                cache_tensor = wp.to_torch(getattr(self.d, cache_name))
                cache_tensor[env_ids] = 0.0

        # 6. 重新下发指令
        # 获取需要重置的环境数量
        num_reset = len(env_ids)

        # 使用均匀分布公式: rand * (max - min) + min
        # 前向速度 v_x
        min_vx, max_vx = self.command_ranges["lin_vel_x"]
        self.commands[env_ids, 0] = torch.rand(num_reset, device=self.device) * (max_vx - min_vx) + min_vx

        # 侧向速度 v_y
        min_vy, max_vy = self.command_ranges["lin_vel_y"]
        self.commands[env_ids, 1] = torch.rand(num_reset, device=self.device) * (max_vy - min_vy) + min_vy

        # 转向速度 yaw
        min_yaw, max_yaw = self.command_ranges["ang_vel_yaw"]
        self.commands[env_ids, 2] = torch.rand(num_reset, device=self.device) * (max_yaw - min_yaw) + min_yaw
