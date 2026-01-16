# Copyright (c) 2024-2025, RheoAgent Project
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""RheoAgent - 流体仿真核心引擎 (Fluid Simulation Core Engine)

基于 Taichi 的 MPM (Material Point Method) 流体仿真引擎。

使用方式:
    from rheo import init_taichi
    init_taichi(num_envs=1, enable_grad=False)
    
    from rheo import TaichiEnv
    env = TaichiEnv(...)
"""

# 只导出 init_taichi，其他组件需要在 init_taichi() 之后导入
def init_taichi(num_envs=1, enable_grad=True, device_memory_GB=None):
    """初始化 Taichi 环境
    
    必须在使用任何其他 rheo 组件之前调用。
    
    Args:
        num_envs: 并行环境数量
        enable_grad: 是否启用梯度计算
        device_memory_GB: GPU 显存分配 (GB)
    """
    from .taichi_env import init_taichi as _init_taichi
    _init_taichi(num_envs=num_envs, enable_grad=enable_grad, device_memory_GB=device_memory_GB)


def __getattr__(name):
    """延迟导入 - 只有在访问时才导入 Taichi 相关组件"""
    if name == "TaichiEnv":
        from .taichi_env import TaichiEnv
        return TaichiEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "init_taichi",
    "TaichiEnv",
]
