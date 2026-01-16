# Copyright (c) 2024-2025, RheoAgent Project
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""RheoAgent 环境模块

导入此模块会自动注册所有环境到 gymnasium。
"""

# 导入所有环境以触发注册
from . import pouring
from . import tablematerials

__all__ = [
    "pouring",
    "tablematerials",
]

