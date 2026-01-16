# Copyright (c) 2024-2025, RheoAgent Project
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""配置工具函数

注意：新架构使用 dataclass 配置 (env_cfg.py)，此模块保留仅用于兼容旧代码。
"""

import os
import yaml
from yacs.config import CfgNode
from rheo.utils.misc import get_src_dir


def make_cls_config(self, cfg=None):
    """创建类配置"""
    _cfg = self.default_config()
    if cfg is not None:
        if isinstance(cfg, str):
            _cfg.merge_from_file(cfg)
        else:
            _cfg.merge_from_other_cfg(cfg)
    return _cfg


def merge_dict(a, b):
    """合并字典"""
    if b is None:
        return a
    import copy
    a = copy.deepcopy(a)
    for key in a:
        if key in b:
            if not isinstance(b[key], dict):
                a[key] = b[key]
            else:
                assert not isinstance(a[key], list)
                a[key] = merge_dict(a[key], b[key])
    for key in b:
        if key not in a:
            raise ValueError("Key is not in dict A!")
    return a


def merge_lists(a, b):
    """合并列表"""
    outs = []
    assert isinstance(a, list) and isinstance(b, list)
    for i in range(len(a)):
        assert isinstance(a[i], dict)
        x = a[i]
        if i < len(b):
            x = merge_dict(a[i], b[i])
        outs.append(x)
    return outs


def list_to_cfg(l):
    """将列表中的字典转换为 CfgNode"""
    l_new = []
    for x in l:
        x_new = CfgNode(x)
        x_new.freeze()
        l_new.append(x_new)
    return l_new


def load_config(cfg_file_name=None):
    """加载配置文件
    
    Args:
        cfg_file_name: 配置文件路径（相对于 source 目录）
    
    Returns:
        CfgNode: 配置对象
    """
    cfg = CfgNode()
    cfg.set_new_allowed(True)
    if cfg_file_name is not None:
        cfg.merge_from_file(os.path.join(get_src_dir(), cfg_file_name))
    cfg.freeze()
    return cfg
