# Copyright (c) 2024-2025, RheoAgent Project
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""GL Renderer Source Module

导入编译好的 flex_renderer .so 文件。
"""

import os
import sys

# 获取当前目录路径并添加到 sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# 查找 .so 文件
_so_files = [f for f in os.listdir(_current_dir) if f.endswith('.so')]

if _so_files:
    # 提取模块名（去掉 .so 扩展名）
    _so_name = _so_files[0].replace('.cpython-', '.').split('.')[0]
    try:
        # 直接导入（Python 会自动查找 .so 文件）
        flex_renderer = __import__(_so_name)
    except ImportError as e:
        # 如果直接导入失败，尝试使用 importlib
        try:
            import importlib.util
            _so_path = os.path.join(_current_dir, _so_files[0])
            spec = importlib.util.spec_from_file_location(_so_name, _so_path)
            flex_renderer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(flex_renderer)
        except Exception as e2:
            raise ImportError(
                f"Failed to load flex_renderer from {_so_path}. "
                f"Original error: {e}, Secondary error: {e2}. "
                "The .so file may need to be recompiled for your Python version."
            )
else:
    raise ImportError(
        "flex_renderer .so file not found in gl_renderer_src directory. "
        "Please compile the renderer first."
    )

__all__ = ['flex_renderer']
