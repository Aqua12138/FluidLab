# Copyright (c) 2024-2025, RheoAgent Project
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""渲染器模块

只支持 GL 渲染器。
"""

from rheo.utils.misc import is_headless

# GL 渲染器
GLRenderer = None

def get_gl_renderer():
    """获取 GL 渲染器类"""
    global GLRenderer
    if GLRenderer is None:
        if is_headless():
            raise ImportError("GLRenderer is not available in headless mode.")
        try:
            from .gl_renderer import GLRenderer as _GLRenderer
            GLRenderer = _GLRenderer
        except ImportError as e:
            raise ImportError(
                f"GLRenderer is not available. Please compile flex_renderer first. Error: {e}"
            )
    return GLRenderer

def __getattr__(name):
    """延迟加载渲染器类"""
    if name == 'GLRenderer':
        return get_gl_renderer()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['GLRenderer', 'get_gl_renderer']
