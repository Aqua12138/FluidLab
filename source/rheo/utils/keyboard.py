# Copyright (c) 2024-2025, RheoAgent Project
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""键盘控制模块

使用 pynput 实现全局键盘监听，支持长按检测。
"""

import numpy as np
from rheo.utils.misc import is_headless

# 只在非 headless 模式下导入 pynput
if not is_headless():
    try:
        from pynput import keyboard
        PYNPUT_AVAILABLE = True
    except ImportError:
        PYNPUT_AVAILABLE = False
        print("Warning: pynput not available. Keyboard control disabled.")
else:
    PYNPUT_AVAILABLE = False


class KeyboardPolicy:
    """基础键盘策略类
    
    使用 pynput 全局监听键盘，支持长按检测。
    位置控制由环境 reset 管理，此类只负责速度/动作输出。
    """
    
    def __init__(self, v_lin=0.01, v_ang=0.01):
        """
        Args:
            v_lin: 线速度幅值
            v_ang: 角速度幅值
        """
        self.keys_activated = set()
        self.linear_v_mag = v_lin
        self.angular_v_mag = v_ang
        self.running = True
        
        if PYNPUT_AVAILABLE:
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release
            )
            self.listener.start()
        else:
            self.listener = None
    
    def _on_press(self, key):
        """按键按下事件"""
        try:
            char = key.char.lower() if hasattr(key, 'char') and key.char else None
            if char:
                self.keys_activated.add(char)
        except AttributeError:
            # 处理特殊键
            if key == keyboard.Key.esc:
                self.running = False
            elif key == keyboard.Key.space:
                self.keys_activated.add('space')
    
    def _on_release(self, key):
        """按键释放事件"""
        try:
            char = key.char.lower() if hasattr(key, 'char') and key.char else None
            if char and char in self.keys_activated:
                self.keys_activated.discard(char)
        except AttributeError:
            if key == keyboard.Key.space:
                self.keys_activated.discard('space')
    
    def get_action_v(self, i=0):
        """获取速度动作 - 子类应重写"""
        return np.zeros(6)
    
    def is_running(self):
        """检查是否继续运行（ESC 退出）"""
        return self.running
    
    def stop(self):
        """停止监听"""
        self.running = False
        if self.listener:
            self.listener.stop()


class KeyboardPolicy_6DOF(KeyboardPolicy):
    """6自由度键盘控制策略
    
    按键映射:
        W/S: X轴移动 (前/后)
        A/D: Z轴移动 (左/右)
        Q/E: Y轴移动 (上/下)
        I/K: Roll 旋转
        J/L: Pitch 旋转
        U/O: Yaw 旋转
    """
    
    def get_action_v(self, i=0):
        action_v = np.zeros(6)
        
        # 线性移动
        if 'w' in self.keys_activated:
            action_v[0] += self.linear_v_mag
        if 's' in self.keys_activated:
            action_v[0] -= self.linear_v_mag
        if 'q' in self.keys_activated:
            action_v[1] += self.linear_v_mag
        if 'e' in self.keys_activated:
            action_v[1] -= self.linear_v_mag
        if 'd' in self.keys_activated:
            action_v[2] += self.linear_v_mag
        if 'a' in self.keys_activated:
            action_v[2] -= self.linear_v_mag
        
        # 旋转
        if 'i' in self.keys_activated:
            action_v[3] += self.angular_v_mag
        if 'k' in self.keys_activated:
            action_v[3] -= self.angular_v_mag
        if 'l' in self.keys_activated:
            action_v[4] += self.angular_v_mag
        if 'j' in self.keys_activated:
            action_v[4] -= self.angular_v_mag
        if 'o' in self.keys_activated:
            action_v[5] += self.angular_v_mag
        if 'u' in self.keys_activated:
            action_v[5] -= self.angular_v_mag
        
        return action_v


class KeyboardPolicy_XYZ(KeyboardPolicy):
    """XYZ 线性移动键盘控制策略"""
    
    def get_action_v(self, i=0):
        action_v = np.zeros(3)
        
        if 'w' in self.keys_activated:
            action_v[0] += self.linear_v_mag
        if 's' in self.keys_activated:
            action_v[0] -= self.linear_v_mag
        if 'q' in self.keys_activated:
            action_v[1] += self.linear_v_mag
        if 'e' in self.keys_activated:
            action_v[1] -= self.linear_v_mag
        if 'd' in self.keys_activated:
            action_v[2] += self.linear_v_mag
        if 'a' in self.keys_activated:
            action_v[2] -= self.linear_v_mag
        
        return action_v


class KeyboardPolicy_Rotation(KeyboardPolicy):
    """仅旋转的键盘控制策略 (Z/X 控制 Yaw)"""
    
    def get_action_v(self, i=0):
        action_v = np.zeros(6)
        
        if 'z' in self.keys_activated:
            action_v[5] += self.angular_v_mag
        if 'x' in self.keys_activated:
            action_v[5] -= self.angular_v_mag
        
        return action_v


# 兼容旧命名
KeyboardPolicy_vxy_wz = KeyboardPolicy_6DOF
KeyboardPolicy_vxy = KeyboardPolicy_XYZ
KeyboardPolicy_wz = KeyboardPolicy_Rotation
