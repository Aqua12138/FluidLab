# CONSISTENCY 作为主要粘度参数

## 设计原则

**CONSISTENCY (K) 是 Herschel–Bulkley 模型的主要粘度参数。**

### 核心概念

1. **CONSISTENCY (K)**：H-B 模型的核心粘度参数
   - 直接参与有效粘度计算：`μ_eff = τ_y项 + K × γ̇^(n-1)`
   - 当 n=1 且 τ_y=0 时，`μ_eff = K`（牛顿流体，K 就是动态粘度）
   - 这是**主要参数**，应该优先设置

2. **MU (μ)**：辅助参数
   - 用于向后兼容（未启用 H-B 时作为固定粘度）
   - 用于计算 `mu_max = μ × 1000`（数值稳定性）
   - 当启用 H-B 时，MU 从 CONSISTENCY 派生

## 实现逻辑

### 在 `mpm_simulator.py` 中

```python
# 当启用 H-B 模型时
if use_herschel_bulkley:
    # 使用 CONSISTENCY 作为 mu_base（用于 mu_max 计算）
    mu_base_for_max = consistency
    # 如果 CONSISTENCY 为 0，fallback 到 MU（向后兼容）
    if mu_base_for_max < 1e-6:
        mu_base_for_max = mu
    
    mu_eff = compute_effective_viscosity(
        shear_rate,
        yield_stress,
        consistency,  # 主要参数
        flow_index,
        mu_base_for_max,  # 从 CONSISTENCY 派生
        m
    )
```

### 参数设置建议

对于 H-B 材料，应该：

1. **优先设置 CONSISTENCY**：这是主要粘度参数
2. **MU 与 CONSISTENCY 保持一致**：用于向后兼容和 mu_max 计算
3. **特殊情况**：
   - 如果 CONSISTENCY = 0，系统会 fallback 到 MU
   - 如果未启用 H-B（n=1, τ_y=0），使用 MU 作为固定粘度

## 参数关系总结

| 参数 | 角色 | 用途 | 优先级 |
|------|------|------|--------|
| **CONSISTENCY (K)** | 主要参数 | H-B 模型的核心粘度参数 | ⭐⭐⭐ 最高 |
| **MU (μ)** | 辅助参数 | 向后兼容、mu_max 计算 | ⭐⭐ 次要 |
| **YIELD_STRESS (τ_y)** | 核心参数 | 屈服应力 | ⭐⭐⭐ 最高 |
| **FLOW_INDEX (n)** | 核心参数 | 流动指数 | ⭐⭐⭐ 最高 |
| **RHO (ρ)** | 基础参数 | 密度 | ⭐⭐⭐ 必需 |
| **LAMDA (λ)** | 基础参数 | 体积模量 | ⭐⭐⭐ 必需 |

## 实际考虑的参数（5个核心）

对于 H-B 模型，实际需要设置的**核心参数**是：

1. **CONSISTENCY (K)** - 主要粘度参数
2. **YIELD_STRESS (τ_y)** - 屈服应力
3. **FLOW_INDEX (n)** - 流动指数
4. **RHO (ρ)** - 密度
5. **LAMDA (λ)** - 体积模量

**MU** 作为辅助参数，通常设置为与 CONSISTENCY 相同的值。

## 示例：THICK_CREAM 材料

```python
# 主要参数（核心）
CONSISTENCY[THICK_CREAM] = 350.0  # 主要粘度参数
YIELD_STRESS[THICK_CREAM] = 15.0  # 屈服应力
FLOW_INDEX[THICK_CREAM] = 0.65    # 流动指数（剪切变稀）
RHO[THICK_CREAM] = 0.8            # 密度
LAMDA[THICK_CREAM] = 277.78       # 体积模量

# 辅助参数（与 CONSISTENCY 保持一致）
MU[THICK_CREAM] = 350.0           # 与 CONSISTENCY 相同，用于向后兼容
```

## 数学关系

### 当 n=1 且 τ_y=0（牛顿流体）

```
μ_eff = K  （常数）
```

此时：**CONSISTENCY = 有效粘度 = MU**（概念上等价）

### 当 n=1 且 τ_y>0（Bingham 模型）

```
μ_eff = τ_y × (1-exp(-m×γ̇))/γ̇ + K
```

此时：**CONSISTENCY (K)** 是有效粘度的基础部分

### 当 n≠1（一般 H-B 模型）

```
μ_eff = τ_y × (1-exp(-m×γ̇))/γ̇ + K × γ̇^(n-1)
```

此时：**CONSISTENCY (K)** 控制幂律项的强度

## 结论

- **CONSISTENCY 是主要参数**：直接控制 H-B 模型的粘度
- **MU 是辅助参数**：用于向后兼容和数值稳定性
- **当 n=1 且 τ_y=0 时**：CONSISTENCY 和 MU 在概念上等价
- **实际设置时**：优先设置 CONSISTENCY，MU 通常与 CONSISTENCY 保持一致
