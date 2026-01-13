# 碰撞检测分析与改进方案

## 当前碰撞逻辑分析

### 1. 碰撞检测位置
- **网格点检测**：在 `grid_op()` 中，碰撞检测只在**网格点**上进行（`I*self.dx`）
- **粒子检测**：在 `g2p()` 中，**没有**对静态物体进行碰撞检测
- **问题**：如果粒子在两个网格点之间快速移动，可能会漏检

### 2. 穿模问题根源

#### 网格分辨率限制
```python
# mpm_simulator.py
self.n_grid = int(64 * quality)  # 默认 quality=1
self.dx = 1 / self.n_grid        # dx ≈ 0.0156 (quality=1)
```

- 如果物体厚度 < `dx`（约 0.0156），可能无法检测到碰撞
- 桌子厚度为 0.1（scale y=1），理论上应该足够，但：
  - SDF 分辨率可能不够高
  - 网格分辨率可能不够高
  - 没有位置修正，只有速度修正

#### SDF 分辨率限制
- 默认 `sdf_res=128`
- SDF 覆盖范围：`[-0.6, 0.6]`（归一化后的网格空间）
- 对于薄物体，可能需要更高的 SDF 分辨率

### 3. 摩擦处理

**已实现摩擦**：
```python
# static.py line 96-97
rel_v_t_friction = rel_v_t / rel_v_t_norm * max(0, rel_v_t_norm + normal_component * self.friction)
```

**摩擦系数**：
- `PLATE: 0.1`（在 `macros.py` 中定义）
- 摩擦系数范围：0.0（无摩擦）到 10.0+（完全粘附）

**摩擦模型**：
- 使用库仑摩擦模型
- 切向速度 = 原始切向速度 - 摩擦损失
- 摩擦损失 = `normal_component * friction`

## 改进方案

### 方案 1：提高分辨率（最简单）

#### 1.1 提高 SDF 分辨率
```python
self.taichi_env.add_static(
    file='table.obj',
    pos=(0.5, 0.15, 0.5),
    euler=(0.0, 0.0, 0.0),
    scale=(1, 1, 1),
    material=PLATE,
    has_dynamics=True,
    sdf_res=256,  # 从 128 提高到 256（或更高）
)
```

**优点**：
- 简单，只需修改一个参数
- 提高碰撞检测精度

**缺点**：
- SDF 生成时间增加（首次运行）
- 内存占用增加

#### 1.2 提高网格分辨率
```python
self.taichi_env = TaichiEnv(
    dim=3,
    particle_density=1e6,
    max_substeps_local=20,
    gravity=(0.0, -9.8, 0.0),
    horizon=self.horizon,
    quality=2,  # 从默认 1 提高到 2，网格分辨率从 64 提高到 128
)
```

**优点**：
- 提高整体模拟精度
- 减少穿模概率

**缺点**：
- 计算量增加 8 倍（3D）
- 内存占用增加

### 方案 2：在粒子层面添加碰撞检测（推荐）

修改 `g2p()` 函数，在粒子位置也进行碰撞检测：

```python
# 在 mpm_simulator.py 的 g2p() 函数中添加
@ti.kernel
def g2p(self, f: ti.i32):
    for p in range(self.n_particles):
        if self.particles_ng[f, p].used:
            # ... 现有代码 ...
            
            # collide with statics (新增)
            if ti.static(self.n_statics > 0):
                new_x_tmp = self.particles[f, p].x + self.dt * new_v
                for i in ti.static(range(self.n_statics)):
                    # 检查粒子位置是否在静态物体内
                    if self.statics[i].is_collide(new_x_tmp):
                        # 将粒子推出物体
                        signed_dist = self.statics[i].sdf(new_x_tmp)
                        normal_vec = self.statics[i].normal(new_x_tmp)
                        # 位置修正：将粒子推出物体表面
                        push_distance = -signed_dist + 0.001  # 小偏移避免重复碰撞
                        new_x_tmp = new_x_tmp + normal_vec * push_distance
                        # 速度修正
                        new_v = self.statics[i].collide(new_x_tmp, new_v)
            
            # advect to next frame
            self.particles[f+1, p].x = new_x_tmp
            self.particles[f+1, p].v = new_v
            # ...
```

**优点**：
- 直接检测粒子位置，避免漏检
- 添加位置修正，防止穿模
- 不影响网格分辨率

**缺点**：
- 需要修改核心代码
- 计算量略有增加

### 方案 3：增加安全距离（保守方案）

在碰撞检测中添加安全距离：

```python
# 在 static.py 的 collide() 函数中
@ti.func
def collide(self, pos_world, mat_v):
    if ti.static(self.has_dynamics):
        signed_dist = self.sdf(pos_world)
        safety_margin = 0.01  # 安全距离
        if signed_dist <= safety_margin:  # 提前检测
            # ... 现有碰撞处理 ...
```

**优点**：
- 简单，只需修改一个函数
- 提前检测，减少穿模

**缺点**：
- 可能产生"浮空"效果
- 需要调整安全距离

### 方案 4：使用软碰撞（适合薄物体）

参考 `Dynamic` 类的 `softness` 参数：

```python
self.taichi_env.add_static(
    file='table.obj',
    pos=(0.5, 0.15, 0.5),
    euler=(0.0, 0.0, 0.0),
    scale=(1, 1, 1),
    material=PLATE,
    has_dynamics=True,
    sdf_res=256,
    softness=100.0,  # 添加软碰撞参数
)
```

但需要修改 `Static` 类以支持 `softness`。

## 推荐配置

### 快速改进（无需修改代码）
```python
# 在 tablematerials_env.py 中
self.taichi_env = TaichiEnv(
    dim=3,
    particle_density=1e6,
    max_substeps_local=20,
    gravity=(0.0, -9.8, 0.0),
    horizon=self.horizon,
    quality=2,  # 提高网格分辨率
)

self.taichi_env.add_static(
    file='table.obj',
    pos=(0.5, 0.15, 0.5),
    euler=(0.0, 0.0, 0.0),
    scale=(1, 1, 1),
    material=PLATE,
    has_dynamics=True,
    sdf_res=256,  # 提高 SDF 分辨率
)
```

### 摩擦系数调整
如果需要更强的摩擦（防止滑动）：
```python
# 在 macros.py 中
FRICTION = {
    # ...
    PLATE   : 0.5,  # 从 0.1 提高到 0.5（中等摩擦）
    # 或
    PLATE   : 1.0,  # 高摩擦（类似橡胶）
}
```

## 总结

1. **摩擦已实现**：当前摩擦系数为 0.1，可以调整
2. **穿模问题**：主要由于网格分辨率限制，建议：
   - 提高 SDF 分辨率到 256
   - 提高网格分辨率（quality=2）
   - 或实现粒子层面的碰撞检测（需要修改核心代码）

3. **最佳实践**：
   - 对于薄物体（< 0.02），使用更高的 SDF 分辨率
   - 对于快速移动的粒子，考虑在粒子层面添加碰撞检测
   - 根据材料特性调整摩擦系数
