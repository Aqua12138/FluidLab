# FluidLab: Herschel-Bulkley 非牛顿流体材料演示

本文档展示了 FluidLab 中基于 **Herschel-Bulkley 本构模型** 实现的多种非牛顿流体材料的仿真效果。

## Herschel-Bulkley 模型简介

Herschel-Bulkley 模型是描述非牛顿流体行为的经典本构模型，其有效粘度公式为：

```
μ_eff = (τ_y / γ̇) + K × (γ̇)^(n-1)
```

其中：
- **τ_y (YIELD_STRESS)**: 屈服应力 - 当剪切应力小于此值时，材料表现为固体
- **K (CONSISTENCY)**: 稠度系数 - 控制材料的基础粘度
- **n (FLOW_INDEX)**: 流动指数 - 决定材料的剪切特性
  - n = 1: 牛顿流体（恒定粘度）
  - n < 1: 剪切变稀（pseudoplastic）- 剪切越快，粘度越低
  - n > 1: 剪切增稠（dilatant）- 剪切越快，粘度越高
- **ρ (RHO)**: 密度 - 材料的质量密度

---

## 材料演示

### 1. 水 (Water) - 牛顿流体

![Water](gif/water.gif)

**材料特性**: 标准牛顿流体，粘度恒定，无屈服应力

| 参数 | 值 | 说明 |
|------|-----|------|
| **RHO (密度)** | 1.0 | 归一化密度，以水为基准 |
| **YIELD_STRESS (屈服应力)** | 0.0 | 无屈服应力，始终为流体状态 |
| **CONSISTENCY (稠度系数)** | 1.0 | 低粘度，流动性好 |
| **FLOW_INDEX (流动指数)** | 1.0 | n=1，牛顿流体，粘度不随剪切速率变化 |

**物理行为**: 典型的低粘度牛顿流体，流动顺畅，无剪切依赖性。

---

### 2. 蜂蜜 (Honey) - 剪切变稀流体

![Honey](gif/honey.gif)

**材料特性**: 高粘度剪切变稀流体，搅拌时粘度降低

| 参数 | 值 | 说明 |
|------|-----|------|
| **RHO (密度)** | 1.4 | 比水重，密度较高 |
| **YIELD_STRESS (屈服应力)** | 0.0 | 无屈服应力 |
| **CONSISTENCY (稠度系数)** | 2000.0 | 高粘度，流动性差 |
| **FLOW_INDEX (流动指数)** | 0.5 | n<1，强剪切变稀，搅拌时明显变稀 |

**物理行为**: 静止时非常粘稠，但在高剪切速率下（如搅拌）粘度显著降低，表现出"拉丝"特性。

---

### 3. 血液 (Blood) - 弱剪切变稀流体

![Blood](gif/blood.gif)

**材料特性**: 轻微剪切变稀，具有轻微屈服应力

| 参数 | 值 | 说明 |
|------|-----|------|
| **RHO (密度)** | 1.05 | 略重于水 |
| **YIELD_STRESS (屈服应力)** | 5.0 | 轻微屈服应力，需要一定剪切力才能流动 |
| **CONSISTENCY (稠度系数)** | 50.0 | 中等粘度 |
| **FLOW_INDEX (流动指数)** | 0.7 | n<1，轻微剪切变稀 |

**物理行为**: 在微血管中表现出非牛顿特性，但在宏观流动中接近牛顿流体。具有轻微的屈服应力，需要克服一定阈值才能开始流动。

---

### 4. 玉米淀粉浆 (Cornstarch) - 剪切增稠流体

![Cornstarch](gif/cornstarch.gif)

**材料特性**: 剪切增稠流体（Dilatant），慢速流动为液体，快速冲击为固体

| 参数 | 值 | 说明 |
|------|-----|------|
| **RHO (密度)** | 1.1 | 略重于水 |
| **YIELD_STRESS (屈服应力)** | 0.0 | 无屈服应力 |
| **CONSISTENCY (稠度系数)** | 100.0 | 中等稠度 |
| **FLOW_INDEX (流动指数)** | 1.5 | n>1，剪切增稠，快速剪切时粘度急剧增大 |

**物理行为**: 典型的"非牛顿流体"行为 - 慢速推拉时像液体一样流动，但快速冲击时会瞬间变硬，表现出类似固体的特性。这就是著名的"Oobleck"效应。

---

### 5. 酸奶 (Yogurt) - 塑性剪切变稀流体

![Yogurt](gif/yogurt.gif)

**材料特性**: 具有屈服应力的剪切变稀流体，能保持形状

| 参数 | 值 | 说明 |
|------|-----|------|
| **RHO (密度)** | 1.05 | 略重于水 |
| **YIELD_STRESS (屈服应力)** | 12.0 | 有屈服应力，能保持形状 |
| **CONSISTENCY (稠度系数)** | 400.0 | 较高粘度 |
| **FLOW_INDEX (流动指数)** | 0.6 | n<1，剪切变稀 |

**物理行为**: 具有屈服应力的塑性流体。在静止状态下能保持形状（如勺子挖出的坑洼），但一旦施加足够的剪切力，就会开始流动。流动时表现出剪切变稀特性，搅拌越快越顺滑。

---

## 参数对比表

| 材料 | RHO | YIELD_STRESS | CONSISTENCY | FLOW_INDEX | 流体类型 |
|------|-----|--------------|-------------|------------|----------|
| **Water** | 1.0 | 0.0 | 1.0 | 1.0 | 牛顿流体 |
| **Honey** | 1.4 | 0.0 | 2000.0 | 0.5 | 剪切变稀 |
| **Blood** | 1.05 | 5.0 | 50.0 | 0.7 | 弱剪切变稀+屈服 |
| **Cornstarch** | 1.1 | 0.0 | 100.0 | 1.5 | 剪切增稠 |
| **Yogurt** | 1.05 | 12.0 | 400.0 | 0.6 | 剪切变稀+屈服 |

---

## 技术实现

FluidLab 使用 **Material Point Method (MPM)** 进行流体仿真，并实现了完整的 Herschel-Bulkley 本构模型：

1. **可微分性**: 所有参数支持梯度反向传播，可用于优化任务
2. **数值稳定性**: 
   - 使用 Papanastasiou 正则化处理屈服应力项
   - 对剪切增稠材料（n>1）实施粘度上限限制
   - 剪切速率和幂律项的截断保护
3. **物理真实性**: 基于 Froude-Reynolds 相似性理论，保证无量纲数不变性

所有材料参数定义在 `fluidlab/configs/macros.py` 中，可以直接修改以调整材料行为。

---

## 运行演示

要运行材料演示，可以使用以下命令：

```bash
python fluidlab/run.py \
  --cfg_file configs/exp_tablematerials.yaml \
  --auto_rotate \
  --auto_rotate_steps 350 \
  --auto_rotate_speed 0.005 \
  --auto_rotate_wait_steps 100 \
  --auto_rotate_stop_steps 2000 \
  --renderer_type GL
```

**参数说明**:
- `--cfg_file configs/exp_tablematerials.yaml`: 使用材料演示环境配置
- `--auto_rotate`: 启用自动旋转模式
- `--auto_rotate_steps 350`: 旋转阶段执行 350 步
- `--auto_rotate_speed 0.005`: 旋转速度（角速度）
- `--auto_rotate_wait_steps 100`: 初始暂停 100 步，等待流体稳定
- `--auto_rotate_stop_steps 2000`: 旋转后停止 2000 步，观察流体稳定
- `--renderer_type GL`: 使用 GL 渲染器（更高质量的视觉效果）

**运行流程**:
1. **等待阶段** (100 步): 流体在静止状态下稳定
2. **旋转阶段** (350 步): 玻璃杯以指定速度旋转，观察不同材料的流动行为
3. **停止阶段** (2000 步): 停止旋转，观察流体的稳定过程和最终状态

---

## 相关文件

- 材料参数定义: `fluidlab/configs/macros.py`
- MPM 求解器: `fluidlab/fluidengine/simulators/mpm_simulator.py`
- 环境配置: `fluidlab/envs/tablematerials_env.py`
- 实验配置: `fluidlab/configs/exp_tablematerials.yaml`

---

## 引用

如果您使用了 FluidLab 的 Herschel-Bulkley 实现，请引用原始论文：

```bibtex
@inproceedings{xian2023fluidlab,
  title={FluidLab: A Differentiable Environment for Benchmarking Complex Fluid Manipulation},
  author={Xian, Zhou and Zhu, Bo and Xu, Zhenjia and Tung, Hsiao-Yu and Torralba, Antonio and Fragkiadaki, Katerina and Gan, Chuang},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}
}
```



# FluidLab GLRenderer 配置指南

## 1. 安装 Docker

```bash
curl -fsSL https://get.docker.com | bash
sudo usermod -aG docker $USER
newgrp docker
```

## 2. 安装 NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## 3. 验证 GPU 访问

```bash
sudo docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

## 4. 构建 Docker 镜像

```bash
cd /home/xu/FluidLab
sudo docker build -t fluidlab-glrenderer fluidlab/fluidengine/renderers/gl_renderer_src
```

## 5. 编译 GL 渲染器

### 5.1 开放 X11 显示权限

```bash
xhost +local:root
```

### 5.2 启动容器

```bash
cd /home/xu/FluidLab
sudo docker run \
  -v ${PWD}/fluidlab/fluidengine/renderers/gl_renderer_src:/workspace \
  -v /home/xu/miniconda3:/home/xu/miniconda3 \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -it fluidlab-glrenderer:latest bash
```

### 5.3 容器内编译

```bash
source /home/xu/miniconda3/bin/activate rheo
cd /workspace
source prepare.sh
source compile.sh
exit
```

编译完成后，`.so` 文件会自动保存到宿主机目录。

## 6. 解决 CUDA-OpenGL 互操作问题

### 6.1 检查显卡配置

```bash
lspci | grep -i vga
prime-select query
nvidia-smi
```

### 6.2 临时使用 NVIDIA GPU 渲染

```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python fluidlab/run.py \
  --cfg_file configs/exp_tablematerials.yaml \
  --auto_rotate \
  --auto_rotate_steps 350 \
  --auto_rotate_speed 0.005 \
  --auto_rotate_wait_steps 100 \
  --auto_rotate_stop_steps 2000 \
  --renderer_type GL
```

### 6.3 永久切换到 NVIDIA 专用模式

```bash
sudo prime-select nvidia
sudo reboot
```

重启后，OpenGL 将默认使用 NVIDIA GPU，无需环境变量。

## 7. 验证 GL 渲染器

```bash
python fluidlab/run.py \
  --cfg_file configs/exp_tablematerials.yaml \
  --auto_rotate \
  --auto_rotate_steps 350 \
  --auto_rotate_speed 0.005 \
  --auto_rotate_wait_steps 100 \
  --auto_rotate_stop_steps 2000 \
  --renderer_type GL
```