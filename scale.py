import numpy as np


def calculate_sim_params(
        # --- 现实参数 (SI 单位) ---
        real_length_scale=0.1,  # 特征长度 (例如杯子高度/开口直径), 米
        real_g=9.8,  # 重力加速度, m/s^2
        real_rho=1000.0,  # 密度, kg/m^3 (水)

        # 现实流变参数
        real_tau0=20.0,  # 屈服应力, Pa (番茄酱)
        real_K=10.0,  # 稠度, Pa·s^n
        real_n=0.3,  # 流动指数 (无量纲，Sim和Real一样)

        # --- 仿真设置 (Target) ---
        sim_length_scale=0.1,  # 仿真里对应的长度 (例如你看杯子占了半个屏幕)
        sim_g=9.8,  # 仿真重力 (建议设为1.0以获得更稳定的梯度)
        sim_rho=1.0  # 仿真密度 (建议设为1.0)
):
    # 1. 计算基础缩放因子 (Scale Factors)
    # S = Sim / Real
    S_L = sim_length_scale / real_length_scale
    S_g = sim_g / real_g
    S_rho = sim_rho / real_rho

    # 2. 推导导出缩放因子
    # 基于 Froude Similarity (重力相似性)
    S_v = np.sqrt(S_g * S_L)  # 速度缩放
    S_t = S_L / S_v  # 时间缩放 (1秒代表多久)

    # 基于 Force/Stress Scaling
    S_tau = S_rho * S_g * S_L  # 应力缩放 (Pa -> SimStress)

    # 基于 Reynolds Similarity (粘性相似性)
    # K 单位: Pa * s^n -> Stress * Time^n
    S_K = S_tau * (S_t ** real_n)

    # 3. 计算结果
    sim_tau0 = real_tau0 * S_tau
    sim_K = real_K * S_K

    print("-" * 30)
    print(f"=== Scaling Report ===")
    print(f"[Scales] Length: {S_L:.2f}x, Gravity: {S_g:.2f}x, Density: {S_rho:.2e}x")
    print(f"[Scales] Time:   {S_t:.2f}x (Sim 1.0s = Real {1 / S_t:.2f}s)")
    print(f"[Scales] Stress: {S_tau:.2e}x")
    print("-" * 30)
    print(f"=== Input (Real SI) ===")
    print(f"Tau0: {real_tau0} Pa")
    print(f"K:    {real_K} Pa·s^{real_n}")
    print("-" * 30)
    print(f"=== Output (FluidLab Config) ===")
    print(f"YIELD_STRESS (sim): {sim_tau0:.4f}")
    print(f"CONSISTENCY  (sim): {sim_K:.4f}")
    print(f"FLOW_INDEX   (sim): {real_n} (不变)")
    print(f"RHO          (sim): {sim_rho}")

    return sim_tau0, sim_K


# --- 案例测试：番茄酱 (Ketchup) ---
# 假设现实中：杯子10cm, 番茄酱屈服20Pa, 粘度10, 变稀0.3
# 假设仿真中：杯子是0.5大小, 重力1.0, 密度1.0
calculate_sim_params(
    real_length_scale=0.1,
    real_g=9.8,
    real_rho=1100.0,  # 番茄酱比水稍重
    real_tau0=20.0,
    real_K=10.0,
    real_n=0.3,

    sim_length_scale=0.1,
    sim_g=9.8,  # 假设我们保持重力数值9.8不变看看效果
    sim_rho=1.0
)