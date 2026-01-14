############ material type #############
WATER          = 0
MILK           = 1
COFFEE         = 2
ELASTIC        = 3
ICECREAM       = 4
RIGID          = 5
RIGID_HEAVY    = 6
RIGID_LIGHT    = 7
MILK_VIS       = 8
COFFEE_VIS     = 9
ELASTIC_DEMO   = 10
PLASTIC_DEMO   = 11
INVISCID_DEMO  = 12
VISCOUS_DEMO   = 13
INVISCID_DEMO2 = 14
INVISCID_DEMO3 = 15
ICECREAM1      = 16
THICK_CREAM    = 17  # 非牛顿粘稠材料（剪切变稀，有屈服应力）

# Herschel-Bulkley 模型常见材料
WATER_NEWTON   = 18  # 牛顿流体（n=1, τ_y=0）
HONEY          = 19  # 蜂蜜（剪切变稀，n<1）
KETCHUP        = 20  # 番茄酱（剪切变稀+屈服应力，n<1, τ_y>0）
TOOTHPASTE     = 21  # 牙膏（剪切变稀+屈服应力，n<1, τ_y>0）
BLOOD          = 22  # 血液（剪切变稀，n<1）
PAINT          = 23  # 油漆（剪切变稀，n<1）
CORNSTARCH     = 24  # 玉米淀粉浆（剪切增稠，n>1）
SILLY_PUTTY    = 25  # 橡皮泥（剪切增稠，n>1）
YOGURT         = 26  # 酸奶（剪切变稀+屈服应力，n<1, τ_y>0）

CUP       = 50
TANK      = 51
LADDLE    = 52
POURER    = 53
DISPENSER = 54
CONE      = 55
ROBOT     = 56
BOTTLE    = 57
PILLAR    = 58
STIRRER   = 59
PLATE     = 60
BOWL      = 61
TABLE     = 62  # 桌子（静态物体，高摩擦）

FRAME    = 100
TARGET   = 101
EFFECTOR = 102

MAT_LIQUID              = 200
MAT_PLASTO_ELASTIC      = 201
MAT_ELASTIC             = 202
MAT_RIGID               = 203
MAT_PLASTO_ELASTIC_DEMO = 204

############ material name #############
MAT_NAME = {
    WATER          : 'water',
    INVISCID_DEMO  : 'inviscid-demo',
    INVISCID_DEMO2 : 'inviscid-demo2',
    INVISCID_DEMO3 : 'inviscid-demo3',
    VISCOUS_DEMO   : 'viscous-demo',
    MILK           : 'milk',
    COFFEE         : 'coffee',
    ELASTIC        : 'elastic',
    ELASTIC_DEMO   : 'elastic-demo',
    PLASTIC_DEMO   : 'plastic-demo',
    RIGID          : 'rigid',
    RIGID_HEAVY    : 'rigid-heavy',
    RIGID_LIGHT    : 'rigid-light',
    ICECREAM       : 'ice-cream',
    ICECREAM1       : 'ice-cream1',
    MILK_VIS       : 'milk-viscous',
    COFFEE_VIS     : 'coffee-viscous',
    THICK_CREAM    : 'thick-cream',
    WATER_NEWTON   : 'water-newton',
    HONEY          : 'honey',
    KETCHUP        : 'ketchup',
    TOOTHPASTE     : 'toothpaste',
    BLOOD          : 'blood',
    PAINT          : 'paint',
    CORNSTARCH     : 'cornstarch',
    SILLY_PUTTY    : 'silly-putty',
    YOGURT         : 'yogurt',
}

############ material class #############
MAT_CLASS = {
    WATER          : MAT_LIQUID,
    INVISCID_DEMO  : MAT_LIQUID,
    INVISCID_DEMO2 : MAT_LIQUID,
    INVISCID_DEMO3 : MAT_LIQUID,
    VISCOUS_DEMO   : MAT_LIQUID,
    MILK           : MAT_LIQUID,
    COFFEE         : MAT_LIQUID,
    ELASTIC        : MAT_ELASTIC,
    ELASTIC_DEMO   : MAT_ELASTIC,
    PLASTIC_DEMO   : MAT_PLASTO_ELASTIC_DEMO,
    RIGID          : MAT_RIGID,
    RIGID_HEAVY    : MAT_RIGID,
    RIGID_LIGHT    : MAT_RIGID,
    ICECREAM       : MAT_PLASTO_ELASTIC,
    ICECREAM1       : MAT_PLASTO_ELASTIC,
    MILK_VIS       : MAT_LIQUID,
    COFFEE_VIS     : MAT_LIQUID,
    THICK_CREAM    : MAT_LIQUID,
    WATER_NEWTON   : MAT_LIQUID,
    HONEY          : MAT_LIQUID,
    KETCHUP        : MAT_LIQUID,
    TOOTHPASTE     : MAT_LIQUID,
    BLOOD          : MAT_LIQUID,
    PAINT          : MAT_LIQUID,
    CORNSTARCH     : MAT_LIQUID,
    SILLY_PUTTY    : MAT_LIQUID,
    YOGURT         : MAT_LIQUID,
}

############ default color #############
COLOR = {
    WATER          : (0.3, 0.8, 1.0, 0.0),
    # WATER          : (0.3, 0.8, 1.0, 0.0),
    INVISCID_DEMO  : (0.3, 0.8, 1.0, 1.0),
    INVISCID_DEMO2 : (1.0, 0.2, 0.1, 1.0),
    INVISCID_DEMO3 : (1.0, 0.2, 0.1, 1.0),
    VISCOUS_DEMO   : (1.0, 0.2, 0.1, 1.0),
    # INVISCID_DEMO  : (0.3, 0.8, 1.0, 0.0),
    # INVISCID_DEMO2 : (0.3, 0.8, 1.0, 0.0),
    # INVISCID_DEMO3 : (1.0, 0.2, 0.1, 0.2),
    # VISCOUS_DEMO   : (1.0, 0.2, 0.1, 0.2),
    MILK           : (0.9, 0.9, 0.9, 1.0),
    COFFEE         : (0.58, 0.42, 0.22, 1.0),
    # COFFEE       : (0.48, 0.32, 0.12, 0.8),
    ELASTIC        : (1.0, 1.0, 1.0, 1.0),
    ELASTIC_DEMO   : (1.0, 1.0, 1.0, 1.0),
    PLASTIC_DEMO   : (1.0, 1.0, 1.0, 1.0),
    ICECREAM       : (1.0, 1.0, 1.0, 1.0),
    ICECREAM1       : (1.0, 1.0, 1.0, 1.0),
    RIGID          : (1.0, 0.5, 0.5, 1.0),
    RIGID_HEAVY    : (1.0, 0.5, 0.5, 1.0),
    RIGID_LIGHT    : (1.0, 0.5, 0.5, 1.0),
    MILK_VIS       : (0.9, 0.9, 0.9, 1.0),
    COFFEE_VIS     : (0.58, 0.42, 0.22, 1.0),
    THICK_CREAM    : (0.95, 0.95, 0.9, 1.0),  # 浅奶油色
    WATER_NEWTON   : (0.2, 0.5, 0.9, 0.0),   # 蓝色（水）
    HONEY          : (1.0, 0.7, 0.0, 0.5), # 金黄透亮色（蜂蜜）- 平衡的金黄色
    KETCHUP        : (0.9, 0.1, 0.1, 1.0),   # 红色（番茄酱）
    TOOTHPASTE     : (0.95, 0.95, 1.0, 1.0), # 白色（牙膏）
    BLOOD          : (0.6, 0.1, 0.1, 1.0),   # 深红色（血液）
    PAINT          : (0.9, 0.7, 0.1, 1.0),   # 黄色（油漆）
    CORNSTARCH     : (1.0, 1.0, 0.9, 1.0),   # 浅黄色（玉米淀粉）
    SILLY_PUTTY    : (0.8, 0.6, 0.9, 1.0),   # 紫色（橡皮泥）
    YOGURT         : (1.0, 0.98, 0.9, 1.0),  # 乳白色（酸奶）

    CUP       : (0.9, 0.9, 0.9, 1.0),
    TANK      : (0.70, 0.95, 0.96, 0.6),
    BOWL      : (0.78, 0.56, 0.12, 1.0),
    LADDLE    : (1.0, 1.0, 1.0, 1.0),
    POURER    : (1.0, 1.0, 1.0, 1.0),
    DISPENSER : (1.0, 1.0, 1.0, 1.0),
    CONE      : (0.645, 0.474, 0.303, 1.0),
    ROBOT     : (1.0, 1.0, 1.0, 1.0),
    BOTTLE    : (0.70, 0.95, 0.96, 0.3),  # 玻璃杯：透明效果（alpha=0.3，值越小越透明，0.0完全透明，1.0完全不透明）
    PILLAR    : (1.0, 1.0, 1.0, 1.0),
    STIRRER   : (1.0, 1.0, 1.0, 1.0),
    PLATE     : (1.0, 1.0, 1.0, 1.0),
    TABLE     : (0.55, 0.35, 0.20, 1.0),  # 木色（棕色）

    FRAME    : (1.0, 0.2, 0.2, 1.0),
    TARGET   : (0.2, 0.9, 0.2, 0.4),
    EFFECTOR : (1.0, 0.0, 0.0, 1.0),
}


############ properties #############
FRICTION = {
    CUP     : 0.5,
    TANK    : 0.5,
    BOWL    : 0.0,
    LADDLE  : 0.1,
    CONE    : 8.0,
    BOTTLE  : 0.4,
    PILLAR  : 0.0,
    STIRRER : 8.0,
    PLATE   : 0.1,
    TABLE   : 2.0,  # 桌子，高摩擦系数（防止材料滑动）
}

# MU: 向后兼容，仅当CONSISTENCY未定义时使用
# 所有材料应优先使用CONSISTENCY，MU仅作为fallback
MU = {
    WATER          : 0.0,
    INVISCID_DEMO  : 0.0,
    INVISCID_DEMO2 : 0.0,
    INVISCID_DEMO3 : 0.0,
    VISCOUS_DEMO   : 800.0,
    MILK           : 0.0,
    COFFEE         : 0.0,
    MILK_VIS       : 200.0,
    COFFEE_VIS     : 200.0,
    THICK_CREAM    : 350.0,
    WATER_NEWTON   : 1.0,
    HONEY          : 5000.0,
    KETCHUP        : 200.0,
    TOOTHPASTE     : 300.0,
    BLOOD          : 50.0,
    PAINT          : 150.0,
    CORNSTARCH     : 100.0,
    SILLY_PUTTY    : 2000.0,
    YOGURT         : 400.0,
    ELASTIC        : 416.67,
    ELASTIC_DEMO   : 10.0,
    PLASTIC_DEMO   : 160.0,
    ICECREAM       : 416.67,
    ICECREAM1      : 216.67,
    RIGID          : 416.67,
    RIGID_HEAVY    : 416.67,
    RIGID_LIGHT    : 416.67,
}

LAMDA = {
    WATER          : 277.78,
    INVISCID_DEMO  : 277.78,
    INVISCID_DEMO2 : 277.78,
    INVISCID_DEMO3 : 277.78,
    VISCOUS_DEMO   : 277.78,
    MILK           : 277.78,
    COFFEE         : 277.78,
    MILK_VIS       : 277.78,
    COFFEE_VIS     : 277.78,
    THICK_CREAM    : 277.78,
    WATER_NEWTON   : 277.78,
    HONEY          : 277.78,
    KETCHUP        : 277.78,
    TOOTHPASTE     : 277.78,
    BLOOD          : 277.78,
    PAINT          : 277.78,
    CORNSTARCH     : 277.78,
    SILLY_PUTTY    : 277.78,
    YOGURT         : 277.78,
    ELASTIC        : 277.78,
    ELASTIC_DEMO   : 100.0,
    PLASTIC_DEMO   : 277.78,
    ICECREAM       : 277.78,
    ICECREAM1       : 277.78,
    RIGID          : 277.78,
    RIGID_HEAVY    : 277.78,
    RIGID_LIGHT    : 277.78,
}

RHO = {
    WATER          : 1.0,
    INVISCID_DEMO  : 5.0,
    INVISCID_DEMO2 : 1.0,
    INVISCID_DEMO3 : 3.0,
    VISCOUS_DEMO   : 5.0,
    MILK           : 0.5,
    COFFEE         : 1.0,
    MILK_VIS       : 1.0,
    COFFEE_VIS     : 1.0,
    THICK_CREAM    : 0.8,  # 密度：略轻于水，模拟奶泡
    WATER_NEWTON   : 1.0,  # 密度：水
    HONEY          : 1.4,  # 密度：蜂蜜（比水重）
    KETCHUP        : 1.1,  # 密度：番茄酱
    TOOTHPASTE     : 1.2,  # 密度：牙膏
    BLOOD          : 1.05, # 密度：血液
    PAINT          : 1.3,  # 密度：油漆
    CORNSTARCH     : 1.1,  # 密度：玉米淀粉浆
    SILLY_PUTTY    : 1.2,  # 密度：橡皮泥
    YOGURT         : 1.05, # 密度：酸奶
    ELASTIC        : 1.0,
    ELASTIC_DEMO   : 1.0,
    PLASTIC_DEMO   : 1.0,
    ICECREAM       : 0.5,
    ICECREAM1       : 0.5,
    RIGID          : 1.0,
    RIGID_HEAVY    : 10.0,
    RIGID_LIGHT    : 0.5,
}

############ Herschel-Bulkley parameters #############
# Note: For H-B materials, CONSISTENCY (K) is the PRIMARY viscosity parameter.
# MU is derived from CONSISTENCY for backward compatibility and mu_max calculation.

# Yield stress (τ_y): 屈服应力，当剪切应力小于此值时材料表现为固体
YIELD_STRESS = {
    WATER          : 0.0,
    INVISCID_DEMO  : 0.0,
    INVISCID_DEMO2 : 0.0,
    INVISCID_DEMO3 : 0.0,
    VISCOUS_DEMO   : 0.0,
    MILK           : 0.0,
    COFFEE         : 0.0,
    MILK_VIS       : 0.0,
    COFFEE_VIS     : 0.0,
    THICK_CREAM    : 15.0,  # 屈服应力：材料需要一定剪切应力才能流动（模拟粘稠奶泡）
    WATER_NEWTON   : 0.0,   # 屈服应力：牛顿流体，无屈服应力
    HONEY          : 0.0,   # 屈服应力：蜂蜜，无屈服应力（纯剪切变稀）
    KETCHUP        : 20.0,  # 屈服应力：番茄酱，有屈服应力（需要挤压才能流动）
    TOOTHPASTE     : 100.0,  # 屈服应力：牙膏，有屈服应力
    BLOOD          : 5.0,   # 屈服应力：血液，轻微屈服应力
    PAINT          : 10.0,  # 屈服应力：油漆，有屈服应力
    CORNSTARCH     : 0.0,   # 屈服应力：玉米淀粉，无屈服应力（纯剪切增稠）
    SILLY_PUTTY    : 0.0,   # 屈服应力：橡皮泥，无屈服应力（纯剪切增稠）
    YOGURT         : 12.0,  # 屈服应力：酸奶，有屈服应力
    ELASTIC        : 0.0,
    ELASTIC_DEMO   : 0.0,
    PLASTIC_DEMO   : 0.0,
    ICECREAM       : 0.0,
    ICECREAM1       : 0.0,
    RIGID          : 0.0,
    RIGID_HEAVY    : 0.0,
    RIGID_LIGHT    : 0.0,
}

# Consistency coefficient (K): 稠度系数，控制材料的基础粘度
# PRIMARY viscosity parameter for ALL materials (both liquid and non-liquid).
# For Newtonian fluids (n=1, τ_y=0), K equals dynamic viscosity.
# For non-Newtonian fluids, K is the consistency coefficient in Herschel-Bulkley model.
# For non-liquid materials (elastic, rigid, etc.), K represents the viscosity-like parameter.
CONSISTENCY = {
    # MAT_LIQUID materials
    WATER          : 0.0,      # 牛顿流体，无粘度（理想流体）
    INVISCID_DEMO  : 0.0,      # 无粘度演示
    INVISCID_DEMO2 : 0.0,      # 无粘度演示2
    INVISCID_DEMO3 : 0.0,      # 无粘度演示3
    VISCOUS_DEMO   : 800.0,    # 高粘度演示（牛顿流体）
    MILK           : 0.0,      # 牛奶，低粘度（牛顿流体）
    COFFEE         : 0.0,      # 咖啡，低粘度（牛顿流体）
    MILK_VIS       : 200.0,    # 粘性牛奶（牛顿流体）
    COFFEE_VIS     : 200.0,    # 粘性咖啡（牛顿流体）
    THICK_CREAM    : 350.0,    # 稠奶油（非牛顿，剪切变稀）
    WATER_NEWTON   : 1.0,      # 牛顿流体，低粘度
    HONEY          : 2000.0,   # 蜂蜜（非牛顿，剪切变稀，高粘度）
    KETCHUP        : 200.0,    # 番茄酱（非牛顿，剪切变稀+屈服应力）
    TOOTHPASTE     : 1000.0,    # 牙膏（非牛顿，剪切变稀+屈服应力）
    BLOOD          : 50.0,     # 血液（非牛顿，剪切变稀）
    PAINT          : 150.0,    # 油漆（非牛顿，剪切变稀）
    CORNSTARCH     : 100.0,    # 玉米淀粉浆（非牛顿，剪切增稠）
    SILLY_PUTTY    : 1000.0,   # 橡皮泥（非牛顿，剪切增稠）
    YOGURT         : 400.0,    # 酸奶（非牛顿，剪切变稀+屈服应力）
    
    # 非液体材料（统一使用CONSISTENCY）
    ELASTIC        : 416.67,
    ELASTIC_DEMO   : 10.0,
    PLASTIC_DEMO   : 160.0,
    ICECREAM       : 416.67,
    ICECREAM1      : 216.67,
    RIGID          : 416.67,
    RIGID_HEAVY    : 416.67,
    RIGID_LIGHT    : 416.67,
}

# Flow index (n): 流动指数
# n = 1: 牛顿流体
# n < 1: 剪切变稀 (pseudoplastic)
# n > 1: 剪切增稠 (dilatant)
FLOW_INDEX = {
    WATER          : 1.0,      # 牛顿流体
    INVISCID_DEMO  : 1.0,
    INVISCID_DEMO2 : 1.0,
    INVISCID_DEMO3 : 1.0,
    VISCOUS_DEMO   : 1.0,      # 牛顿流体（高粘度）
    MILK           : 1.0,
    COFFEE         : 1.0,
    MILK_VIS       : 1.0,
    COFFEE_VIS     : 1.0,
    THICK_CREAM    : 0.65,    # 流动指数 < 1：剪切变稀，搅拌时变稀（模拟奶泡特性）
    WATER_NEWTON   : 1.0,    # 流动指数 = 1：牛顿流体（恒定粘度）
    HONEY          : 0.5,    # 流动指数 < 1：剪切变稀（n=0.5，强剪切变稀）
    KETCHUP        : 0.3,    # 流动指数 < 1：剪切变稀（n=0.3，强剪切变稀）
    TOOTHPASTE     : 0.4,    # 流动指数 < 1：剪切变稀（n=0.4）
    BLOOD          : 0.7,    # 流动指数 < 1：剪切变稀（n=0.7，轻微剪切变稀）
    PAINT          : 0.6,    # 流动指数 < 1：剪切变稀（n=0.6）
    CORNSTARCH     : 1.5,    # 流动指数 > 1：剪切增稠（n=1.5，剪切增稠）
    SILLY_PUTTY    : 1.3,    # 流动指数 > 1：剪切增稠（n=1.3，剪切增稠）
    YOGURT         : 0.6,    # 流动指数 < 1：剪切变稀（n=0.6）
    ELASTIC        : 1.0,
    ELASTIC_DEMO   : 1.0,
    PLASTIC_DEMO   : 1.0,
    ICECREAM       : 1.0,
    ICECREAM1       : 1.0,
    RIGID          : 1.0,
    RIGID_HEAVY    : 1.0,
    RIGID_LIGHT    : 1.0,
}

############ dtype #############
import numpy as np
import torch
import taichi as ti
dprecision = 32
# dprecision = 64
DTYPE_TI = eval(f'ti.f{dprecision}')
DTYPE_NP = eval(f'np.float{dprecision}')
DTYPE_TC = eval(f'torch.float{dprecision}')

EPS = 1e-12

############ misc #############
NOWHERE = [-100.0, -100.0, -100.0]


