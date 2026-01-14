# GL渲染模式透明度修复说明

## 问题分析

在GL渲染模式下，mesh（如杯子）的透明度不起作用，原因如下：

1. **Fragment Shader未使用Alpha通道**：
   - 位置：`fluidlab/fluidengine/renderers/gl_renderer_src/FlexRenderer/bindings/opengl/shadersGL.cpp` 第878行
   - 问题：`gl_FragColor = vec4(pow(fog, vec3(1.0 / 2.2)), 1.0);` 硬编码alpha为1.0
   - 应该：使用 `gl_TexCoord[4].w`（alpha值）

2. **GL_BLEND被禁用**：
   - 位置：第335行 `glDisable(GL_BLEND);`
   - 问题：即使有alpha值，也无法进行透明混合

3. **颜色数组已包含Alpha**：
   - `renderer_add_mesh` 函数（bindings.cpp第246行）已经正确读取了RGBA 4个分量
   - 数据是完整的，只是渲染时没有使用

## 修复方案

### 方案1：修改Fragment Shader（推荐）

修改 `shadersGL.cpp` 的 fragment shader：

**第845行**，获取alpha值：
```cpp
vec3 color = gl_TexCoord[4].xyz;
float alpha = gl_TexCoord[4].w;  // 添加这行
```

**第878行**，使用alpha值：
```cpp
// 原代码：
gl_FragColor = vec4(pow(fog, vec3(1.0 / 2.2)), 1.0);

// 修改为：
gl_FragColor = vec4(pow(fog, vec3(1.0 / 2.2)), alpha);
```

### 方案2：启用GL_BLEND

在渲染mesh之前启用混合：

**在 `DrawGpuMesh` 或 `DrawMesh` 函数中**，添加：
```cpp
glEnable(GL_BLEND);
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
```

**渲染完成后**，根据情况决定是否禁用：
```cpp
// 如果后续不需要透明，可以禁用
glDisable(GL_BLEND);
```

### 方案3：完整的修复代码

在 `shadersGL.cpp` 中：

1. **Fragment Shader修改**（约第845-878行）：
```cpp
void main()
{
    // ... 现有代码 ...
    
    vec3 n = gl_TexCoord[0].xyz;
    vec4 colorWithAlpha = gl_TexCoord[4];  // 改为vec4，包含alpha
    vec3 color = colorWithAlpha.xyz;
    float alpha = colorWithAlpha.w;  // 获取alpha值
    
    // ... 现有光照计算 ...
    
    vec3 fog = mix(vec3(fogColor), diffuse + ambient, exp(gl_TexCoord[7].z * fogColor.w));
    
    gl_FragColor = vec4(pow(fog, vec3(1.0 / 2.2)), alpha);  // 使用alpha
}
```

2. **启用混合**（在渲染mesh的函数中）：
```cpp
void DrawGpuMesh(GpuMesh *m, const Matrix44 &xform, const Vec3 &color)
{
    // 检查mesh是否有透明顶点
    bool hasTransparency = false;
    // 可以通过检查mesh的colors来判断
    
    if (hasTransparency) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(GL_FALSE);  // 禁用深度写入，允许透明物体正确渲染
    }
    
    // ... 现有渲染代码 ...
    
    if (hasTransparency) {
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
    }
}
```

## 编译说明

修改后需要重新编译GL渲染器：

```bash
cd fluidlab/fluidengine/renderers/gl_renderer_src
# 根据你的构建系统编译
# 通常需要重新运行setup.py或CMake
```

## 临时解决方案

如果不想修改C++代码，可以考虑：

1. **使用GGUI渲染器**：GGUI已经支持透明度
2. **调整材质颜色**：虽然GL模式下不透明，但可以通过调整RGB值模拟透明效果（不推荐）

## 注意事项

1. **渲染顺序**：透明物体需要从后往前渲染，可能需要调整渲染顺序
2. **性能影响**：启用混合会增加渲染负担
3. **深度测试**：透明物体可能需要禁用深度写入（`glDepthMask(GL_FALSE)`）

## 验证

修改后，在GL模式下运行：
```python
env = TableMaterialsEnv(version=0, renderer_type='GL')
# 杯子应该显示为半透明
```

如果杯子仍然不透明，检查：
1. 是否重新编译了GL渲染器
2. `macros.py` 中 `BOTTLE` 的alpha值是否 < 1.0
3. 渲染顺序是否正确
