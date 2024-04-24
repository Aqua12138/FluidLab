import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# 图像文件的路径
image_path = '/home/zhx/PycharmProjects/fluids/FluidLab-test/debug/gridsensor3d/frame_040.npy'  # 确保路径和文件格式正确

# 加载图像
img = np.load(image_path)[:, :, :]
img_rotated = np.rot90(img, k=-1)  # k=-1表示向右旋转90度
# 显示图像
plt.imshow(img_rotated)
plt.title('GridSensor3D Image')
plt.axis('off')  # 不显示坐标轴
plt.show()
