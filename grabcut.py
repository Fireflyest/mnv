import cv2
import numpy as np
import matplotlib.pyplot as plt

def grabcut_segmentation(image_path, output_path):
    # 读取图像
    image = cv2.imread(image_path)
    mask = np.zeros(image.shape[:2], np.uint8)

    # 创建背景模型和前景模型
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # 定义矩形区域，包含前景对象
    rect = (50, 50, image.shape[1] - 100, image.shape[0] - 100)

    # 应用 GrabCut 算法
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # 将所有确定为前景或可能为前景的像素设置为 1，其余设置为 0
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # 应用掩码到图像
    result = image * mask2[:, :, np.newaxis]

    # 保存结果
    cv2.imwrite(output_path, result)

    # 显示结果
    plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(122), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), plt.title('Segmented Image')
    plt.show()

image_path = './temp/ice.jpg'
output_path = './out/grabcut.jpg'
grabcut_segmentation(image_path, output_path)