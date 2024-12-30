import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_template(image_path, template_path, output_path):
    # 读取图像和模板
    image = cv2.imread(image_path)
    template = cv2.imread(template_path)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 获取模板的宽度和高度
    w, h = template_gray.shape[::-1]

    # 定义不同的模板匹配方法
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    # 创建一个图形窗口
    plt.figure(figsize=(12, 8))

    for i, method in enumerate(methods):
        img_copy = image.copy()
        method_eval = eval(method)

        # 使用模板匹配
        res = cv2.matchTemplate(image_gray, template_gray, method_eval)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # 如果使用 TM_SQDIFF 或 TM_SQDIFF_NORMED，取最小值
        if method in ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + w, top_left[1] + h)

        # 在图像上绘制矩形框
        cv2.rectangle(img_copy, top_left, bottom_right, (0, 255, 0), 2)

        # 将结果图像叠加到最终结果图像上
        result_image = cv2.addWeighted(image, 0.5, img_copy, 0.5, 0)

        # 在子图中显示结果
        plt.subplot(2, 3, i + 1)
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title(method)
        plt.axis('off')

    # 保存结果
    plt.savefig(output_path)

image_path = './temp/image.jpg'
template_path = './temp/image_template.jpg'
output_path = './out/find.jpg'
find_template(image_path, template_path, output_path)