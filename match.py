import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# 直方图比较
def calculate_histogram_similarity(image1, image2):
    hsv_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    hsv_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
    hist_image1 = cv2.calcHist([hsv_image1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist_image2 = cv2.calcHist([hsv_image2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist_image1, hist_image1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_image2, hist_image2, 0, 1, cv2.NORM_MINMAX)
    similarity = cv2.compareHist(hist_image1, hist_image2, cv2.HISTCMP_CORREL)
    return similarity

# 结构相似性（SSIM）
def calculate_ssim(image1, image2):
    score, _ = ssim(image1, image2, full=True)
    return score

# 特征匹配
def calculate_feature_matching(image1, image2):
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(image2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    similarity = len(matches) / max(len(keypoints1), len(keypoints2))
    return similarity

image1_path = './temp/image.jpg'
image_paths = [
    './temp/xie.jpg',
    './temp/zheng.jpg',
    './temp/ice.jpg',
    './temp/waterpolo.jpg'
]

# 读取原图
color_image1 = cv2.imread(image1_path)
gray_image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)

# 初始化一个空的列表来存储合并后的图像
combined_images = []

for image_path in image_paths:
    # 读取对比图像
    color_image2 = cv2.imread(image_path)
    gray_image2 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 调整图像尺寸，使它们相同
    if gray_image1.shape != gray_image2.shape:
        gray_image2 = cv2.resize(gray_image2, (gray_image1.shape[1], gray_image1.shape[0]))
        color_image2 = cv2.resize(color_image2, (color_image1.shape[1], color_image1.shape[0]))

    # 计算相似度
    hist_similarity = calculate_histogram_similarity(color_image1, color_image2)
    ssim_score = calculate_ssim(gray_image1, gray_image2)
    feature_similarity = calculate_feature_matching(gray_image1, gray_image2)

    # 将两张图片并排显示
    combined_image = np.hstack((color_image1, color_image2))

    # 在图片上显示相似度
    cv2.putText(combined_image, f'Histogram Similarity: {hist_similarity:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(combined_image, f'SSIM: {ssim_score:.2f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(combined_image, f'Feature Matching Similarity: {feature_similarity:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 将合并后的图像添加到列表中
    combined_images.append(combined_image)

# 将所有合并后的图像垂直堆叠
final_image = np.vstack(combined_images)

# 保存图片
cv2.imwrite('./out/match.jpg', final_image)