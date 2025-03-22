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

# 特征匹配 - 使用SIFT
def calculate_feature_matching(image1, image2):
    # 创建SIFT对象
    sift = cv2.SIFT_create()
    
    # 检测关键点和计算描述符
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
    
    # 检查描述符是否有效
    if descriptors1 is None or descriptors2 is None or len(descriptors1) == 0 or len(descriptors2) == 0:
        return 0.0, 0, np.zeros((10, 10, 3), dtype=np.uint8)  # 返回空图像作为替代
    
    # 使用FLANN进行快速匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # 匹配描述符
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    # 应用Lowe's ratio测试来筛选好的匹配
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:  # 确保有两个匹配
            m, n = m_n
            if m.distance < 0.75 * n.distance:  # Lowe's ratio test
                good_matches.append(m)
    
    # 计算相似度分数（好的匹配占总关键点的比例）
    similarity = len(good_matches) / max(len(keypoints1), len(keypoints2)) if max(len(keypoints1), len(keypoints2)) > 0 else 0
    
    # 绘制匹配
    match_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches[:30], None, 
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return similarity, len(good_matches), match_image

# 主程序
def main():
    # 标准大小
    STANDARD_SIZE = (224, 224)
    
    image1_path = './temp/chair1.png'
    image_paths = [
        './temp/chair1.png',
        './temp/chair2.png',
        './temp/chair_offset.png',
        './temp/chair_big.png',
        './temp/chair_small.png'
    ]
    
    # 读取原图并调整尺寸
    original_color_image1 = cv2.imread(image1_path)
    original_gray_image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    
    # 保存原始尺寸以展示
    original_size_img1 = original_color_image1.copy()
    
    # 调整为标准尺寸
    color_image1 = cv2.resize(original_color_image1, STANDARD_SIZE)
    gray_image1 = cv2.resize(original_gray_image1, STANDARD_SIZE)
    
    # 初始化列表
    combined_images = []
    match_images = []
    original_vs_resized = []
    
    # 创建原始图与调整后对比图
    comparison = np.hstack((original_size_img1, cv2.resize(color_image1, (224, original_size_img1.shape[0]))))
    cv2.putText(comparison, f'Original Size: {original_size_img1.shape[1]}x{original_size_img1.shape[0]}', 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(comparison, f'Resized: {STANDARD_SIZE[0]}x{STANDARD_SIZE[1]}', 
                (original_size_img1.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    original_vs_resized.append(comparison)
    
    print(f"所有图片将被调整至标准尺寸: {STANDARD_SIZE}")
    
    for image_path in image_paths:
        print(f"处理图片: {image_path}")
        
        # 读取对比图像
        original_color_image2 = cv2.imread(image_path)
        original_gray_image2 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # 保存原始尺寸
        original_shape = original_color_image2.shape[:2][::-1]  # 宽x高
        
        # 调整为标准尺寸
        color_image2 = cv2.resize(original_color_image2, STANDARD_SIZE)
        gray_image2 = cv2.resize(original_gray_image2, STANDARD_SIZE)
        
        # 计算相似度（所有都使用标准尺寸的图片）
        hist_similarity = calculate_histogram_similarity(color_image1, color_image2)
        ssim_score = calculate_ssim(gray_image1, gray_image2)
        feature_similarity, num_good_matches, match_img = calculate_feature_matching(gray_image1, gray_image2)
        
        # 保存匹配图像
        match_images.append(match_img)
        
        # 为可视化准备原始图像
        # 将两张图像的高度调整为一致
        height = max(original_color_image1.shape[0], original_color_image2.shape[0])
        img1_resized = cv2.resize(original_color_image1, 
                                 (int(original_color_image1.shape[1] * height / original_color_image1.shape[0]), height))
        img2_resized = cv2.resize(original_color_image2, 
                                 (int(original_color_image2.shape[1] * height / original_color_image2.shape[0]), height))
        
        # 并排显示原始图像
        combined_image = np.hstack((img1_resized, img2_resized))
        
        # 在图片上显示相似度结果
        cv2.putText(combined_image, f'hist: {hist_similarity:.2f}', (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined_image, f'SSIM: {ssim_score:.2f}', (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined_image, f'SIFT: {feature_similarity:.2f}', (10, 180), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined_image, f'point: {num_good_matches}', (10, 210), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        combined_images.append(combined_image)
        
        # 创建原始与调整后的对比
        if image_path != image1_path:  # 跳过第一张图，因为已经添加过了
            comparison = np.hstack((original_color_image2, cv2.resize(color_image2, (224, original_color_image2.shape[0]))))
            cv2.putText(comparison, f'原始尺寸: {original_shape[0]}x{original_shape[1]}', 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(comparison, f'调整尺寸: {STANDARD_SIZE[0]}x{STANDARD_SIZE[1]}', 
                        (original_color_image2.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            original_vs_resized.append(comparison)
    
    # 保存结果
    os.makedirs('./out', exist_ok=True)
    
    # 将相同宽度的图像垂直堆叠
    def stack_images(images):
        # 找出最大宽度
        max_width = max(img.shape[1] for img in images)
        # 调整所有图像到相同宽度
        resized_images = []
        for img in images:
            aspect_ratio = img.shape[0] / img.shape[1]
            new_height = int(aspect_ratio * max_width)
            resized_images.append(cv2.resize(img, (max_width, new_height)))
        # 垂直堆叠
        return np.vstack(resized_images)
    
    # 堆叠并保存结果
    final_image = stack_images(combined_images)
    match_summary = stack_images(match_images)
    resized_comparison = stack_images(original_vs_resized)
    
    cv2.imwrite('./out/match_standardized.jpg', final_image)
    cv2.imwrite('./out/sift_match_standardized.jpg', match_summary)
    cv2.imwrite('./out/original_vs_resized.jpg', resized_comparison)
    
    print("分析完成! 结果已保存至 ./out/ 目录")
    print(f"所有图片均已调整至统一尺寸 {STANDARD_SIZE[0]}x{STANDARD_SIZE[1]} 后进行比较")

if __name__ == "__main__":
    import os
    main()