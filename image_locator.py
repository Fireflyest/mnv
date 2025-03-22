import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import time
import os

from mobilenetv3_u import MultiHeadMobileNetV3

class ImageLocator:
    """
    通过特征提取和匹配，在大图中定位小图位置
    """
    def __init__(self, model=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        # 模型加载
        if model is None:
            self.model = MultiHeadMobileNetV3(num_classes=5).to(device)
        else:
            self.model = model.to(device)
        
        self.model.eval()
    
    def extract_features(self, img):
        """从图像中提取特征向量"""
        with torch.no_grad():
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            _, _, features = self.model(img_tensor)
        return features
    
    def locate_by_regions(self, small_features, large_img, suspicious_points, region_size=None, 
                         top_k=5, show_visualization=False, save_path=None, window_size=(224, 224), 
                         stride=None, similarity_metric="cosine", early_stop_threshold=0.95,
                         windows_per_point=5):
        """
        根据可疑点位置在大图中定位小图，轮流在多个可疑点周围搜索
        
        参数:
        - small_features: 小图的特征向量
        - large_img: PIL Image，大图
        - suspicious_points: 可疑点列表，格式为[(x1, y1), (x2, y2), ...]
        - region_size: 每个可疑点周围搜索区域的大小 (width, height)，默认为window_size的3倍
        - top_k: 返回的最匹配位置数量
        - show_visualization: 是否显示可视化结果
        - save_path: 保存可视化结果的路径
        - window_size: 滑动窗口大小
        - stride: 滑动步长
        - similarity_metric: 相似度度量，可选 "cosine"(余弦相似度) 或 "l2"(欧氏距离)
        - early_stop_threshold: 早停阈值，如果找到相似度大于此值的匹配则立即停止搜索
        - windows_per_point: 每轮每个可疑点处理的窗口数量
        
        返回:
        - top_positions: 最匹配的位置列表
        - top_scores: 相应的相似度得分
        """
        if region_size is None:
            region_size = (window_size[0] * 3, window_size[1] * 3)
            
        if stride is None:
            stride = (window_size[0] // 4, window_size[1] // 4)
            
        print(f"开始在{len(suspicious_points)}个可疑点附近搜索... 窗口大小: {window_size}, 区域大小: {region_size}, 步长: {stride}")
        print(f"采用轮流搜索策略，相似度阈值: {early_stop_threshold}, 每轮每点处理窗口数: {windows_per_point}")
        
        # 将小图特征向量归一化（如果使用余弦相似度）
        if similarity_metric == "cosine":
            small_features_norm = F.normalize(small_features, p=2, dim=1)
        
        # 存储所有找到的匹配结果
        best_position = None
        best_score = -float('inf') if similarity_metric == "cosine" else float('inf')
        all_positions = []
        all_scores = []
        
        # 记录搜索区域（用于可视化）
        search_regions = []
        
        # 为每个可疑点准备搜索状态
        point_states = []
        for idx, (cx, cy) in enumerate(suspicious_points):
            # 计算区域边界，确保不超出图像边界
            img_width, img_height = large_img.size
            x1 = max(0, cx - region_size[0] // 2)
            y1 = max(0, cy - region_size[1] // 2)
            x2 = min(img_width, x1 + region_size[0])
            y2 = min(img_height, y1 + region_size[1])
            
            # 可能需要调整x1, y1以保持区域大小
            x1 = max(0, x2 - region_size[0])
            y1 = max(0, y2 - region_size[1])
            
            search_regions.append((x1, y1, x2, y2))
            
            # 计算中心点
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # 创建螺旋状搜索顺序
            width_steps = (x2 - x1 - window_size[0]) // stride[0] + 1
            height_steps = (y2 - y1 - window_size[1]) // stride[1] + 1
            
            # 计算中心点对应的网格坐标
            center_grid_x = (center_x - x1) // stride[0]
            center_grid_y = (center_y - y1) // stride[1]
            
            # 创建螺旋遍历序列
            spiral_coords = self._generate_spiral_coords(center_grid_x, center_grid_y, width_steps, height_steps)
            
            # 保存该可疑点的状态
            point_states.append({
                'coords': spiral_coords,
                'current_index': 0,  # 当前处理到的索引位置
                'region': (x1, y1, x2, y2),
                'center': (cx, cy),
                'finished': False,   # 是否已处理完所有窗口
                'windows_processed': 0  # 已处理窗口数
            })
            
            print(f"可疑点 #{idx+1}: ({cx, cy}), 搜索区域: ({x1, y1}) -> ({x2, y2}), 共{len(spiral_coords)}个潜在窗口")
            
        extraction_start = time.time()
        found_good_match = False
        active_points = list(range(len(suspicious_points)))  # 活跃点索引列表
        
        # 轮流处理每个可疑点，直到找到好的匹配或所有点处理完毕
        while active_points and not found_good_match:
            points_to_remove = []
            
            # 轮流处理每个活跃可疑点
            for point_idx in active_points:
                state = point_states[point_idx]
                cx, cy = state['center']
                x1, y1, x2, y2 = state['region']
                
                windows_this_round = 0
                print(f"处理可疑点 #{point_idx+1}: ({cx, cy}), 当前进度: {state['windows_processed']}/{len(state['coords'])}窗口")
                
                # 处理指定数量的窗口后切换到下一个点
                while windows_this_round < windows_per_point and not state['finished']:
                    if state['current_index'] >= len(state['coords']):
                        state['finished'] = True
                        points_to_remove.append(point_idx)
                        print(f"可疑点 #{point_idx+1} 处理完毕！")
                        break
                    
                    grid_x, grid_y = state['coords'][state['current_index']]
                    state['current_index'] += 1
                    
                    # 转换回实际坐标
                    x = x1 + grid_x * stride[0]
                    y = y1 + grid_y * stride[1]
                    
                    # 确保不超出区域边界
                    if x < x1 or y < y1 or x + window_size[0] > x2 or y + window_size[1] > y2:
                        continue
                    
                    # 裁剪窗口
                    window = large_img.crop((x, y, x + window_size[0], y + window_size[1]))
                    
                    # 提取特征
                    features = self.extract_features(window).cpu()
                    state['windows_processed'] += 1
                    windows_this_round += 1
                    
                    # 计算相似度并检查是否满足早停条件
                    if similarity_metric == "cosine":
                        # 计算余弦相似度
                        features_norm = F.normalize(features, p=2, dim=1)
                        similarity = torch.mm(small_features_norm, features_norm.t()).item()
                        
                        # 更新最佳匹配
                        if similarity > best_score:
                            best_score = similarity
                            best_position = (x, y, window_size[0], window_size[1])
                        
                        # 记录结果
                        all_positions.append((x, y, window_size[0], window_size[1]))
                        all_scores.append(similarity)
                        
                        # 检查是否达到早停阈值
                        if similarity >= early_stop_threshold:
                            print(f"在可疑点 #{point_idx+1} 附近找到高度匹配 (相似度: {similarity:.4f} >= {early_stop_threshold})，停止搜索")
                            found_good_match = True
                            break
                    else:  # 使用L2距离
                        # 计算L2距离
                        distance = torch.cdist(small_features, features, p=2).item()
                        
                        # L2距离越小越好
                        if distance < best_score:
                            best_score = distance
                            best_position = (x, y, window_size[0], window_size[1])
                        
                        # 记录结果
                        all_positions.append((x, y, window_size[0], window_size[1]))
                        all_scores.append(-distance)  # 转换为负值以便后续排序
                        
                        # 对于L2距离，较小值表示更好的匹配
                        if distance <= 1 - early_stop_threshold:
                            print(f"在可疑点 #{point_idx+1} 附近找到高度匹配 (距离: {distance:.4f} <= {1-early_stop_threshold})，停止搜索")
                            found_good_match = True
                            break
                
                # 打印当前可疑点进度
                print(f"可疑点 #{point_idx+1} 本轮处理了 {windows_this_round} 个窗口，总计: {state['windows_processed']}")
                
                # 如果找到了好的匹配，停止所有处理
                if found_good_match:
                    break
                
                # 如果该点处理完毕，将其标记为移除
                if state['finished'] and point_idx not in points_to_remove:
                    points_to_remove.append(point_idx)
            
            # 从活跃列表中移除已处理完毕的点
            for idx in points_to_remove:
                if idx in active_points:
                    active_points.remove(idx)
        
        extraction_time = time.time() - extraction_start
        windows_total = sum(state['windows_processed'] for state in point_states)
        print(f"特征提取总用时: {extraction_time:.2f}秒, 共处理 {windows_total} 个窗口")
        
        # 如果没有提取到任何特征，返回空结果
        if len(all_positions) == 0:
            print("警告: 未提取到任何特征，请检查可疑点位置和区域大小")
            return [], []
        
        # 确定返回结果
        if best_position is not None and similarity_metric == "cosine" and best_score >= early_stop_threshold and top_k == 1:
            top_positions = [best_position]
            top_scores = [best_score]
        else:
            # 对所有结果排序
            sorted_indices = sorted(range(len(all_scores)), key=lambda i: all_scores[i], reverse=True)
            
            # 获取前k个最佳匹配
            top_indices = sorted_indices[:min(top_k, len(all_positions))]
            top_positions = [all_positions[i] for i in top_indices]
            top_scores = [all_scores[i] for i in top_indices]
        
        # 可视化结果
        if show_visualization or save_path:
            # 创建一个副本用于标记可疑点
            marked_img = large_img.copy()
            draw = ImageDraw.Draw(marked_img)
            
            # 标记可疑点和搜索区域
            for (cx, cy), (x1, y1, x2, y2) in zip(suspicious_points, search_regions):
                # 绘制可疑点
                point_radius = 5
                draw.ellipse((cx-point_radius, cy-point_radius, cx+point_radius, cy+point_radius), 
                            fill=(0, 255, 255))  # 青色
                
                # 绘制搜索区域
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 255), width=2)
            
            # 可视化匹配结果
            self._visualize_results(marked_img, top_positions, top_scores, similarity_metric, 
                                  show_visualization, save_path, suspicious_points)
        
        return top_positions, top_scores
    
    def locate_by_regions_multi_scale(self, small_features, large_img, suspicious_points, 
                                     region_size=None, top_k=5, show_visualization=False, 
                                     save_path=None, window_sizes=None, stride_ratio=0.25, 
                                     similarity_metric="cosine", early_stop_threshold=0.95,
                                     windows_per_point=5):
        """
        在不知道小图尺寸的情况下，使用多尺度窗口搜索可疑点附近区域
        
        参数:
        - small_features: 小图的特征向量
        - large_img: PIL Image，大图
        - suspicious_points: 可疑点列表，格式为[(x1, y1), (x2, y2), ...]
        - region_size: 每个可疑点周围搜索区域的大小 (width, height)
        - top_k: 返回的最匹配位置数量
        - show_visualization: 是否显示可视化结果
        - save_path: 保存可视化结果的路径
        - window_sizes: 尝试的窗口尺寸列表，格式为[(w1,h1), (w2,h2), ...]，默认尝试多个预设尺寸
        - stride_ratio: 步长与窗口尺寸的比例
        - similarity_metric: 相似度度量，可选 "cosine"(余弦相似度) 或 "l2"(欧氏距离)
        - early_stop_threshold: 早停阈值，如果找到相似度大于此值的匹配则立即停止搜索
        - windows_per_point: 每轮每个可疑点处理的窗口数量
        
        返回:
        - top_positions: 最匹配的位置列表
        - top_scores: 相应的相似度得分
        """
        # 默认尝试多种窗口尺寸
        if window_sizes is None:
            window_sizes = [
                (100, 100), (150, 150), (200, 200), (250, 250), (300, 300),
                (100, 150), (150, 100), (200, 150), (150, 200), (250, 200), (200, 250)
            ]
        
        print(f"开始多尺度搜索... 将尝试 {len(window_sizes)} 种不同窗口大小")
        
        # 存储所有尺度上的匹配结果
        all_positions = []
        all_scores = []
        best_window_size = None
        best_score = -float('inf') if similarity_metric == "cosine" else float('inf')
        
        # 在每个窗口尺寸上搜索
        for win_idx, window_size in enumerate(window_sizes):
            print(f"\n[{win_idx+1}/{len(window_sizes)}] 使用窗口尺寸: {window_size}")
            
            # 对当前窗口尺寸计算步长
            stride = (int(window_size[0] * stride_ratio), int(window_size[1] * stride_ratio))
            
            # 计算当前窗口尺寸的搜索区域大小
            curr_region_size = region_size
            if region_size is None:
                # 默认区域大小为窗口大小的3倍
                curr_region_size = (window_size[0] * 3, window_size[1] * 3)
            
            # 在当前尺度上搜索
            curr_positions, curr_scores = self.locate_by_regions(
                small_features=small_features,
                large_img=large_img,
                suspicious_points=suspicious_points,
                region_size=curr_region_size,
                top_k=top_k,
                show_visualization=False,  # 仅在最后显示
                save_path=None,  # 仅在最后保存
                window_size=window_size,
                stride=stride,
                similarity_metric=similarity_metric,
                early_stop_threshold=early_stop_threshold,
                windows_per_point=windows_per_point
            )
            
            # 如果找到匹配
            if curr_positions:
                # 添加到总结果中
                all_positions.extend(curr_positions)
                all_scores.extend(curr_scores)
                
                # 检查是否找到更好的匹配
                if similarity_metric == "cosine":
                    if curr_scores[0] > best_score:
                        best_score = curr_scores[0]
                        best_window_size = window_size
                        
                        # 如果找到高度匹配，可以提前结束搜索
                        if curr_scores[0] >= early_stop_threshold:
                            print(f"在窗口尺寸 {window_size} 找到高度匹配 (相似度: {curr_scores[0]:.4f})，提前结束搜索")
                            break
                else:  # L2距离
                    if curr_scores[0] > best_score:  # 已转为负值，越大越好
                        best_score = curr_scores[0]
                        best_window_size = window_size
                        
                        # 如果找到高度匹配，可以提前结束搜索
                        if -curr_scores[0] <= 1 - early_stop_threshold:  # 转回原始L2距离
                            print(f"在窗口尺寸 {window_size} 找到高度匹配 (距离: {-curr_scores[0]:.4f})，提前结束搜索")
                            break
        
        # 如果没有找到任何匹配
        if not all_positions:
            print("未找到任何匹配")
            return [], []
        
        # 对所有结果按相似度排序
        sorted_indices = sorted(range(len(all_scores)), key=lambda i: all_scores[i], reverse=True)
        
        # 获取前k个最佳匹配
        top_indices = sorted_indices[:min(top_k, len(all_positions))]
        top_positions = [all_positions[i] for i in top_indices]
        top_scores = [all_scores[i] for i in top_indices]
        
        print(f"\n搜索完成! 最佳窗口尺寸: {best_window_size}, 最佳匹配得分: {best_score:.4f}")
        
        # 可视化结果
        if show_visualization or save_path:
            # 创建一个副本用于标记
            marked_img = large_img.copy()
            draw = ImageDraw.Draw(marked_img)
            
            # 标记所有可疑点
            for cx, cy in suspicious_points:
                point_radius = 5
                draw.ellipse((cx-point_radius, cy-point_radius, cx+point_radius, cy+point_radius), 
                            fill=(0, 255, 255))  # 青色
            
            # 可视化匹配结果
            self._visualize_results(marked_img, top_positions, top_scores, similarity_metric, 
                                   show_visualization, save_path, suspicious_points)
        
        return top_positions, top_scores
    
    def _generate_spiral_coords(self, center_x, center_y, width, height):
        """生成从中心点向外扩散的螺旋坐标序列"""
        # 从中心开始的螺旋坐标
        result = [(center_x, center_y)]
        
        # 生成螺旋坐标
        layer = 1
        while layer <= max(width, height):
            # 向右
            for i in range(layer):
                if center_x + i < width and center_y - layer//2 >= 0:
                    result.append((center_x + i, center_y - layer//2))
            
            # 向下
            for i in range(layer):
                if center_x + layer//2 < width and center_y + i < height:
                    result.append((center_x + layer//2, center_y + i))
            
            # 向左
            for i in range(layer):
                if center_x - i >= 0 and center_y + layer//2 < height:
                    result.append((center_x - i, center_y + layer//2))
            
            # 向上
            for i in range(layer):
                if center_x - layer//2 >= 0 and center_y - i >= 0:
                    result.append((center_x - layer//2, center_y - i))
            
            layer += 1
        
        return result
    
    def _visualize_results(self, large_img, positions, scores, metric, show=True, save_path=None, suspicious_points=None):
        """可视化定位结果"""
        # 创建副本以绘制结果
        img_with_boxes = large_img.copy()
        draw = ImageDraw.Draw(img_with_boxes)
        
        # 为不同得分设置不同颜色
        colors = [
            (255, 0, 0),    # 红色 - 最匹配
            (255, 165, 0),  # 橙色
            (255, 255, 0),  # 黄色
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 蓝色
        ]
        
        # 绘制结果
        plt.figure(figsize=(12, 10))
        plt.subplot(1, 1, 1)
        plt.imshow(img_with_boxes)
        plt.title("locator")
        plt.axis('off')
        
        # 如果有可疑点，在可视化结果中也标记出来
        if suspicious_points:
            for cx, cy in suspicious_points:
                circle = plt.Circle((cx, cy), 5, color='cyan', fill=True, alpha=0.7)
                plt.gca().add_patch(circle)
                plt.text(cx+10, cy+10, "suspicious_point", color='cyan', fontsize=10, 
                        bbox=dict(facecolor='white', alpha=0.7))
        
        # 在图像上标出每个位置
        for i, ((x, y, w, h), score) in enumerate(zip(positions, scores)):
            color = colors[min(i, len(colors)-1)]
            
            # 绘制矩形框
            draw.rectangle([x, y, x+w, y+h], outline=color, width=3)
            
            # 添加得分文本
            if metric == "cosine":
                score_text = f"{score:.2f}"
            else:
                score_text = f"{-score:.2f}"
            
            # 在图像上显示排名
            draw.text((x+5, y+5), f"#{i+1}: {score_text}", fill=color)
            
            # 在可视化图中添加标注
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor=f"C{i}", linewidth=2)
            plt.gca().add_patch(rect)
            plt.text(x+5, y+5, f"#{i+1}: {score_text}", color=f"C{i}", fontsize=12, 
                    bbox=dict(facecolor='white', alpha=0.7))
        
        # 保存或显示结果
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            img_with_boxes.save(save_path.replace('.png', '_raw.png'))
            print(f"结果已保存至: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()

# 使用示例
if __name__ == "__main__":
    # 创建定位器
    locator = ImageLocator()
    
    # 加载大图和小图
    large_img_path = "./temp/big_image.png"  # 替换为实际大图路径
    small_img_path = "./temp/small_image.png"  # 替换为实际小图路径
    
    large_img = Image.open(large_img_path).convert('RGB')
    
    small_img = Image.open(small_img_path).convert('RGB')
    # 提取小图特征
    small_features = locator.extract_features(small_img)

    suspicious_points = [(600, 750), (1100, 750), (800, 300)]  # 示例坐标，替换为实际可疑点位置
    
    # 使用多尺度搜索（不依赖小图的尺寸）
    top_positions, top_scores = locator.locate_by_regions_multi_scale(
        small_features,
        large_img,
        suspicious_points=suspicious_points,
        region_size=(400, 400),
        top_k=3,
        show_visualization=True,
        save_path="./out/multi_scale_localization_result.png",
        window_sizes=None,  # 使用默认的多个尺寸
        stride_ratio=0.125,  # 步长为窗口大小的1/8
        similarity_metric="cosine",
        early_stop_threshold=0.93,
        windows_per_point=2
    )
    
    print("最匹配的位置:")
    for i, ((x, y, w, h), score) in enumerate(zip(top_positions, top_scores)):
        print(f"#{i+1} - 位置: ({x}, {y}), 尺寸: {w}x{h}, 相似度得分: {score:.4f}")
