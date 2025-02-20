import numpy as np
import math

class PoseAnalyzer:
    def __init__(self):
        # 评分权重
        self.weights = {
            'front_knee_angle': 0.4,  # 前腿膝盖角度权重增加
            'back_leg_straight': 0.3, # 后腿伸直程度权重降低
            'body_vertical': 0.1,      # 保持躯干垂直度权重
            'stability': 0.2           # 保持稳定性权重
        }
        self.min_frame_interval = 5  # 两个关键帧之间的最小间隔帧数
        self.last_key_frame = -self.min_frame_interval  # 上一个关键帧的索引

        # 标准参数
        self.standards = {
            'front_knee_angle': 90,    # 保持前膝90度标准
            'back_knee_angle': 160,    # 降低后腿伸直要求(由170改为150)
            'vertical_angle': 90       # 保持躯干垂直
        }
        
        # 放宽关键帧检测参数
        self.frame_window = 30         # 保持检测窗口大小
        self.key_frame_threshold = 0.6 # 降低判定阈值(由0.8改为0.6)
        self.consecutive_frames = 5     # 减少需要保持的连续帧数(由5改为3)
        self.min_frame_interval = 15  # 两个关键帧之间的最小间隔帧数
        self.last_key_frame = -self.min_frame_interval  # 上一个关键帧的索引
    
    def calculate_angle(self, point1, point2, point3):
        """计算三个点形成的角度"""
        a = np.array([point1['x'], point1['y'], point1['z']])
        b = np.array([point2['x'], point2['y'], point2['z']])
        c = np.array([point3['x'], point3['y'], point3['z']])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)
    
    def is_gong_bu_frame(self, frame_data):
        """判断是否为弓步关键帧"""
        # 获取关键点
        left_hip = frame_data.get('左髋')
        right_hip = frame_data.get('右髋')
        left_knee = frame_data.get('左膝')
        right_knee = frame_data.get('右膝')
        left_ankle = frame_data.get('左踝')
        right_ankle = frame_data.get('右踝')
        
        if not all([left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]):
            return False
        
        # 计算左右腿z轴差值，判断哪条腿是后腿并检查是否伸直
        left_leg_z_diff = abs(left_hip['z'] - left_knee['z']) + abs(left_knee['z'] - left_ankle['z'])
        right_leg_z_diff = abs(right_hip['z'] - right_knee['z']) + abs(right_knee['z'] - right_ankle['z'])

        # 检查双脚是否着地
        feet_height_diff = abs(left_ankle['y'] - right_ankle['y'])
        max_feet_height_diff = 0.05  # 允许的最大高度差

        lowest_y = max(left_ankle['y'], right_ankle['y'])
        # 判断双脚是否都接近地面
        is_both_feet_grounded = (
            feet_height_diff < max_feet_height_diff and  # 双脚高度差小
            abs(left_ankle['y'] - lowest_y) < max_feet_height_diff and  # 左脚接近地面
            abs(right_ankle['y'] - lowest_y) < max_feet_height_diff     # 右脚接近地面
        )
    
        if not is_both_feet_grounded:
            return False   
        
        # 计算左右膝盖角度
        left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
        
        # 弓步特征判断（以左腿前为例）
        is_left_forward = (
            abs(left_knee_angle - self.standards['front_knee_angle']) < 30 and  # 前腿弯曲
            abs(right_knee_angle - self.standards['back_knee_angle']) < 30 and  # 后腿伸直
            right_leg_z_diff < 0.07  # 后腿在同一平面上
        )
        
        # 右腿前的情况
        is_right_forward = (
            abs(right_knee_angle - self.standards['front_knee_angle']) < 30 and
            abs(left_knee_angle - self.standards['back_knee_angle']) < 30 and
            left_leg_z_diff < 0.07  # 后腿在同一平面
        )
        
        # 添加辅助判断：检查髋部高度差
        
        return (is_left_forward or is_right_forward)
    
    def score_gong_bu(self, frame_data):
        """对弓步动作进行打分"""
        if not self.is_gong_bu_frame(frame_data):
            return 0
            
        scores = {}
        
        # 1. 前腿膝盖角度评分
        front_knee = min(self.calculate_angle(frame_data['左髋'], frame_data['左膝'], frame_data['左踝']),
                        self.calculate_angle(frame_data['右髋'], frame_data['右膝'], frame_data['右踝']))
        scores['front_knee_angle'] = 100 - abs(front_knee - self.standards['front_knee_angle'])
        
        # 2. 后腿伸直程度评分
        back_knee = max(self.calculate_angle(frame_data['左髋'], frame_data['左膝'], frame_data['左踝']),
                       self.calculate_angle(frame_data['右髋'], frame_data['右膝'], frame_data['右踝']))
        scores['back_leg_straight'] = 100 - abs(back_knee - self.standards['back_knee_angle'])
        
        # 3. 躯干垂直度评分
        spine_angle = self.calculate_angle(
            frame_data['左肩'], 
            frame_data['左髋'], 
            {'x': frame_data['左髋']['x'], 'y': frame_data['左髋']['y'] + 1, 'z': frame_data['左髋']['z']}
        )
        scores['body_vertical'] = 100 - abs(spine_angle - self.standards['vertical_angle'])
        
        # 4. 计算稳定性得分（通过关键点的可见性和位置变化）
        key_points_visibility = [
            frame_data['左髋']['v'],
            frame_data['右髋']['v'],
            frame_data['左膝']['v'],
            frame_data['右膝']['v'],
            frame_data['左踝']['v'],
            frame_data['右踝']['v']
        ]
        scores['stability'] = sum(key_points_visibility) / len(key_points_visibility) * 100
        
        # 计算总分
        final_score = sum(score * self.weights[key] for key, score in scores.items())
        
        return final_score

    def detect_key_frames(self, frame_sequence):
        """
        检测关键帧序列
        frame_sequence: 包含连续帧数据的列表
        返回关键帧的索引列表
        """
        key_frames = []
        potential_key_frame_count = 0
        
        for i in range(len(frame_sequence)):
            # 检查是否满足最小帧间隔要求
            if i - self.last_key_frame < self.min_frame_interval:
                continue
                
            frame_data = frame_sequence[i]
            
            # 判断是否为弓步姿势
            if self.is_gong_bu_frame(frame_data):
                potential_key_frame_count += 1
                
                # 如果连续多帧都是弓步姿势
                if potential_key_frame_count >= self.consecutive_frames:
                    # 从这些连续帧中选择得分最高的作为关键帧
                    start_idx = i - self.consecutive_frames + 1
                    scores = [self.score_gong_bu(frame_sequence[j]) 
                            for j in range(start_idx, i + 1)]
                    
                    best_frame_idx = start_idx + scores.index(max(scores))
                    
                    if best_frame_idx not in key_frames:
                        key_frames.append(best_frame_idx)
                        self.last_key_frame = best_frame_idx  # 更新最后关键帧索引
                        potential_key_frame_count = 0  # 重置计数器
            else:
                potential_key_frame_count = 0
                
        return key_frames

    def analyze_sequence(self, frame_sequence):
        """
        分析整个动作序列
        frame_sequence: 包含连续帧数据的列表
        返回关键帧信息和得分
        """
        key_frames = self.detect_key_frames(frame_sequence)
        analysis_result = {
            'key_frames': key_frames,
            'scores': [],
            'details': []
        }
        
        for frame_idx in key_frames:
            frame_data = frame_sequence[frame_idx]
            score = self.score_gong_bu(frame_data)
            
            analysis_result['scores'].append({
                'frame_index': frame_idx,
                'score': score
            })
            
            # 添加详细分析信息
            analysis_result['details'].append({
                'frame_index': frame_idx,
                'front_knee_angle': min(
                    self.calculate_angle(frame_data['左髋'], frame_data['左膝'], frame_data['左踝']),
                    self.calculate_angle(frame_data['右髋'], frame_data['右膝'], frame_data['右踝'])
                ),
                'back_knee_angle': max(
                    self.calculate_angle(frame_data['左髋'], frame_data['左膝'], frame_data['左踝']),
                    self.calculate_angle(frame_data['右髋'], frame_data['右膝'], frame_data['右踝'])
                )
            })
            
        return analysis_result
