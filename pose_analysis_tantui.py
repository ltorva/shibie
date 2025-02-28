import numpy as np
import math

class PoseAnalyzer_tantui:
    def __init__(self):
        # 评分权重
        self.weights = {
            'kick_height': 0.4,        # 踢腿高度权重
            'support_leg': 0.3,        # 支撑腿稳定性
            'body_vertical': 0.2,      # 躯干垂直度
            'explosive_power': 0.1     # 爆发力
        }

        # 标准参数
        self.standards = {
            'support_leg_angle': 160,  # 支撑腿伸直程度
            'min_kick_height': 1.0,    # 踢腿脚跟高于支撑腿膝盖的最小比例
            'vertical_angle': 90,      # 躯干垂直度
            'heel_ground_threshold': 0.05  # 脚跟离地判定阈值
        }
        
        # 关键帧检测参数
        self.consecutive_frames = 3     # 连续帧数要求
        self.min_frame_interval = 15   # 最小帧间隔
        self.last_key_frame = -self.min_frame_interval

    def is_tan_tui_frame(self, frame_data):
        """判断是否为弹腿关键帧"""
        # 获取关键点
        left_hip = frame_data.get('左髋')
        right_hip = frame_data.get('右髋')
        left_knee = frame_data.get('左膝')
        right_knee = frame_data.get('右膝')
        left_ankle = frame_data.get('左踝')
        right_ankle = frame_data.get('右踝')
        
        if not all([left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle]):
            return False

        # 1. 判断支撑腿（通过y坐标判断哪只脚在地面）
        is_left_support = left_ankle['y'] > right_ankle['y']
        
        if is_left_support:
            support_leg = {
                'hip': left_hip,
                'knee': left_knee,
                'ankle': left_ankle
            }
            kick_leg = {
                'hip': right_hip,
                'knee': right_knee,
                'ankle': right_ankle
            }
        else:
            support_leg = {
                'hip': right_hip,
                'knee': right_knee,
                'ankle': right_ankle
            }
            kick_leg = {
                'hip': left_hip,
                'knee': left_knee,
                'ankle': left_ankle
            }

        # 2. 检查支撑腿是否稳定
        support_leg_angle = self.calculate_angle(
            support_leg['hip'],
            support_leg['knee'],
            support_leg['ankle']
        )
        if support_leg_angle < self.standards['support_leg_angle']:
            return False

        # 3. 检查踢腿高度（脚跟需高于支撑腿膝盖）
        if kick_leg['ankle']['y'] >= support_leg['knee']['y']:
            return False

        # 4. 检查支撑脚跟是否离地
        if self._is_heel_lifted(support_leg['ankle']):
            return False

        return True

    def score_tan_tui(self, frame_data):
        """对弹腿动作进行打分"""
        if not self.is_tan_tui_frame(frame_data):
            return 0
            
        scores = {}
        
        # 1. 踢腿高度评分
        kick_height_ratio = self._calculate_kick_height_ratio(frame_data)
        scores['kick_height'] = min(100, kick_height_ratio * 100)
        
        # 2. 支撑腿稳定性评分
        support_leg_score = self._evaluate_support_leg(frame_data)
        scores['support_leg'] = support_leg_score
        
        # 3. 躯干垂直度评分
        body_vertical_score = self._evaluate_body_vertical(frame_data)
        scores['body_vertical'] = body_vertical_score
        
        # 4. 爆发力评分（需要连续帧数据）
        scores['explosive_power'] = 80  # 默认分数，实际应基于连续帧分析
        
        # 计算总分
        final_score = sum(score * self.weights[key] for key, score in scores.items())
        
        return final_score

    def _calculate_kick_height_ratio(self, frame_data):
        """计算踢腿高度比例"""
        is_left_support = frame_data['左踝']['y'] > frame_data['右踝']['y']
        if is_left_support:
            support_knee_y = frame_data['左膝']['y']
            kick_ankle_y = frame_data['右踝']['y']
        else:
            support_knee_y = frame_data['右膝']['y']
            kick_ankle_y = frame_data['左踝']['y']
            
        height_diff = support_knee_y - kick_ankle_y
        return height_diff / self.standards['min_kick_height']

    def _evaluate_support_leg(self, frame_data):
        """评估支撑腿稳定性"""
        is_left_support = frame_data['左踝']['y'] > frame_data['右踝']['y']
        if is_left_support:
            angle = self.calculate_angle(
                frame_data['左髋'],
                frame_data['左膝'],
                frame_data['左踝']
            )
        else:
            angle = self.calculate_angle(
                frame_data['右髋'],
                frame_data['右膝'],
                frame_data['右踝']
            )
            
        return 100 - abs(angle - self.standards['support_leg_angle'])

    def _is_heel_lifted(self, ankle):
        """检查脚跟是否离地"""
        return ankle['y'] < self.standards['heel_ground_threshold']

    def detect_key_frames(self, frame_sequence):
        """检测关键帧序列"""
        key_frames = []
        potential_key_frame_count = 0
        
        for i in range(len(frame_sequence)):
            if i - self.last_key_frame < self.min_frame_interval:
                continue
                
            frame_data = frame_sequence[i]
            
            if self.is_tan_tui_frame(frame_data):
                potential_key_frame_count += 1
                
                if potential_key_frame_count >= self.consecutive_frames:
                    start_idx = i - self.consecutive_frames + 1
                    scores = [self.score_tan_tui(frame_sequence[j]) 
                            for j in range(start_idx, i + 1)]
                    
                    best_frame_idx = start_idx + scores.index(max(scores))
                    
                    if best_frame_idx not in key_frames:
                        key_frames.append(best_frame_idx)
                        self.last_key_frame = best_frame_idx
                        potential_key_frame_count = 0
            else:
                potential_key_frame_count = 0
                
        return key_frames


    def analyze_sequence(self, frame_sequence):

        """
        分析整个弹腿动作序列
        frame_sequence: 包含连续帧数据的列表
        返回关键帧信息和得分
        """
        key_frames = self.detect_key_frames(frame_sequence)
        analysis_result_tantui = {
            'key_frames': key_frames,
            'scores': [],
            'details': []
        }
        
        for frame_idx in key_frames:
            frame_data = frame_sequence[frame_idx]
            score = self.score_tan_tui(frame_data)
            
            # 判断支撑腿
            is_left_support = frame_data['左踝']['y'] > frame_data['右踝']['y']
            if is_left_support:
                support_leg = {
                    'hip': frame_data['左髋'],
                    'knee': frame_data['左膝'],
                    'ankle': frame_data['左踝']
                }
                kick_leg = {
                    'hip': frame_data['右髋'],
                    'knee': frame_data['右膝'],
                    'ankle': frame_data['右踝']
                }
            else:
                support_leg = {
                    'hip': frame_data['右髋'],
                    'knee': frame_data['右膝'],
                    'ankle': frame_data['右踝']
                }
                kick_leg = {
                    'hip': frame_data['左髋'],
                    'knee': frame_data['左膝'],
                    'ankle': frame_data['左踝']
                }

            # 添加得分信息
            analysis_result_tantui['scores'].append({
                'frame_index': frame_idx,
                'score': score,
                'support_leg': 'left' if is_left_support else 'right'
            })
            
            # 添加详细分析信息
            analysis_result_tantui['details'].append({
                'frame_index': frame_idx,
                'support_leg_angle': self.calculate_angle(
                    support_leg['hip'],
                    support_leg['knee'],
                    support_leg['ankle']
                ),
                'kick_leg_angle': self.calculate_angle(
                    kick_leg['hip'],
                    kick_leg['knee'],
                    kick_leg['ankle']
                ),
                'kick_height_ratio': self._calculate_kick_height_ratio(frame_data),
                'is_heel_lifted': self._is_heel_lifted(support_leg['ankle']),
                'support_leg': 'left' if is_left_support else 'right'
            })
            
        return analysis_result_tantui

    def _evaluate_body_vertical(self, frame_data):
        """评估躯干垂直度"""
        try:
            # 检查是否有所需的关键点
            required_points = ['左髋', '右髋', '左肩', '右肩']
            if not all(frame_data.get(point) for point in required_points):
                return 0

            # 获取髋部中点作为躯干底部参考点
            hip_mid_x = (frame_data['左髋']['x'] + frame_data['右髋']['x']) / 2
            hip_mid_y = (frame_data['左髋']['y'] + frame_data['右髋']['y']) / 2
            
            # 获取肩部中点作为躯干顶部参考点
            shoulder_mid_x = (frame_data['左肩']['x'] + frame_data['右肩']['x']) / 2
            shoulder_mid_y = (frame_data['左肩']['y'] + frame_data['右肩']['y']) / 2
            
            # 计算躯干与垂直线的夹角
            # 垂直线的方向向量是(0, -1)，因为y轴向下为正
            trunk_vector = np.array([shoulder_mid_x - hip_mid_x, shoulder_mid_y - hip_mid_y])
            vertical_vector = np.array([0, -1])
            
            # 标准化向量
            trunk_norm = np.linalg.norm(trunk_vector)
            if trunk_norm == 0:
                return 0
            trunk_vector = trunk_vector / trunk_norm
            
            # 计算夹角（弧度）
            cos_angle = np.dot(trunk_vector, vertical_vector)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angle_deg = np.degrees(angle)
            
            # 计算分数：角度越接近90度，分数越高
            # 允许10度的误差范围
            max_deviation = 10
            deviation = abs(angle_deg - self.standards['vertical_angle'])
            if deviation <= max_deviation:
                score = 100 * (1 - deviation / max_deviation)
            else:
                score = max(0, 100 * (1 - deviation / 90))
                
            return score

        except Exception as e:
            print(f"计算躯干垂直度时出错: {str(e)}")
            return 0

    def calculate_angle(self, point1, point2, point3):
        """
        计算三个点形成的角度
        point1, point2, point3: 包含x,y,z坐标的字典
        返回角度（度数）
        """
        try:
            # 转换为numpy数组便于计算
            a = np.array([point1['x'], point1['y'], point1.get('z', 0)])
            b = np.array([point2['x'], point2['y'], point2.get('z', 0)])
            c = np.array([point3['x'], point3['y'], point3.get('z', 0)])
            
            # 计算向量
            ba = a - b
            bc = c - b
            
            # 计算夹角的余弦值
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            # 防止数值误差导致的 domain error
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            
            # 转换为角度
            angle = np.degrees(np.arccos(cosine_angle))
            
            return angle
            
        except Exception as e:
            print(f"计算角度时出错: {str(e)}")
            return 0
