import numpy as np

class TanTuiDengTuiScorer:
    def __init__(self):
        # 动作规格标准
        self.standards = {
            'kick_height_threshold': 1.0,  # 脚跟需高于支撑腿膝盖
            'heel_ground_threshold': 0.05,  # 脚跟离地判定阈值
            'min_knee_angle': 30,  # 最小屈膝角度(判断屈伸过程)
            'min_straight_angle': 165  # 最小伸直角度
        }

    def score_sequence(self, analysis_result, frame_sequence):
        """评分主函数"""
        total_score = 10.0  # 满分10分
        deductions = {
            'specs': [],  # 规格扣分
            'errors': [],  # 错误扣分
            'performance': []  # 演练扣分
        }

        # 遍历每个关键帧的详细信息
        for detail in analysis_result['details']:
            frame_idx = detail['frame_index']
            frame_data = frame_sequence[frame_idx]
            
            # 1. 检查规格要求
            specs = self._check_specifications(detail, frame_data)
            deductions['specs'].extend(specs)
            
            # 2. 检查动作错误
            errors = self._check_errors(detail, frame_sequence, frame_idx)
            deductions['errors'].extend(errors)

        # 计算规格扣分
        specs_count = len(deductions['specs'])
        if specs_count > 0:
            specs_deduction = min(3.1, {
                1: 0.1,
                2: 1.1,
                3: 2.1,
                4: 3.1
            }.get(specs_count, 3.1))
        else:
            specs_deduction = 0

        # 计算错误扣分
        error_deduction = len(deductions['errors']) * 0.1

        # 最终分数
        final_score = total_score - specs_deduction - error_deduction
        final_score = max(0, min(10, final_score))

        return {
            'score': final_score,
            'deductions': deductions,
            'details': {
                'specs_deduction': specs_deduction,
                'error_deduction': error_deduction
            }
        }

    def _check_specifications(self, detail, frame_data):
        """检查动作规格"""
        specs_errors = []
        
        # 获取支撑腿和踢腿信息
        is_left_support = frame_data['左踝']['y'] > frame_data['右踝']['y']
        
        # 获取支撑腿膝盖和踢腿脚踝的y坐标
        if is_left_support:
            support_knee_y = frame_data['左膝']['y']
            kick_ankle_y = frame_data['右踝']['y']
        else:
            support_knee_y = frame_data['右膝']['y']
            kick_ankle_y = frame_data['左踝']['y']
        
        # 检查脚跟高度是否达标（y值越小表示位置越高）
        if kick_ankle_y >= support_knee_y:  # 如果踢腿脚踝的y坐标大于等于支撑腿膝盖的y坐标
            specs_errors.append({
                'type': 'height',
                'message': '脚跟未高于支撑腿膝盖',
                'frame': detail['frame_index'],
                'details': {
                    'kick_ankle_y': kick_ankle_y,
                    'support_knee_y': support_knee_y
                }
            })

        return specs_errors

    def _check_errors(self, detail, frame_sequence, current_idx):
        """检查动作错误"""
        errors = []
        
        # 1. 检查支撑脚跟离地
        if detail['is_heel_lifted']:
            errors.append({
                'type': 'heel_lifted',
                'message': '支撑脚脚跟离地',
                'frame': current_idx
            })

        # 2. 检查屈伸过程
        """
                if not self._check_bend_straight_process(frame_sequence, current_idx):
            errors.append({
                'type': 'no_bend_straight',
                'message': '没有屈伸过程',
                'frame': current_idx
            })
        """


        # 3. 检查蹬、弹发力方式
        """
                if not self._check_force_pattern(frame_sequence, current_idx, detail['motion_type']):
            errors.append({
                'type': 'wrong_force_pattern',
                'message': '蹬、弹发力方式错误',
                'frame': current_idx
            })

         """

        return errors

    def _check_bend_straight_process(self, frame_sequence, current_idx):
        """检查是否有屈伸过程"""
        # 检查前5帧的角度变化
        start_idx = max(0, current_idx - 5)
        sequence = frame_sequence[start_idx:current_idx + 1]
        
        min_angle = float('inf')
        max_angle = 0
        
        for frame in sequence:
            # 获取踢腿角度
            kick_leg_angle = self._get_kick_leg_angle(frame)
            min_angle = min(min_angle, kick_leg_angle)
            max_angle = max(max_angle, kick_leg_angle)
        
        # 必须有明显的屈伸过程：最小角度要足够小，最大角度要足够大
        return (min_angle <= self.standards['min_knee_angle'] and 
                max_angle >= self.standards['min_straight_angle'])

    def _check_force_pattern(self, frame_sequence, current_idx, motion_type):
        """检查发力方式是否正确"""
        # 弹腿：快速屈伸
        # 蹬腿：匀速伸展
        if motion_type == 'tantui':
            return self._check_tantui_force(frame_sequence, current_idx)
        else:
            return self._check_dengtui_force(frame_sequence, current_idx)

    def _get_kick_leg_angle(self, frame_data):
        """获取踢腿角度"""
        is_left_support = frame_data['左踝']['y'] > frame_data['右踝']['y']
        if is_left_support:
            return self._calculate_angle(
                frame_data['右髋'],
                frame_data['右膝'],
                frame_data['右踝']
            )
        return self._calculate_angle(
            frame_data['左髋'],
            frame_data['左膝'],
            frame_data['左踝']
        )

    def _calculate_angle(self, point1, point2, point3):
        """计算角度"""
        # 与PoseAnalyzer中相同的角度计算方法
        try:
            a = np.array([point1['x'], point1['y']])
            b = np.array([point2['x'], point2['y']])
            c = np.array([point3['x'], point3['y']])
            
            ba = a - b
            bc = c - b
            
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
            
            return angle
        except:
            return 0