import numpy as np
from pose_analysis import PoseAnalyzer

def create_test_frame():
    """创建测试帧数据"""
    return {
        '左髋': {'x': 0.1805, 'y': 0.6718, 'z': -0.0258, 'v': 0.9980},
        '右髋': {'x': 0.1630, 'y': 0.6705, 'z': 0.0258, 'v': 0.9972},
        '左膝': {'x': 0.1827, 'y': 0.7459, 'z': -0.0067, 'v': 0.9830},
        '右膝': {'x': 0.1673, 'y': 0.7423, 'z': 0.0732, 'v': 0.6769},
        '左踝': {'x': 0.1820, 'y': 0.8047, 'z': 0.0610, 'v': 0.9329},
        '右踝': {'x': 0.1758, 'y': 0.7889, 'z': 0.1779, 'v': 0.8309},
    }

def analyze_frame():
    """分析测试帧"""
    analyzer = PoseAnalyzer()
    frame_data = create_test_frame()
    
    print("1. 检查双脚高度差:")
    feet_height_diff = abs(frame_data['左踝']['y'] - frame_data['右踝']['y'])
    print(f"   双脚高度差: {feet_height_diff:.4f}")
    print(f"   最大允许高度差: {0.05}")
    print(f"   是否符合要求: {feet_height_diff < 0.05}")
    
    print("\n2. 检查腿部z轴差值:")
    left_leg_z_diff = abs(frame_data['左髋']['z'] - frame_data['左膝']['z']) + \
                      abs(frame_data['左膝']['z'] - frame_data['左踝']['z'])
    right_leg_z_diff = abs(frame_data['右髋']['z'] - frame_data['右膝']['z']) + \
                       abs(frame_data['右膝']['z'] - frame_data['右踝']['z'])
    print(f"   左腿z轴差值: {left_leg_z_diff:.4f}")
    print(f"   右腿z轴差值: {right_leg_z_diff:.4f}")
    print(f"   最大允许差值: 0.1")
    
    print("\n3. 计算膝盖角度:")
    left_knee_angle = analyzer.calculate_angle(
        frame_data['左髋'], frame_data['左膝'], frame_data['左踝'])
    right_knee_angle = analyzer.calculate_angle(
        frame_data['右髋'], frame_data['右膝'], frame_data['右踝'])
    print(f"   左膝角度: {left_knee_angle:.2f}°")
    print(f"   右膝角度: {right_knee_angle:.2f}°")
    print(f"   标准前腿角度: {analyzer.standards['front_knee_angle']}°")
    print(f"   标准后腿角度: {analyzer.standards['back_knee_angle']}°")
    
    print("\n4. 检查髋部高度差:")
    hip_height_diff = abs(frame_data['左髋']['y'] - frame_data['右髋']['y'])
    print(f"   髋部高度差: {hip_height_diff:.4f}")
    
    print("\n5. 弓步判断结果:")
    is_gong_bu = analyzer.is_gong_bu_frame(frame_data)
    print(f"   是否判定为弓步: {is_gong_bu}")
    
    if is_gong_bu:
        print("\n6. 弓步得分:")
        score = analyzer.score_gong_bu(frame_data)
        print(f"   总分: {score:.2f}")

if __name__ == "__main__":
    analyze_frame()