import numpy as np
from pose_analysis_gongbu import PoseAnalyzer

def create_test_frame(raw_data_text):
    """创建测试帧数据"""
    frame_data = {}
    lines = raw_data_text.strip().split('\n')
    
    for line in lines:
        try:
            key, values = line.split(': ')
            # 分离 x,y,z,v 值
            coords = {}
            for item in values.split(', '):
                coord, value = item.split('=')
                coords[coord] = float(value)
            frame_data[key] = coords
        except Exception as e:
            print(f"解析行失败: {line}")
            print(f"错误信息: {str(e)}")
            continue
            
    return frame_data

def analyze_frame(raw_data_text):
    """分析测试帧"""
    analyzer = PoseAnalyzer()
    frame_data = create_test_frame(raw_data_text)
    
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

# 使用示例
raw_data = """左髋: x=0.7110, y=0.8464, z=-0.0463, v=0.9992
右髋: x=0.7014, y=0.8454, z=0.0463, v=0.9991
左膝: x=0.6974, y=0.9303, z=-0.0522, v=0.9431
右膝: x=0.6858, y=0.9175, z=0.0602, v=0.9650
左踝: x=0.7460, y=0.9598, z=0.0008, v=0.9663
右踝: x=0.6994, y=1.0024, z=0.0919, v=0.9824
"""

if __name__ == "__main__":
    analyze_frame(raw_data)