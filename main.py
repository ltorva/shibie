from pose_analysis import PoseAnalyzer
import os
import json

def load_sequence_data(output_folder):
    """加载整个序列的帧数据"""
    frame_sequence = []
    frame_files = sorted([f for f in os.listdir(output_folder) if f.startswith('frame_')])
    
    for frame_file in frame_files:
        with open(os.path.join(output_folder, frame_file), 'r') as f:
            frame_data = {}
            for line in f:
                parts = line.strip().split(': ')
                if len(parts) == 2:
                    key = parts[0]
                    values = parts[1].replace('x=', '').replace('y=', '').replace('z=', '').replace('v=', '')
                    x, y, z, v = map(float, values.split(', '))
                    frame_data[key] = {'x': x, 'y': y, 'z': z, 'v': v}
        frame_sequence.append(frame_data)
    
    return frame_sequence

def main():
    # 创建分析器实例
    analyzer = PoseAnalyzer()
    
    # 加载整个序列的数据
    frame_sequence = load_sequence_data('output1')
    
    # 分析序列
    result = analyzer.analyze_sequence(frame_sequence)
    
    # 输出结果
    print(f"找到 {len(result['key_frames'])} 个关键帧:")
    for score_info in result['scores']:
        print(f"帧 {score_info['frame_index']}: 得分 {score_info['score']:.2f}")
    
    # 保存详细分析结果
    with open('analysis_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()