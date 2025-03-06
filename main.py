from pose_analysis_gongbu import PoseAnalyzer_gongbu
from pose_analysis_tantui import PoseAnalyzer_tantui
from score_tantuidengtui import TanTuiDengTuiScorer
import os
import json

def load_sequence_data(output_folder):
    """加载整个序列的帧数据"""
    frame_sequence = []
    
    # 打印当前工作目录和目标文件夹
    print(f"当前工作目录: {os.getcwd()}")
    print(f"要读取的文件夹: {os.path.abspath(output_folder)}")
    
    # 检查文件夹是否存在
    if not os.path.exists(output_folder):
        raise FileNotFoundError(f"文件夹不存在: {output_folder}")
    
    # 修改文件匹配模式
    frame_files = []
    for f in os.listdir(output_folder):
        try:
            # 修改为匹配 frame_0.txt 格式
            if f.startswith('frame_') and f.endswith('.txt'):
                frame_num = int(f[6:-4])  # 提取frame_X.txt中的X
                frame_files.append((frame_num, f))
                print(f"找到文件: {f}")
        except ValueError as e:
            print(f"跳过文件 {f}: {str(e)}")
    
    if not frame_files:
        raise ValueError(f"在 {output_folder} 中没有找到帧数据文件")
    
    # 按帧号排序
    frame_files.sort(key=lambda x: x[0])
    print(f"\n找到 {len(frame_files)} 个帧文件")


    
    # 处理每个帧文件
    for frame_num, frame_file in frame_files:
        try:
            file_path = os.path.join(output_folder, frame_file)
            print(f"处理: {frame_file}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                frame_data = {}
                for line in f:
                    if ':' not in line:
                        continue
                    
                    try:
                        key, value_str = line.strip().split(': ')
                        values = {}
                        
                        for item in value_str.split(', '):
                            coord, val = item.split('=')
                            values[coord] = float(val)
                            
                        frame_data[key] = values
                    except Exception as e:
                        print(f"解析错误 - 行: {line.strip()}")
                        print(f"错误信息: {str(e)}")
                
                frame_sequence.append(frame_data)
                
        except Exception as e:
            print(f"处理文件失败: {frame_file}")
            print(f"错误信息: {str(e)}")
    
    print(f"成功加载 {len(frame_sequence)} 帧数据")
    return frame_sequence

def main():
    try:
        # 创建分析器实例
        #analyzer = PoseAnalyzer_gongbu()
        analyzer = PoseAnalyzer_tantui()
        score_analyzer = TanTuiDengTuiScorer()
        # 使用绝对路径加载数据
        output_folder = os.path.join(os.getcwd(), 'output1')
        frame_sequence = load_sequence_data(output_folder)
        
        if frame_sequence:
            # 分析序列
            result = analyzer.analyze_sequence(frame_sequence)
            
            #print(f"\n找到 {len(result['key_frames'])} 个关键帧:")
            for score_info in result['scores']:
                print(f"帧 {score_info['frame_index']}: 得分 {score_info['score']:.2f}")
                
            # 保存结果
            with open('analysis_result_tantui.json', 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            score_result = score_analyzer.score_sequence(result, frame_sequence)
            print(f"总分: {score_result['score']:.1f}")
            print("\n规格扣分:")
            for spec in score_result['deductions']['specs']:
                print(f"- 第{spec['frame']}帧: {spec['message']}")
    
            print("\n动作错误:")
            for error in score_result['deductions']['errors']:
                print(f"- 第{error['frame']}帧: {error['message']}")            
                
    except Exception as e:
        print(f"程序执行出错: {str(e)}")

if __name__ == "__main__":
    main()