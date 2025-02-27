import os
import json

def compare_frame_data(frame_number):
    """比较不同目录下同一帧的数据"""
    
    # 读取两个目录的数据
    path1 = f"output1/frame_{frame_number}.txt"  # 修正文件名
    path2 = f"selected_frames/frame_{frame_number}_data.txt"
    
    def read_data(filepath):
        if not os.path.exists(filepath):
            print(f"文件不存在: {filepath}")
            return None
            
        data = {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        # 处理每一行数据
                        parts = line.strip().split(': ')
                        if len(parts) == 2:
                            key = parts[0]
                            # 解析 x=0.1698, y=0.5138, z=-0.1414, v=0.9987 格式
                            values = {}
                            for item in parts[1].split(', '):
                                k, v = item.split('=')
                                values[k] = float(v)
                            data[key] = values
                    except Exception as e:
                        print(f"解析行出错: {line.strip()}")
                        print(f"错误信息: {str(e)}")
                        continue
        except Exception as e:
            print(f"读取文件出错: {filepath}")
            print(f"错误信息: {str(e)}")
            return None
            
        return data
    
    # 读取并比较数据
    data1 = read_data(path1)
    data2 = read_data(path2)
    
    if data1 is None or data2 is None:
        return
    
    # 比较数据
    print(f"\n比较第 {frame_number} 帧数据:")
    print(f"文件1: {path1}")
    print(f"文件2: {path2}\n")
    
    all_keys = set(data1.keys()) | set(data2.keys())
    for key in sorted(all_keys):
        if key not in data1:
            print(f"{key}: 在文件1中缺失")
            continue
        if key not in data2:
            print(f"{key}: 在文件2中缺失")
            continue
            
        for coord in ['x', 'y', 'z', 'v']:
            val1 = data1[key].get(coord)
            val2 = data2[key].get(coord)
            if val1 is None or val2 is None:
                print(f"{key} {coord}: 坐标缺失")
                continue
            if abs(val1 - val2) > 0.0001:
                print(f"{key} {coord}: {val1:.4f} vs {val2:.4f}, 差值: {abs(val1-val2):.4f}")

if __name__ == "__main__":
    compare_frame_data(30)