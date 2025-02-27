import cv2
import mediapipe as mp
import numpy as np
import os
import glob

class PoseDetector:
    # 定义身体部位映射
    BODY_PARTS = {
        0: "鼻子",
        1: "左眼(内)", 2: "左眼", 3: "左眼(外)",
        4: "右眼(内)", 5: "右眼", 6: "右眼(外)",
        7: "左耳", 8: "右耳",
        9: "嘴(左)", 10: "嘴(右)",
        11: "左肩", 12: "右肩",
        13: "左肘", 14: "右肘",
        15: "左手腕", 16: "右手腕",
        17: "左手", 18: "右手",
        19: "左小指", 20: "右小指",
        21: "左食指", 22: "右食指",
        23: "左髋", 24: "右髋",
        25: "左膝", 26: "右膝",
        27: "左踝", 28: "右踝",
        29: "左脚", 30: "右脚",
        31: "左脚趾", 32: "右脚趾"
    }
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,     # 动态视频模式
            model_complexity=1,          # 提高模型复杂度 (0-2)
            smooth_landmarks=True,       # 启用平滑
            enable_segmentation=True,    # 启用分割以提高准确性
            smooth_segmentation=True,    # 平滑分割
            min_detection_confidence=0.6, # 提高检测置信度
            min_tracking_confidence=0.6   # 提高追踪置信度
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def get_next_output_folder(self, base_dir):
        # 查找所有output开头的文件夹
        output_folders = glob.glob(os.path.join(base_dir, 'output*'))
        if not output_folders:
            return os.path.join(base_dir, 'output1')
        
        # 提取所有文件夹的编号
        numbers = []
        for folder in output_folders:
            try:
                num = int(folder.replace(os.path.join(base_dir, 'output'), ''))
                numbers.append(num)
            except ValueError:
                continue
        
        # 如果没有编号文件夹，返回output1
        if not numbers:
            return os.path.join(base_dir, 'output1')
        
        # 返回最大编号+1的文件夹名
        next_num = max(numbers) + 1
        return os.path.join(base_dir, f'output{next_num}')

    def process_video(self, video_path):
        # 检查文件是否存在
        if not os.path.exists(video_path):
            print(f"Error: 视频文件不存在: {video_path}")
            return
            
        # 尝试打开视频文件
        cap = cv2.VideoCapture(video_path)
        frame_count = 0  # 初始化帧计数器
        if not cap.isOpened():
            print(f"Error: 无法打开视频文件: {video_path}")
            print("可能的原因：")
            print("1. 视频文件格式不受支持")
            print("2. 视频文件可能已损坏")
            print("3. 缺少必要的视频解码器")
            return

        # 获取新的输出文件夹路径
        output_dir = self.get_next_output_folder(os.path.dirname(video_path))
        
        # 获取视频基本信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 创建视频写入器保存处理后的视频
        output_video_path = os.path.join(output_dir, 'processed_video.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("视频读取完成或出错")
                break

            # 保存原始帧
            original_frame = frame.copy()
            
            # 预处理用于显示和检测的帧
            processed_frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
            processed_frame = cv2.convertScaleAbs(processed_frame, alpha=1.2, beta=10)
            processed_frame = cv2.GaussianBlur(processed_frame, (3, 3), 0)
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # 处理图像
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                # 在处理后的帧上绘制姿态标记
                self.mp_draw.draw_landmarks(
                    processed_frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # 添加帧号信息
                frame_info = f'Frame: {frame_count}'
                cv2.putText(processed_frame, frame_info, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 保存坐标数据 - 使用原始尺寸
                coordinates = []
                for landmark in results.pose_landmarks.landmark:
                    # 还原到原始尺寸
                    x = landmark.x / 0.8  # 还原缩放
                    y = landmark.y / 0.8
                    coordinates.append({
                        'x': x,
                        'y': y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                
                # 创建输出文件夹
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
                # 只保存一次坐标数据，使用原始尺寸
                filename = os.path.join(output_dir, f'frame_{frame_count}.txt')
                with open(filename, 'w', encoding='utf-8') as f:
                    for i, coord in enumerate(coordinates):
                        body_part = self.BODY_PARTS.get(i, f"未知点{i}")
                        f.write(f"{body_part}: x={coord['x']:.4f}, y={coord['y']:.4f}, z={coord['z']:.4f}, v={coord['visibility']:.4f}\n")

                # 保存处理后的帧用于视频输出
                resized_processed = cv2.resize(processed_frame, (frame_width, frame_height))
                out.write(resized_processed)
                
                # 显示处理后的帧
                cv2.imshow('Pose Detection', processed_frame)
                
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"坐标数据已保存到文件夹: {output_dir}")

    def export_frames(self, video_path, frame_numbers):
        # 检查文件是否存在
        if not os.path.exists(video_path):
            print(f"Error: 视频文件不存在: {video_path}")
            return
            
        # 尝试打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: 无法打开视频文件: {video_path}")
            print("可能的原因：")
            print("1. 视频文件格式不受支持")
            print("2. 视频文件可能已损坏")
            print("3. 缺少必要的视频解码器")
            return

        # 创建输出文件夹
        output_dir = os.path.join(os.path.dirname(video_path), 'selected_frames')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        frame_count = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if frame_count in frame_numbers:
                # 添加与 process_video 相同的预处理步骤
                frame = cv2.resize(frame, (0, 0), fx=0.8, fy=0.8)
                frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
                frame = cv2.GaussianBlur(frame, (3, 3), 0)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                results = self.pose.process(frame_rgb)

                if results.pose_landmarks:
                    # 绘制骨骼点和连接线
                    self.mp_draw.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    
                    # 添加帧号和姿态信息
                    frame_info = f'Frame: {frame_count}'
                    cv2.putText(frame, frame_info, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # 保存坐标数据用于姿态分析
                    coordinates = []
                    for landmark in results.pose_landmarks.landmark:
                        coordinates.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z,
                            'visibility': landmark.visibility
                        })
                    
                    # 保存带有骨骼点的图片
                    output_path = os.path.join(output_dir, f'frame_{frame_count}.jpg')
                    cv2.imwrite(output_path, frame)
                    
                    # 保存姿态数据
                    data_path = os.path.join(output_dir, f'frame_{frame_count}_data.txt')
                    with open(data_path, 'w', encoding='utf-8') as f:
                        for i, coord in enumerate(coordinates):
                            body_part = self.BODY_PARTS.get(i, f"未知点{i}")
                            f.write(f"{body_part}: x={coord['x']:.4f}, y={coord['y']:.4f}, z={coord['z']:.4f}, v={coord['visibility']:.4f}\n")
                    
                    print(f"已保存第 {frame_count} 帧及其数据")

            frame_count += 1

        cap.release()
        print(f"所有帧和数据已保存到: {output_dir}")

def main():
    detector = PoseDetector()

    video_path = '1.mp4'  # 确保视频文件在正确的路径
    # 检查文件是否存在
    if not os.path.exists(video_path):
        print(f"Error: 视频文件不存在: {video_path}")
        return
    # 调用实例方法而不是类方法
    #detector.process_video(video_path)
    
    # 指定要导出的帧号
    frame_numbers = [
                125,
    190,
    255,
    282,
    356,
    382,
    455,
    472,
    503,
    598,
    641,
    666,
    691
  ]
    detector.export_frames('1.mp4', frame_numbers)

if __name__ == "__main__":
    main()