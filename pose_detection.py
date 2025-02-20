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
        self.pose = self.mp_pose.Pose(static_image_mode=False,
                                     model_complexity=1,
                                     min_detection_confidence=0.5,
                                     min_tracking_confidence=0.5)
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
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("视频读取完成或出错")
                break

            # 将BGR图像转换为RGB图像
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 处理图像
            results = self.pose.process(frame_rgb)

            # 在图像上绘制姿态标记
            if results.pose_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # 保存坐标数据
                coordinates = []
                for landmark in results.pose_landmarks.landmark:
                    coordinates.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                
                # 创建新的输出文件夹
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
                # 保存到文件
                filename = os.path.join(output_dir, f'frame_{frame_count}.txt')
                with open(filename, 'w', encoding='utf-8') as f:
                    for i, coord in enumerate(coordinates):
                        body_part = self.BODY_PARTS.get(i, f"未知点{i}")
                        f.write(f"{body_part}: x={coord['x']:.4f}, y={coord['y']:.4f}, z={coord['z']:.4f}, v={coord['visibility']:.4f}\n")

            # 显示结果
            cv2.imshow('Pose Detection', frame)
            frame_count += 1  # 递增帧计数器

            # 按'q'键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
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
                # 保存图片
                output_path = os.path.join(output_dir, f'frame_{frame_count}.jpg')
                cv2.imwrite(output_path, frame)
                print(f"已保存第 {frame_count} 帧到: {output_path}")

            frame_count += 1

        cap.release()
        print(f"所有指定帧的图片已保存到文件夹: {output_dir}")

def main():
    detector = PoseDetector()
    # 指定要导出的帧号
    frame_numbers = [
                195,
    399,
    607,
    662
  ]
    detector.export_frames('1.mp4', frame_numbers)

if __name__ == "__main__":
    main()