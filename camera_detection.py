import cv2
import mediapipe as mp
import numpy as np
import os
from config import POSE_CONFIG
from pose_detection import PoseDetector

class CameraDetector:
    def __init__(self):
        self.detector = PoseDetector()
        self.config = POSE_CONFIG
        
        # 创建输出文件夹
        if self.config['save_coordinates'] and not os.path.exists(self.config['output_folder']):
            os.makedirs(self.config['output_folder'])
    
    def start_detection(self):
        # 初始化摄像头
        cap = cv2.VideoCapture(self.config['camera_id'])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera_width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera_height'])
        cap.set(cv2.CAP_PROP_FPS, self.config['camera_fps'])
        
        frame_count = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("无法获取摄像头画面")
                break
                
            # 处理图像
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.pose.process(frame_rgb)
            
            # 绘制姿态标记
            if results.pose_landmarks:
                if self.config['draw_landmarks']:
                    self.detector.mp_draw.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        self.detector.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.detector.mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                
                # 保存坐标数据
                if self.config['save_coordinates']:
                    coordinates = []
                    for landmark in results.pose_landmarks.landmark:
                        coordinates.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z,
                            'visibility': landmark.visibility
                        })
                    
                    # 保存到文件
                    filename = os.path.join(self.config['output_folder'], f'frame_{frame_count}.txt')
                    with open(filename, 'w') as f:
                        for i, coord in enumerate(coordinates):
                            body_part = self.detector.BODY_PARTS.get(i, f"未知点{i}")
                        f.write(f"{body_part}: x={coord['x']:.4f}, y={coord['y']:.4f}, z={coord['z']:.4f}, v={coord['visibility']:.4f}\n")
            
            # 显示帧率
            cv2.putText(frame, f'FPS: {int(cap.get(cv2.CAP_PROP_FPS))}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示结果
            cv2.imshow('Camera Pose Detection', frame)
            frame_count += 1
            
            # 按'q'键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    detector = CameraDetector()
    detector.start_detection()

if __name__ == "__main__":
    main()