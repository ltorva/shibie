import cv2
import torch
import numpy as np
import os
from yolov7.models.experimental import attempt_load
from hybrik.models import create_model
from hybrik.utils.config import update_config

class Pose3DDetector:
    # 定义扩展的身体部位映射
    BODY_PARTS_3D = {
        # 躯干
        0: "鼻子", 1: "颈部", 2: "胸部中心",
        3: "左肩", 4: "右肩",
        5: "左肘", 6: "右肘",
        7: "左手腕", 8: "右手腕",
        9: "左髋", 10: "右髋",
        11: "左膝", 12: "右膝",
        13: "左踝", 14: "右踝",
        # 手部细节
        15: "左手指", 16: "右手指",
        17: "左拇指", 18: "右拇指",
        # 脚部细节
        19: "左脚跟", 20: "右脚跟",
        21: "左脚尖", 22: "右脚尖"
    }

    def __init__(self):
        # 初始化YOLO-Pose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo_model = attempt_load('yolov7-w6-pose.pt', self.device)
        
        # 初始化HybrIK
        hybrik_cfg = update_config('configs/hybrik_config.yaml')
        self.hybrik_model = create_model(hybrik_cfg).to(self.device)
        self.hybrik_model.load_state_dict(torch.load('hybrik_checkpoint.pth'))
        self.hybrik_model.eval()

    def process_video(self, video_path):
        # 检查文件是否存在
        if not os.path.exists(video_path):
            print(f"Error: 视频文件不存在: {video_path}")
            return

        # 初始化视频捕获
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        if not cap.isOpened():
            print(f"Error: 无法打开视频文件: {video_path}")
            return

        # 创建输出目录
        if not os.path.exists('output_3d'):
            os.makedirs('output_3d')

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("视频读取完成或出错")
                break

            # YOLO-Pose 检测
            img = torch.from_numpy(frame).to(self.device).float() / 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # 获取2D姿态
            pred = self.yolo_model(img)[0]
            
            if len(pred):
                # HybrIK 3D重建
                pose_2d = pred[0]['keypoints'].cpu().numpy()
                pose_3d = self.hybrik_model(pose_2d)

                # 绘制3D姿态
                self._draw_3d_pose(frame, pose_3d)

                # 保存坐标数据
                filename = os.path.join('output_3d', f'frame_{frame_count}.txt')
                self._save_3d_coordinates(filename, pose_3d)

            # 显示结果
            cv2.imshow('3D Pose Detection', frame)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def _draw_3d_pose(self, frame, pose_3d):
        # 绘制3D姿态的可视化
        # ...此处实现3D姿态的绘制逻辑...
        pass

    def _save_3d_coordinates(self, filename, pose_3d):
        with open(filename, 'w') as f:
            for i, coord in enumerate(pose_3d):
                body_part = self.BODY_PARTS_3D.get(i, f"未知点{i}")
                f.write(f"{body_part}: x={coord[0]:.4f}, y={coord[1]:.4f}, z={coord[2]:.4f}\n")

def main():
    detector = Pose3DDetector()
    video_path = input("请输入视频文件路径: ")
    detector.process_video(video_path)

if __name__ == "__main__":
    main()