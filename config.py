# 姿态检测配置参数
POSE_CONFIG = {
    # 模型配置
    'static_image_mode': False,
    'model_complexity': 1,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
    
    # 可视化配置
    'draw_landmarks': True,
    'draw_connections': True,
    
    # 输出配置
    'save_coordinates': True,
    'output_folder': 'output',
    
    # 摄像头配置
    'camera_id': 0,  # 默认使用第一个摄像头
    'camera_width': 640,
    'camera_height': 480,
    'camera_fps': 30
}