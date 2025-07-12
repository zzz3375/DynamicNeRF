import cv2
import os

# 设置参数
image_folder = "render_results/plain" # 图片文件夹路径
video_name = 'output_video.mp4'  # 输出视频文件名
fps = 30  # 帧率（每秒多少张图片）

# 获取所有图片并按文件名排序
images = [img for img in os.listdir(image_folder) if img.endswith(('.jpg', '.png', '.jpeg'))]
images.sort()  # 确保图片按顺序排列

# 读取第一张图片，获取尺寸
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# 创建视频写入对象
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# 逐帧写入视频
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)

# 释放资源
video.release()
print(f"视频已生成: {video_name}")