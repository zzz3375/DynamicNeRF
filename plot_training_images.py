import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path  # 替换 os 导入
from PIL import Image
import io

# 设置字体为Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

def visualize_image_grid(log_path, tags_to_visualize, steps_to_visualize, save_path='image_grid.png', TAGS_TO_DISPLAY=None):
    """
    将多个标签在不同步数的图像数据组合成网格进行可视化
    
    Args:
        log_path: TensorBoard日志文件路径
        tags_to_visualize: 需要可视化的图像标签列表(作为行)
        steps_to_visualize: 需要可视化的步数列表(作为列)
        save_path: 可视化结果保存路径
    """
    # 加载TensorBoard数据
    print(f"Loading logs: {log_path}")
    ea = event_accumulator.EventAccumulator(
        log_path,
        size_guidance={
            event_accumulator.IMAGES: 0,  # 加载所有图像
        }
    )
    ea.Reload()  # 加载日志
    
    # 获取所有图像标签
    available_image_tags = ea.Tags()['images']
    print(f"Available image tags ({len(available_image_tags)}):")
    for tag in available_image_tags:
        print(f"  - {tag}")
    
    # 检查请求的标签是否存在
    missing_tags = [tag for tag in tags_to_visualize if tag not in available_image_tags]
    if missing_tags:
        print(f"Warning: Tags not found: {missing_tags}")
        tags_to_visualize = [tag for tag in tags_to_visualize if tag in available_image_tags]
        if not tags_to_visualize:
            print("No valid tags to visualize!")
            return
    
    # 创建图像网格
    rows = len(tags_to_visualize)
    cols = len(steps_to_visualize)
    
    # 动态调整图形大小
    fig_width = cols * 3  # 每列4英寸
    fig_height = rows * 2.0 # 每行3英寸
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    axes: list[list[plt.Axes]]
    
    # 如果只有一个图像，确保axes是二维数组
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    fontsize = 14
    # 填充图像网格
    for i, tag in enumerate(tags_to_visualize):
        tag_name = TAGS_TO_DISPLAY[i] 
        for j, step in enumerate(steps_to_visualize):
            ax: plt.Axes = axes[i, j]
            
            # 获取对应标签和步数的图像
            image_events = ea.Images(tag)
            image_event = next((e for e in image_events if e.step == step), None)
            
            if image_event is not None:
                # 将图像数据转换为numpy数组
                img = Image.open(io.BytesIO(image_event.encoded_image_string))
                img_np = np.array(img)
                
                # 显示图像
                ax.imshow(img_np)
                ax.set_xticks([])  # 隐藏x轴刻度
                ax.set_yticks([])
                if j==0: ax.set_ylabel(tag_name, fontsize=fontsize)
            
                if i==0: ax.set_title(f"{str(step)[:-3]}K steps", fontsize=fontsize)
            else:
                # 如果没有找到图像，显示空白并标记
                ax.text(0.5, 0.5, f"No Image\nTag: {tag}\nStep: {step}", 
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_facecolor('lightgray')
            
            # 隐藏坐标轴
            # ax.axis('off')
            
    
    # 设置行标签（标签名称）
    # for i, tag in enumerate(tags_to_visualize):
    #     axes[i, 0].set_ylabel(tag_name[i], fontsize=14, rotation=0, ha='right', va='center')
    
    # 设置总标题
    # plt.suptitle('TensorBoard Image Visualization Grid', fontsize=18, y=0.99)
    
    # 调整布局
    plt.tight_layout()  # 为suptitle留出空间
    
    # 保存图像（替换 os.makedirs 为 Path.mkdir）
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)  # 使用 pathlib 处理路径
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Grid visualization saved to {save_path}")
    
    # plt.show()

if __name__ == "__main__":
    # 指定TensorBoard日志路径
    LOG_PATH = 'logs/summaries/DJI_20250516151729_0005_V-100f-fps30_DyNeRF_pretrain_test/events.out.tfevents.1748093593.ZhizhangOFFICE.2938.0'
    
    # render
    # TAGS_TO_VISUALIZE = ['rgb_holdout', 'rgb_s', 'rgb_d',]
    # TAGS_TO_DISPLAY = ['Ground Truth', 'Render (NeRF)', 'Render (NeRF + D)']  # 标签显示名称

    # depth
    # TAGS_TO_VISUALIZE = ['rgb_holdout', 'depth_s', 'depth_d']
    # TAGS_TO_DISPLAY = ['Ground Truth', 'Depth (NeRF)', 'Depth (NeRF + D)']  # 标签显示名称

    # flow
    TAGS_TO_VISUALIZE = [ 'induced_flow_f', 'flow_f_gt', 'induced_flow_b', 'flow_b_gt', ]  # 需要可视化的图像标签
    TAGS_TO_DISPLAY = ['Estimated displacements\n(forward)', 'RAFT dense-flow\n(forward)', 'Estimated displacements\n(backward)', 'RAFT dense-flow\n(backward)']  # 标签显示名称d

    # 指定要可视化的步数列表(作为列)
    STEPS_TO_VISUALIZE = [173_000, 272_000, 394_500, 455_000, 500_000]  # 示例步数
    
    # 可视化并保存结果
    visualize_image_grid(
        log_path=LOG_PATH,
        tags_to_visualize=TAGS_TO_VISUALIZE,
        steps_to_visualize=STEPS_TO_VISUALIZE,
        TAGS_TO_DISPLAY=TAGS_TO_DISPLAY,
        save_path='tensorboard_image_grid.png' \
        '',
        
    )