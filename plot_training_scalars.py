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

def plot_scalar_values(log_path, tags_to_visualize, tags_to_display, title, xlabel="Step", alpha = 1):
    """
    可视化多个 scalar 标量在 TensorBoard 中的演化曲线

    Args:
        log_path: TensorBoard 日志文件路径
        tags_to_visualize: 需要可视化的 scalar 标签列表
        tags_to_display: 标签的展示名称（与 tags_to_visualize 对应）
    
    Returns:
        plt.Axes: 绘图的 Axes 对象
    """
    print(f"Loading logs: {log_path}")
    ea = event_accumulator.EventAccumulator(
        log_path,
        size_guidance={event_accumulator.SCALARS: 0}  # 加载所有 scalar 数据
    )
    ea.Reload()

    available_scalar_tags = ea.Tags()['scalars']
    print(f"Available scalar tags ({len(available_scalar_tags)}):")
    for tag in available_scalar_tags:
        print(f"  - {tag}")

    missing_tags = [tag for tag in tags_to_visualize if tag not in available_scalar_tags]
    if missing_tags:
        print(f"Warning: Tags not found: {missing_tags}")
        tags_to_visualize = [tag for tag in tags_to_visualize if tag in available_scalar_tags]
        if not tags_to_visualize:
            print("No valid scalar tags to visualize!")
            return

    # fig, ax = plt.subplots(figsize=(15, 6))
    ax = plt.gca()  # 获取当前的 Axes 对象
    if title is not None: ax.set_title(title, fontsize=16)
    if xlabel is not None: ax.set_xlabel("Step", fontsize=14)
    
    ax.grid(True)

    i=1
    step_min = 0
    step_max = 500_000
    for tag, display_name in zip(tags_to_visualize, tags_to_display):
        scalar_events = ea.Scalars(tag)
        steps = [e.step for e in scalar_events]
        values = [e.value for e in scalar_events]
        # if len(tags_to_visualize)==1: ax.plot(steps, values, 'k', label=display_name, alpha=1 - (1-0.5)*(i-1)/len(tags_to_visualize))
        ax.plot(steps, values, label=display_name, alpha=alpha)

        # step_min = max(step_min, min(steps))
        # step_max = min(step_max, max(steps))
        # ax.set_xlim(step_min, step_max)
        i+=1
    ax.legend(fontsize=12)
    # else: ax.set_ylabel("Value", fontsize=14)
    
    plt.tight_layout()
    return ax

def plot_two_groups_of_scalars(
    log_path,
    tags_to_visualize_group1,
    tags_to_display_group1,
    title_group1,
    tags_to_visualize_group2,
    tags_to_display_group2,
    title_group2,
    save_path='scalar_plot.svg'
):
    """
    输入两组 scalar 标签，分别在2行1列的子图中可视化

    Args:
        log_path: TensorBoard 日志文件路径
        tags_to_visualize_group1: 第一组 scalar 标签列表
        tags_to_display_group1: 第一组标签的展示名称
        title_group1: 第一组子图标题
        tags_to_visualize_group2: 第二组 scalar 标签列表
        tags_to_display_group2: 第二组标签的展示名称
        title_group2: 第二组子图标题
        save_path: 保存路径
    """
    plt.figure(figsize=(15, 10))

    plt.subplot(211)
    plot_scalar_values(
        log_path=log_path,
        tags_to_visualize=tags_to_visualize_group1,
        tags_to_display=tags_to_display_group1,
        title=title_group1,
        xlabel=None
    )

    plt.subplot(212)
    plot_scalar_values(
        log_path=log_path,
        tags_to_visualize=tags_to_visualize_group2,
        tags_to_display=tags_to_display_group2,
        title=title_group2,
        xlabel="Step"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Scalar plot saved as {save_path}")
    plt.show()

def plot_three_groups_of_scalars(
    log_path,
    tags_to_visualize_group1,
    tags_to_display_group1,
    title_group1,
    tags_to_visualize_group2,
    tags_to_display_group2,
    title_group2,
    tags_to_visualize_group3,
    tags_to_display_group3,
    title_group3,
    save_path='scalar_plot.svg'
):
    """
    输入两组 scalar 标签，分别在2行1列的子图中可视化

    Args:
        log_path: TensorBoard 日志文件路径
        tags_to_visualize_group1: 第一组 scalar 标签列表
        tags_to_display_group1: 第一组标签的展示名称
        title_group1: 第一组子图标题
        tags_to_visualize_group2: 第二组 scalar 标签列表
        tags_to_display_group2: 第二组标签的展示名称
        title_group2: 第二组子图标题
        save_path: 保存路径
    """
    plt.figure(figsize=(15, 15))

    plt.subplot(311)
    plot_scalar_values(
        log_path=log_path,
        tags_to_visualize=tags_to_visualize_group1,
        tags_to_display=tags_to_display_group1,
        title=title_group1,
        xlabel=None
    )
    # plt.yscale('log')

    plt.subplot(312)
    plot_scalar_values(
        log_path=log_path,
        tags_to_visualize=tags_to_visualize_group2,
        tags_to_display=tags_to_display_group2,
        title=title_group2,
        xlabel="Step"
    )
    # plt.yscale('log')

    plt.subplot(313)
    plot_scalar_values(
        log_path=log_path,
        tags_to_visualize=tags_to_visualize_group3,
        tags_to_display=tags_to_display_group3,
        title=title_group3,
        xlabel="Step"
    )
    # plt.yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Scalar plot saved as {save_path}")
    plt.show()

def compare_different_scenes(
    log_path1, 
    log_path2, 
    tags_to_visualize,
    tags_to_display1,
    tags_to_display2,
    save_path='compare_scalars',
    titles = None,
):
    plt.figure(figsize=(len(tags_to_visualize)*5, 5))
    
    for i,tag in enumerate(tags_to_visualize):
        plt.subplot(1,len(tags_to_visualize),1+i)
      

        plot_scalar_values(
            log_path=log_path2,
            tags_to_visualize=[tags_to_visualize[i]],
            tags_to_display=[tags_to_display2[i]],
            title=titles[i],
            xlabel="Step"
        )

        plot_scalar_values(
            log_path=log_path1,
            tags_to_visualize=[tags_to_visualize[i]],
            tags_to_display=[tags_to_display1[i]],
            title=None,
            xlabel="Step"
        )
        if "loss" in tags_to_visualize[i]: plt.yscale('log')
        plt.xlim(131_000, 500_000)
    
    plt.tight_layout()
    plt.savefig(save_path+'.svg', dpi=300, bbox_inches='tight')
    print(f"Scalar plot saved as {save_path}")
    plt.show()

    pass

if __name__ == "__main__":
    # 指定TensorBoard日志路径
    # LOG_PATH = 'logs/summaries/DJI_20250516151729_0005_V-100f-fps30_DyNeRF_pretrain_test/events.out.tfevents.1748093593.ZhizhangOFFICE.2938.0'
    
    # render
    # TAGS_TO_VISUALIZE = ['rgb_holdout', 'rgb_s', 'rgb_d',]
    # TAGS_TO_DISPLAY = ['Ground Truth', 'Render (NeRF)', 'Render (NeRF + D)']  # 标签显示名称

    # depth
    # TAGS_TO_VISUALIZE = ['rgb_holdout', 'depth_s', 'depth_d']
    # TAGS_TO_DISPLAY = ['Ground Truth', 'Depth (NeRF)', 'Depth (NeRF + D)']  # 标签显示名称
    

    # flow
    # TAGS_TO_VISUALIZE = ['flow_f_gt', 'induced_flow_f', 'flow_b_gt', 'induced_flow_b']  # 需要可视化的图像标签
    # TAGS_TO_DISPLAY = ['Estimated displacements\n(forward)', 'RAFT dense-flow\n(forward)', 'Estimated displacements\n(backward)', 'RAFT dense-flow\n(backward)']  # 标签显示名称d


    # scalar: depth loss
    # TAGS_TO_VISUALIZE = ['depth_loss']
    # TAGS_TO_DISPLAY = ['Pseudo-depth loss']  # 标签显示名称
    # title = "Pseudo-depth loss"


    # scalar: psnr_sd
    # TAGS_TO_VISUALIZE = ['psnr_s', 'psnr_d']
    # TAGS_TO_DISPLAY = ['PSNR (NeRF)', 'PSNR (NeRF + D)']  # 标签显示名称

    # scalar: motion loss 2 steps
    # motion_loss_2_step()
    # TAGS_TO_VISUALIZE = ['img_d_f_loss', 'img_d_f_f_loss'] + ['img_d_b_loss', 'img_d_b_b_loss']
    # TAGS_TO_DISPLAY = ['Forward 1 step motion', 'Forward 2 step motion'] + ['Backward 1 step motion', 'Backward 2 step motion'] # 标签显示名称    

    # scalar: psnr motion
    # TAGS_TO_VISUALIZE = ['psnr_d_f', 'psnr_d_b']
    # TAGS_TO_DISPLAY = ['forward motion (1 step)', 'backward motion (1 step)']  # 标签显示名称

    # scalar: pseudo-motion loss
    # TAGS_TO_VISUALIZE = ['flow_f_loss', 'flow_b_loss']
    # TAGS_TO_DISPLAY = ['Forward $L_1$ loss', 'Backward $L_1$ loss']  # 标签显示名称
    # title = "RAFT influence during 3D-motion-optimization"  # 图表标题

    # TAGS_TO_VISUALIZE = ['sf_smooth_loss', 'sp_smooth_loss']
    # TAGS_TO_DISPLAY = ['Smooth loss (time-domain)', 'Smooth loss (spatial-domain)']  # 标签显示名称
    # title = "RAFT influence during 3D-motion-optimization"  # 图表标题

    # plt.figure(figsize=(15, 6))
    # ax = plot_scalar_values(LOG_PATH, 
    #                    TAGS_TO_VISUALIZE, 
    #                    TAGS_TO_DISPLAY, 
    #                    title=title,
    #                    xlabel="Step")
    # # plt.xlim(210_000, 490_000)
    # plt.tight_layout()
    # plt.savefig('scalar_plot.svg', dpi=300, bbox_inches='tight')
    # print("Scalar plot saved as scalar_plot")    
    # plot_two_groups_of_scalars(
    #     log_path=LOG_PATH,
    #     tags_to_visualize_group1=['img_d_f_loss','img_d_b_loss'] + [ 'img_d_f_f_loss', 'img_d_b_b_loss'],
    #     tags_to_display_group1=['Forward 1 step motion', 'Backward 1 step motion' ] + ['Forward 2 step motion', 'Backward 2 step motion'],
    #     title_group1="Loss of 3D motion estimation (color MSE)",
    #     tags_to_visualize_group2=['psnr_d_f', 'psnr_d_b'],
    #     tags_to_display_group2=['forward motion (1 step)', 'backward motion (1 step)'],
    #     title_group2="Accuracy of 3D motion estimation (PSNR by motion)",
    #     save_path='motion_and_psnr_plot.svg'
    # )

    # plot_three_groups_of_scalars(
    #     log_path=LOG_PATH,
    #     tags_to_visualize_group1=['sp_smooth_loss'],
    #     tags_to_display_group1=['Motion-smoothness loss (spatial-domain)'],
    #     title_group1="Motion-smoothness loss (spatial-domain)",
    #     tags_to_visualize_group2=['smooth_loss'],
    #     tags_to_display_group2=['Motion-smoothness loss (time-domain)'],
    #     title_group2='Motion-smoothness loss (time-domain)',
    #     tags_to_visualize_group3=['consistency_loss'],
    #     tags_to_display_group3=['Motion-consistency loss (time-domain)'],
    #     title_group3='Motion-consistency loss (time-domain)',
    #     save_path='3_scalar_plot.svg'
    # )

    TAGS_TO_VISUALIZE = [ 'psnr_d', "smooth_loss", "consistency_loss"]  # 需要可视化的图像标签
    TAGS_TO_DISPLAY1 = [ 'mountain' , "mountain", "mountain"]  
    TAGS_TO_DISPLAY2 = [ 'plain' , "plain", "plain"]
    titles = ["PSNR", "Motion smoothness loss", "Motion consistency loss"]

    compare_different_scenes(
            log_path1=r"logs/summaries/inservice-wind-turbine/events.out.tfevents.1752078591.ZhizhangOFFICE.123140.0",
            log_path2=r"logs/summaries/plain-inservice/events.out.tfevents.1752147392.ZhizhangOFFICE.49301.0",
            tags_to_visualize= TAGS_TO_VISUALIZE,
            tags_to_display1 = TAGS_TO_DISPLAY1,
            tags_to_display2 = TAGS_TO_DISPLAY2,
            titles = titles
    )