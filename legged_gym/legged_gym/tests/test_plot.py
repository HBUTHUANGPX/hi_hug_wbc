import pandas as pd
import matplotlib.pyplot as plt

def plot_selected_columns(csv_file):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)
    
    # 检查是否有至少6列
    if df.shape[1] < 6:
        raise ValueError("CSV 文件至少需要6列数据")
    
    # 提取所需的列
    col1, col2, col3, col4, col5, col6 = df.columns[:6]
    
    data1 = df[col1]
    data4 = df[col4]
    data2 = df[col2]
    data5 = df[col5]
    data3 = df[col3]
    data6 = df[col6]
    
    # 创建子图
    fig, axs = plt.subplots(3, 1, figsize=(15, 15))
    fig.suptitle('不同数据列的对比分析', fontsize=20)
    
    # 子图 1
    axs[0].plot(data1, marker='.', linestyle='-', color='blue', label=col1)
    axs[0].plot(data4, marker='.', linestyle='--', color='red', label=col4)
    axs[0].set_title(f'{col1} 和 {col4} 的对比')
    axs[0].set_xlabel('样本编号')
    axs[0].set_ylabel('值')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', linewidth=0.5)
    
    # 子图 2
    axs[1].plot(data2, marker='.', linestyle='-', color='green', label=col2)
    axs[1].plot(data5, marker='.', linestyle='--', color='orange', label=col5)
    axs[1].set_title(f'{col2} 和 {col5} 的对比')
    axs[1].set_xlabel('样本编号')
    axs[1].set_ylabel('值')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', linewidth=0.5)
    
    # 子图 3
    axs[2].plot(data3, marker='.', linestyle='-', color='purple', label=col3)
    axs[2].plot(data6, marker='.', linestyle='--', color='brown', label=col6)
    axs[2].set_title(f'{col3} 和 {col6} 的对比')
    axs[2].set_xlabel('样本编号')
    axs[2].set_ylabel('值')
    axs[2].legend()
    axs[2].grid(True, linestyle='--', linewidth=0.5)
    
    # 调整子图间距
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存图形
    plt.savefig('comparison_plots_optimized.png')
    
    # 显示图形
    plt.show()

if __name__ == "__main__":
    # plot_selected_columns('/home/pi/Documents/data/data_1.csv')
    # plot_selected_columns('/home/pi/Documents/data/data_2.csv')
    # plot_selected_columns('/home/pi/Documents/data/data_1.csv')
    # plot_selected_columns('/home/pi/Documents/data/data_1.csv')


    # plot_selected_columns('/home/pi/Documents/data/data_2_1.csv')
    # plot_selected_columns('/home/pi/Documents/data/data_2_2.csv')
    plot_selected_columns('/home/pi/Documents/data/data_2_3.csv')


