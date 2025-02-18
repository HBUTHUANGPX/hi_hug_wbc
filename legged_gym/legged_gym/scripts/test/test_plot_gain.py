import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/home/pi/HPX_Loco/DreamWaQ-Pi-Sim2sim/kpf_kdf_data.csv', header=None)  

# 或者使用 pandas 自带的 plot
# l = [0,6]
# df.iloc[:500, l].plot()   # 绘制前 6 列
# plt.title("vel hip pitch 0/6 Data")
# l = [1,7]
# df.iloc[:500, l].plot()   # 绘制前 6 列
# plt.title("vel hip roll 1/7 Data")
# l = [2,8]
# df.iloc[:500, l].plot()   # 绘制前 6 列
# plt.title("vel thigh 2/8 Data")
l = [3,9]
df.iloc[2000:3000, l].plot()   # 绘制前 6 列
plt.title("vel calf 3/9 Data")
l = [4,10]
df.iloc[2000:3000, l].plot()   # 绘制前 6 列
plt.title("vel ankle pitch 4/10 Data")

# l = [3+12,9+12]
# df.iloc[2000:2500, l].plot()   # 绘制前 6 列
# plt.title("vel calf 3/9 Data")
# l = [4+12,10+12]
# df.iloc[2000:2500, l].plot()   # 绘制前 6 列
# plt.title("vel ankle pitch 4/10 Data")

l = [3+24,9+24]
df.iloc[2000:3000, l].plot()   # 绘制前 6 列
plt.title("vel calf 3/9 Data")
l = [4+24,10+24]
df.iloc[2000:3000, l].plot()   # 绘制前 6 列
plt.title("vel ankle pitch 4/10 Data")


# l = [5,11]
# df.iloc[:500, l].plot()   # 绘制前 6 列
# plt.title("vel ankle roll 5/11 Data")
plt.show()
