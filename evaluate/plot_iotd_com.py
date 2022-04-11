import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

payload = np.load(r'C:\Users\Administrator\Desktop\uav_tra-master\payload.npy')
payload = payload.T
for i in range(15):
    plt.plot(payload[:][i], label='iotd%s' % i,marker='*')
plt.title('iotd 通信情况')
plt.legend()
