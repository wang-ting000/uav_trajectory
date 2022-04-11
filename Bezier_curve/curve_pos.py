import numpy as np
import matplotlib.pyplot as plt

from bezier_midpoints_3d import cal_Bezier, cal_Bezier_mid,gen_midpoints

p = np.load(r'C:\Users\Administrator\Desktop\uav_tra-master\pos.npy')
num = len(p) - 1

M = np.zeros((2 * num, 3))

# 画图
old_x = [p[i][0] for i in range(len(p))]
old_y = [p[i][1] for i in range(len(p))]
old_z = [p[i][2] for i in range(len(p))]
fig = plt.figure(1)
ax = fig.add_subplot(111, projection="3d")
ax.scatter(old_x, old_y, old_z, marker='*', s=80, color='k', label='navigating points')
# 拟合
B = cal_Bezier(p, num)
B_x = [B[i][0] for i in range(len(B))]
B_y = [B[i][1] for i in range(len(B))]
B_z = [B[i][2] for i in range(len(B))]
plt.plot(B_x, B_y,B_z, 'pink', label='Bezier curve of order %d' % num)
q1 = 0.25
q2 = 1 - q1
# q1<q2
new_l = gen_midpoints(num, M, p, q1, q2)
plt.plot(new_l[0], new_l[1],new_l[2], 'k', label='polygon')
B = cal_Bezier_mid(new_l.T, num)
B_x = [B[i][0] for i in range(len(B))]
B_y = [B[i][1] for i in range(len(B))]
B_z = [B[i][2] for i in range(len(B))]
plt.plot(B_x, B_y, B_z,'g', label='Bezier curve with midpoints')

plt.legend()

plt.title('Bezier curve with midpoints')
