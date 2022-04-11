import numpy as np
import matplotlib.pyplot as plt
import math


def comb(n, m):
    return (math.factorial(n) / (math.factorial(m) * math.factorial(n - m)))


def plot_Bezier(p, num):
    for i in range(num + 1):
        if i == 0:
            plt.scatter(p[i][0], p[i][1], marker='*', color='k', label='navigating points')
        plt.scatter(p[i][0], p[i][1], marker='*', color='k')
    plt.legend()
    x = [p[i][0] for i in range(num + 1)]
    y = [p[i][1] for i in range(num + 1)]
    plt.plot(x, y, label='ori polygen')
    for t in np.arange(0, 1, 0.01):
        B = [0, 0]
        for i in range(num + 1):
            B[0] += comb(num, i) * p[i][0] * ((1 - t) ** (num - i)) * (t ** i)
            B[1] += comb(num, i) * p[i][1] * ((1 - t) ** (num - i)) * (t ** i)
        if t == 0:
            plt.scatter(B[0], B[1], color='g', marker='.', label='Bezier curve')
        plt.scatter(B[0], B[1], color='g', marker='.', )
        plt.pause(0.0001)
        plt.xlabel('x/m')
        plt.ylabel('y/m')
        plt.title('Bezier curve')
        plt.legend(loc=1)


def cal_Bezier_mid(p, num):
    # num是原始数组的长度-1
    gap = 100
    B = np.zeros((gap * (num + 1), 2))
    st = 0
    for t in np.arange(0, 1, 1 / gap):
        # 第一段是一阶贝塞尔曲线
        nn = 1
        for i in range(nn + 1):
            B[0 + st][0] += comb(nn, i) * p[i][0] * ((1 - t) ** (nn - i)) * (t ** i)
            B[0 + st][1] += comb(nn, i) * p[i][1] * ((1 - t) ** (nn - i)) * (t ** i)
        # 中间的若干段是三阶贝塞尔曲线
        nn = 3
        for m in range(1, num):
            for i in range(nn + 1):
                B[m * gap + st][0] += comb(nn, i) * p[1 + (m - 1) * nn + i][0] * ((1 - t) ** (nn - i)) * (t ** i)
                B[m * gap + st][1] += comb(nn, i) * p[1 + (m - 1) * nn + i][1] * ((1 - t) ** (nn - i)) * (t ** i)
        # 最后一段是二阶贝塞尔曲线
        nn = 2
        for i in range(nn + 1):
            B[num * gap + st][0] += comb(nn, i) * p[1 + (num - 1) * 3 + i][0] * ((1 - t) ** (nn - i)) * (t ** i)
            B[num * gap + st][1] += comb(nn, i) * p[1 + (num - 1) * 3 + i][1] * ((1 - t) ** (nn - i)) * (t ** i)
        st += 1
    return B


def cal_Bezier(p, num):
    gap = 1000
    B = np.zeros((gap, 2))
    st = 0
    for t in np.arange(0, 1, 1 / gap):
        # num阶贝塞尔曲线
        for i in range(num + 1):
            B[st][0] += comb(num, i) * p[i][0] * ((1 - t) ** (num - i)) * (t ** i)
            B[st][1] += comb(num, i) * p[i][1] * ((1 - t) ** (num - i)) * (t ** i)
        st += 1
    return B




def gen_midpoints(num,M,p,q1,q2):
    x = []
    y = []
    for k in range(num):
        M[k * 2 + 0][0] = q2 * p[k][0] + q1 * p[k + 1][0]
        M[k * 2 + 0][1] = q2 * p[k][1] + q1 * p[k + 1][1]
        M[k * 2 + 1][0] = q1 * p[k][0] + q2 * p[k + 1][0]
        M[k * 2 + 1][1] = q1 * p[k][1] + q2 * p[k + 1][1]
        x.append(p[k][0])
        x.append(M[k * 2][0])
        x.append(M[k * 2 + 1][0])
        y.append(p[k][1])
        y.append(M[k * 2][1])
        y.append(M[k * 2 + 1][1])
    x.append(p[num][0])
    y.append(p[num][1])
    new_l = np.zeros((2, 3 * num + 1))
    new_l[0] = x
    new_l[1] = y
    return new_l

# 航点
p = [[0, 0], [50, 30], [40, 35], [60, -10], [10, 0],[0,0]]
num = len(p) - 1
M = np.zeros((2 * num, 2))

# 画图
old_x = [p[i][0] for i in range(len(p))]
old_y = [p[i][1] for i in range(len(p))]
plt.scatter(old_x, old_y, marker='*', s=80, color='k', label='navigating points')
# 拟合
B = cal_Bezier(p, num)
B_x = [B[i][0] for i in range(len(B))]
B_y = [B[i][1] for i in range(len(B))]
plt.plot(B_x, B_y, 'pink', label='Bezier curve of order %d' % num)
q1 = 0.25
q2 = 1-q1
#q1<q2
new_l = gen_midpoints(num,M,p,q1,q2)
plt.plot(new_l[0],new_l[1],'k',label = 'polygon')
B = cal_Bezier_mid(new_l.T, num)
B_x = [B[i][0] for i in range(len(B))]
B_y = [B[i][1] for i in range(len(B))]
plt.plot(B_x, B_y, 'g', label='Bezier curve with midpoints')

plt.legend(loc=1)
plt.xlabel('x/m')
plt.ylabel('y/m')
plt.title('Bezier curve with midpoints')
