import numpy as np
import matplotlib.pyplot as plt
import math


def comb(n, m):
    return (math.factorial(n) / (math.factorial(m) * math.factorial(n - m)))




def cal_Bezier_mid(p, num):
    # num是原始数组的长度-1
    gap = 100
    B = np.zeros((gap * (num + 1), 3))
    st = 0
    for t in np.arange(0, 1, 1 / gap):
        # 第一段是一阶贝塞尔曲线
        nn = 1
        for i in range(nn + 1):
            B[0 + st][0] += comb(nn, i) * p[i][0] * ((1 - t) ** (nn - i)) * (t ** i)
            B[0 + st][1] += comb(nn, i) * p[i][1] * ((1 - t) ** (nn - i)) * (t ** i)
            B[0 + st][2] += comb(nn, i) * p[i][2] * ((1 - t) ** (nn - i)) * (t ** i)
        # 中间的若干段是三阶贝塞尔曲线
        nn = 3
        for m in range(1, num):
            for i in range(nn + 1):
                B[m * gap + st][0] += comb(nn, i) * p[1 + (m - 1) * nn + i][0] * ((1 - t) ** (nn - i)) * (t ** i)
                B[m * gap + st][1] += comb(nn, i) * p[1 + (m - 1) * nn + i][1] * ((1 - t) ** (nn - i)) * (t ** i)
                B[m * gap + st][2] += comb(nn, i) * p[1 + (m - 1) * nn + i][2] * ((1 - t) ** (nn - i)) * (t ** i)
        # 最后一段是二阶贝塞尔曲线
        nn = 2
        for i in range(nn + 1):
            B[num * gap + st][0] += comb(nn, i) * p[1 + (num - 1) * 3 + i][0] * ((1 - t) ** (nn - i)) * (t ** i)
            B[num * gap + st][1] += comb(nn, i) * p[1 + (num - 1) * 3 + i][1] * ((1 - t) ** (nn - i)) * (t ** i)
            B[num * gap + st][2] += comb(nn, i) * p[1 + (num - 1) * 3 + i][2] * ((1 - t) ** (nn - i)) * (t ** i)
        st += 1
    return B


def cal_Bezier(p, num):
    gap = 1000
    B = np.zeros((gap, 3))
    st = 0
    for t in np.arange(0, 1, 1 / gap):
        # num阶贝塞尔曲线
        for i in range(num + 1):
            B[st][0] += comb(num, i) * p[i][0] * ((1 - t) ** (num - i)) * (t ** i)
            B[st][1] += comb(num, i) * p[i][1] * ((1 - t) ** (num - i)) * (t ** i)
            B[st][2] += comb(num, i) * p[i][2] * ((1 - t) ** (num - i)) * (t ** i)
        st += 1
    return B




def gen_midpoints(num,M,p,q1,q2):
    x = []
    y = []
    z = []
    for k in range(num):
        M[k * 2 + 0][0] = q2 * p[k][0] + q1 * p[k + 1][0]
        M[k * 2 + 0][1] = q2 * p[k][1] + q1 * p[k + 1][1]
        M[k * 2 + 0][2] = q2 * p[k][2] + q1 * p[k + 1][2]
        M[k * 2 + 1][0] = q1 * p[k][0] + q2 * p[k + 1][0]
        M[k * 2 + 1][1] = q1 * p[k][1] + q2 * p[k + 1][1]
        M[k * 2 + 1][2] = q1 * p[k][2] + q2 * p[k + 1][2]
        x.append(p[k][0])
        x.append(M[k * 2][0])
        x.append(M[k * 2 + 1][0])
        y.append(p[k][1])
        y.append(M[k * 2][1])
        y.append(M[k * 2 + 1][1])
        z.append(p[k][2])
        z.append(M[k * 2][2])
        z.append(M[k * 2 + 1][2])
    x.append(p[num][0])
    y.append(p[num][1])
    z.append(p[num][2])
    new_l = np.zeros((3, 3 * num + 1))
    new_l[0] = x
    new_l[1] = y
    new_l[2] = z
    return new_l


