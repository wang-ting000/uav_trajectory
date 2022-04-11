import numpy as np
import matplotlib.pyplot as plt


def line(p, t, B,ax):
    if len(p) > 1:
        new_p = np.zeros((len(p) - 1, 2))
        for i in range(len(p) - 1):
            new_p[i][0] = p[i][0] * (1 - t) + p[i + 1][0] * t
            new_p[i][1] = p[i][1] * (1 - t) + p[i + 1][1] * t
            #ax.scatter(new_p[i][0], new_p[i][1], new_p[i][2], marker='.', color='y')
            if len(p) == 2:
                ax.scatter(new_p[i][0], new_p[i][1], marker='^', color='g', label='point on Bezier curve')
                B.append([new_p[i][0], new_p[i][1]])
        p = new_p
        x = [p[i][0] for i in range(len(p))]
        y = [p[i][1] for i in range(len(p))]
        ax.plot(x, y, z, 'y')
        B_x = [B[i][0] for i in range(len(B))]
        B_y = [B[i][1] for i in range(len(B))]
        ax.plot(B_x, B_y, 'g')
        ax.legend(loc=1)
        line(p, t, B,ax)


def plot_Bezier(p, num):
    B = []
    gap = 0.001
    fig = plt.figure(1)
    ax = fig.add_subplot()
    # 设置坐标
    ax.set_xlabel('X(m)')
    ax.set_ylabel('Y(m)')
    ax.set_zlabel('Z(m)')
    for t in np.arange(0, 1, gap):
        for i in range(num + 1):
            if i == 0:
                ax.scatter(p[i][0], p[i][1], marker='+', s=80, color='blue', label='start points')

            elif i == 1:
                ax.scatter(p[i][0], p[i][1], marker='*', color='k', label='navigating points')
            elif i == num:
                ax.scatter(p[i][0], p[i][1], marker='o', color='red', alpha=0.3, label='end points')
            else:
                ax.scatter(p[i][0], p[i][1], marker='*', color='k')
        x = [p[i][0] for i in range(num + 1)]
        y = [p[i][1] for i in range(num + 1)]
        ax.plot(x, y, 'k', label='ori polygon')
        plt.legend(loc=1)
        line(p, t, B,ax)
        plt.pause(0.001)
        if t != 1 - gap:
            plt.cla()
        plt.title('Bezier curve')

        



