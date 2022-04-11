import numpy as np
from uav_gym.envs.uav_env import UavConfig
import matplotlib.pyplot as plt
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.cbook import get_sample_data



def myplot(pos, top_config, t, iotd_pos, payload, file_path, fire_pos):
    r = 50
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    fire_x = np.zeros((top_config.fire_num, 100, 100))
    fire_y = np.zeros((top_config.fire_num, 100, 100))
    fire_z = np.zeros((top_config.fire_num, 100, 100))
    for iii in range(top_config.fire_num):
        fire_x[iii] = r * np.outer(np.cos(u), np.sin(v)) + fire_pos[iii][0]
        fire_y[iii] = r * np.outer(np.sin(u), np.sin(v)) + fire_pos[iii][1]
        fire_z[iii] = r * np.outer(np.ones(np.size(u)), np.cos(v)) + fire_pos[iii][2]
    fig = plt.figure(1)
    plt.get_current_fig_manager().full_screen_toggle()  # toggle fullscreen mode
    plt.ion()
    # 设置坐标范围
    for j in range(t):
        # 设定标题等
        fig.suptitle("The trajectory of UAV-BS")
        ax = fig.add_subplot(121, projection="3d")
        # 设置坐标
        ax.view_init(0, 45)
        ax.set_xlabel('X(m)')
        ax.set_ylabel('Y(m)')
        ax.set_zlabel('Z(m)')
        # 先画用户
        # uav初始位置
        for x in range(top_config.fire_num):
            ax.plot_surface(fire_x[x], fire_y[x], fire_z[x], rstride=1, cstride=1, color='r', alpha=1)
        ax.scatter(top_config.UAV_INITIAL_POSITION_X, top_config.UAV_INITIAL_POSITION_Y,
                   top_config.UAV_INITIAL_POSITION_Z, c='r', marker='x',
                   s=30)
        ax.set_xlim(0, top_config.total_x)
        ax.set_ylim(0, top_config.total_y)
        ax.set_zlim(0, top_config.h_max)

        for times in range(j):
            ax.scatter(pos[times][0], pos[times][1], pos[times][2], c='k', marker=r'$\bigodot$', s=25)
        ax.scatter(pos[j][0], pos[j][1], pos[j][2], c='k', marker='x', s=25, label='uav')
        X = np.asarray([iotd_pos[i][0] for i in range(top_config.iotd_num)])
        Y = np.asarray([iotd_pos[i][1] for i in range(top_config.iotd_num)])
        Z = np.asarray([iotd_pos[i][2] for i in range(top_config.iotd_num)])
        p = ax.scatter(X, Y, Z, edgecolors='k', linewidths=1, c=payload[j], vmin=0, vmax=1, cmap='Greens_r', alpha=0.66,
                       s=50,
                       marker='o', label='IoTDs')
        plt.colorbar(p)
        plt.title(u'flight time the %s th seconds' % (j + 1))
        plt.legend()

        ax = fig.add_subplot(122, projection="3d")
        ax.view_init(90, 0)
        # 设置坐标
        # ax.view_init(45, 0)
        ax.set_xlabel('X(m)')
        ax.set_ylabel('Y(m)')
        ax.set_zlabel('Z(m)')
        # 先画用户
        # uav初始位置
        for x in range(top_config.fire_num):
            ax.plot_surface(fire_x[x], fire_y[x], fire_z[x], rstride=1, cstride=1, color='r', alpha=1)
        ax.scatter(top_config.UAV_INITIAL_POSITION_X, top_config.UAV_INITIAL_POSITION_Y,
                   top_config.UAV_INITIAL_POSITION_Z, c='r', marker='x',
                   s=30)
        ax.set_xlim(0, top_config.total_x)
        ax.set_ylim(0, top_config.total_y)
        ax.set_zlim(0, top_config.h_max)

        for times in range(j):
            ax.scatter(pos[times][0], pos[times][1], pos[times][2], c='k', marker=r'$\bigodot$', s=25)
        ax.scatter(pos[j][0], pos[j][1], pos[j][2], c='k', marker='x', s=25, label='uav')
        X = np.asarray([iotd_pos[i][0] for i in range(top_config.iotd_num)])
        Y = np.asarray([iotd_pos[i][1] for i in range(top_config.iotd_num)])
        Z = np.asarray([iotd_pos[i][2] for i in range(top_config.iotd_num)])
        p = ax.scatter(X, Y, Z, edgecolors='k', linewidths=1, c=payload[j], vmin=0, vmax=1, cmap='Greens_r', alpha=0.66,
                       s=50,
                       marker='o', label='IoTDs')
        plt.colorbar(p)
        plt.title(u'flight time the %s th seconds' % (j + 1))
        plt.legend()
        plt.pause(0.2)
        if j != t - 1:
            plt.cla()
            plt.clf()
    print('END')
    plt.ioff()
    foo_fig = plt.gcf()
    fig_name_png = file_path + "/figure_1.png"
    foo_fig.savefig(fig_name_png)

    plt.show()


def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


def main():
    topConfig = UavConfig(train_flag=2)#不保存数据
    # 飞行时间最短的一次路径文件
    date = '20220406/00/'
    file_path = 'user/uav/%s/data/user%d_dis%d/evaluate_log_file' %(date,topConfig.iotd_num,topConfig.total_x)
    file = file_path + '/evaluate_file.txt'

    f = open(file)
    line = f.readline()
    data_list = []
    # 将txt转化为数组
    while line:
        num = str(list(map(float, line.split()))).strip('[').strip(']')
        num = float(num)
        data_list.append(num)
        line = f.readline()
    f.close()
    # 从数组中提取出每个时刻无人机的位置
    length = np.shape(data_list)[0]
    t = int((length - 1) / topConfig.STATE_DIM)
    pos = np.zeros((t + 1, 3))
    iotd_pos = np.zeros((topConfig.iotd_num, 3))
    payload = np.zeros((t + 1, topConfig.iotd_num))
    fire_pos = np.zeros((topConfig.fire_num, 3))
    for k in range(t + 1):
        pos[k][0] = data_list[k * topConfig.STATE_DIM + 0] * topConfig.total_x
        pos[k][1] = data_list[k * topConfig.STATE_DIM + 1] * topConfig.total_y
        pos[k][2] = data_list[k * topConfig.STATE_DIM + 2] * topConfig.h_max
        for kk in range(topConfig.iotd_num):
            payload[k][kk] = data_list[k * topConfig.STATE_DIM + 6 + kk * 8 + 7]
            #iotd_com[k][kk] = data_list
    for number in range(topConfig.iotd_num):
        iotd_pos[number][0] = data_list[6 + number * 8 + 0] * topConfig.total_x
        iotd_pos[number][1] = data_list[6 + number * 8 + 1] * topConfig.total_y
        iotd_pos[number][2] = data_list[6 + number * 8 + 2] * topConfig.IOTD_Z
    for ii in range(topConfig.fire_num):
        fire_pos[ii][0] = data_list[6 + topConfig.iotd_num * 8 + ii * 4 + 0] * topConfig.total_x
        fire_pos[ii][1] = data_list[6 + topConfig.iotd_num * 8 + ii * 4 + 1] * topConfig.total_y
        fire_pos[ii][2] = data_list[6 + topConfig.iotd_num * 8 + ii * 4 + 2] * topConfig.h_max

    plot_path = 'plot'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    np.save('plot/payload',payload)
    np.save('plot/iotd_pos',iotd_pos)
    np.save('plot/pos',pos)
    myplot(pos, topConfig, t, iotd_pos, payload, file_path, fire_pos)
    print(fire_pos)


if __name__ == '__main__':
    main()
