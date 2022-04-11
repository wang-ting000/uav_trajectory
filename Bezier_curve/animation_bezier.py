import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

figure = plt.figure()
ax = figure.add_subplot()


def line(p, t, B, fig):
    if len(p) > 1:
        new_p = np.zeros((len(p) - 1, 2))
        for i in range(len(p) - 1):
            new_p[i][0] = p[i][0] * (1 - t) + p[i + 1][0] * t
            new_p[i][1] = p[i][1] * (1 - t) + p[i + 1][1] * t
            im = plt.scatter(new_p[i][0], new_p[i][1], marker='.', color='y')
            fig.append(im)
            if len(p) == 2:
                im = plt.scatter(new_p[i][0], new_p[i][1], marker='^', color='g', label='point on Bezier curve')
                fig.append(im)
                B.append([new_p[i][0], new_p[i][1]])
        p = new_p
        x = [p[i][0] for i in range(len(p))]
        y = [p[i][1] for i in range(len(p))]
        im, = plt.plot(x, y, 'y')
        fig.append(im)
        B_x = [B[i][0] for i in range(len(B))]
        B_y = [B[i][1] for i in range(len(B))]
        im, = plt.plot(B_x, B_y, 'g')
        fig.append(im)
        line(p, t, B, fig)
        return fig


def plot_Bezier(t, p, num, B):
    fig = []
    for i in range(num + 1):
        if i == 0:
            im = plt.scatter(p[i][0], p[i][1], marker='+', s=80, color='blue', label='start points')
            fig.append(im)

        elif i == 1:
            im = plt.scatter(p[i][0], p[i][1], marker='*', color='k', label='navigating points')
            fig.append(im)
        elif i == num:
            im = plt.scatter(p[i][0], p[i][1], marker='o', color='red', alpha=0.3, label='end points')
            fig.append(im)
        else:
            im = plt.scatter(p[i][0], p[i][1], marker='*', color='k')
            fig.append(im)
    x = [p[i][0] for i in range(num + 1)]
    y = [p[i][1] for i in range(num + 1)]
    im, = plt.plot(x, y, 'k', label='ori polygon')
    fig.append(im)
    # label = ax.text(0, 0, '*--ori polygon', ha='right', va='top', fontsize=8, color="k")
    # fig.append(label)
    im.set_label('ori polygon')
    fig_b = line(p, t, B, fig)
    fig.extend(fig_b)
    if t == 0:
        legs = ax.legend()
    # legs.texts[0].set_text('hwi')
    fig.append(plt.xlabel('x/m'))
    fig.append(plt.ylabel('y/m'))
    fig.append(plt.title('Bezier curve'))

    return fig


ims = []
p = [[0, 0], [-10, 30], [40, 35], [60, -10], [0, 0]]
num = len(p) - 1
B = []
gap = 0.005
for t in np.arange(0, 1, gap):
    figs = plot_Bezier(t, p, num, B)
    ims.append(figs)
ani = animation.ArtistAnimation(figure, ims, interval=1,blit=False)

ani.save("Bezier.gif",writer='pillow')
ani.save("Bezier.mp4", writer='ffmpeg', fps=5,dpi=100)
plt.show()
