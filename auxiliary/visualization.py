import _thread as thread
import visdom
import os
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Visualizer(object):
    def __init__(self, port, env):
        super(Visualizer, self).__init__()
        thread.start_new_thread(os.system, (f"visdom -p {port} > /dev/null 2>&1",))
        vis = visdom.Visdom(port=port, env=env)
        self.vis = vis

    def show_pointclouds(self, points, title=None, Y=None):
        points = points.squeeze()
        if points.size(-1) == 3:
            points = points.contiguous().data.cpu()
        else:
            points = points.transpose(0, 1).contiguous().data.cpu()

        if Y is None:
            self.vis.scatter(X=points, win=title, opts=dict(title=title, markersize=2))
        else:
            if Y.min() < 1:
                Y = Y - Y.min() + 1
            self.vis.scatter(
                X=points, Y=Y, win=title, opts=dict(title=title, markersize=2)
            )

def visualize_points(points, bound=0.5, c=None, out_file=None, show=False):
    fig = plt.figure()
    
    ax = fig.add_subplot(111, projection='3d')

    if len(points) != 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=c)

    ax.set_xlim3d(-bound, bound)
    ax.set_ylim3d(-bound, bound)
    ax.set_zlim3d(-bound, bound)
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.view_init(elev=30, azim=45)

    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()

    plt.close(fig)
