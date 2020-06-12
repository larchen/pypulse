# Written by Akel Hashim.
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        """Class for drawing a 3D arrow.
        :param xs:
        :type xs:
        :param ys:
        :type ys:
        :param zs:
        :type zs:
        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        """
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class BlochSphere:

    def __init__(self,
                 figsize=(10, 10),
                 label_fontsize=35,
                 tick_label_fontsize=20,
                 point_size=60,
                 point_alpha=1.0,
                 show_background_grid=True,
                 show_background=True,
                 rotation_angle=45):
        """
        Class for plotting points and vectors on the Bloch Sphere.
        :param figsize: figure size for Bloch Sphere (default: (10,10))
        :type figsize: tuple
        :param label_fontsize: fontsize for x-, y-, z-labels (default: 35)
        :type label_fontsize: int
        :param tick_label_fontsize:  fontsize for x-, y-, z-ticks (default: 20)
        :type tick_label_fontsize: int
        :param point_size: point size for scatter plots
        :type point_size: int
        :param point_alpha: opacity for points in scatter plots
        :type point_alpha: float
        :param show_background_grid: display x, y, z grids behind Bloch sphere
        :type show_background_grid: bool
        :param show_background: display background behind Bloch sphere
        :type show_background: bool
        :param rotation_angle: angle about the z-axis to rotate the Bloch sphere for viewing
        :type rotation_angle: int
        """

        self.figsize = figsize
        self.label_fontsize = label_fontsize
        self.tick_label_fontsize = tick_label_fontsize
        self.point_size = point_size
        self.point_alpha = point_alpha
        self.show_background_grid = show_background_grid
        self.show_background = show_background
        self.rotation_angle = rotation_angle

        self.fig = None

    def draw_bloch_sphere(self):
        """Draws an empty Bloch sphere."""
        phi = np.linspace(0, 2 * np.pi, 50)
        theta = np.linspace(0, np.pi, 50)
        PHI, THETA = np.meshgrid(phi, theta)

        x_sphere = np.sin(PHI) * np.cos(THETA)
        y_sphere = np.sin(PHI) * np.sin(THETA)
        z_sphere = np.cos(PHI)

        self.fig = plt.figure(figsize=self.figsize)
        self.ax = plt.axes(projection='3d')
        self.ax.plot_wireframe(x_sphere, y_sphere, z_sphere, rstride=1, cstride=1, color='k', alpha=0.1, linewidth=1)
        self.ax.plot([-1, 1], [0, 0], [0, 0], c='k', alpha=0.5)
        self.ax.plot([0, 0], [-1, 1], [0, 0], c='k', alpha=0.5)
        self.ax.plot([0, 0], [0, 0], [-1, 1], c='k', alpha=0.5)
        self.ax.plot(np.cos(phi), np.sin(phi), 0, c='k', alpha=0.5)
        self.ax.plot(np.zeros(50), np.sin(phi), np.cos(phi), c='k', alpha=0.5)
        self.ax.plot(np.sin(phi), np.zeros(50), np.cos(phi), c='k', alpha=0.5)
        self.ax.set_xlabel(r'$\langle x \rangle$', fontsize=self.label_fontsize)
        self.ax.set_ylabel(r'$\langle y \rangle$', fontsize=self.label_fontsize)
        self.ax.set_zlabel(r'$\langle z \rangle$', fontsize=self.label_fontsize)
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)
        self.ax.set_xticklabels(['-1', '', '', '', '', '', '', '', '1'], fontsize=self.tick_label_fontsize)
        self.ax.set_yticklabels(['-1', '', '', '', '', '', '', '', '1'], fontsize=self.tick_label_fontsize)
        self.ax.set_zticklabels(['-1', '', '', '', '', '', '', '', '1'], fontsize=self.tick_label_fontsize)
        self.ax.set_facecolor('white')
        self.ax.grid(self.show_background_grid, color='k')
        if self.show_background is False:
            self.ax.set_axis_off()
        if self.rotation_angle is not None:
            self.ax.view_init(30, self.rotation_angle)

    def add_points(self, points, color=None):
        """Adds points to the Bloch sphere.
        :param points: [x, y, z] coordinates for a point
            Each can be an individual list of multiple coordinates for multiple points.
        :type points: list|np.array
        :param color: color of points for scatter point (default: None)
        :type color: None|str|RGB
        """
        """Add points to the Bloch Sphere."""
        if self.fig is None:
            self.draw_bloch_sphere()

        x, y, z = points
        if color is None:
            self.ax.scatter3D(x, y, z, s=self.point_size, alpha=self.point_alpha)
        else:
            self.ax.scatter3D(x, y, z, s=self.point_size, alpha=self.point_alpha, color=color)

    def add_vector(self, vector, color=None):
        """Add a vector to the Bloch sphere.
        :param vector: [x, y, z] coordinates for the tip of a vector
        :type vector: list|np.array
        :param color: color of vector (default: None)
        :type color: None|str|RGB
        :return:
        :rtype:
        """
        """Add points to the Bloch Sphere."""
        if self.fig is None:
            self.draw_bloch_sphere()

        x, y, z = vector
        if color is None:
            p = self.ax.plot([0, x], [0, y], [0, z], linewidth=3)
            a = Arrow3D([0, x], [0, y], [0, z], mutation_scale=35, arrowstyle='-|>', color=p[0].get_color())
        else:
            self.ax.plot([0, x], [0, y], [0, z], linewidth=3, color=color)
            a = Arrow3D([0, x], [0, y], [0, z], mutation_scale=35, arrowstyle='-|>', color=color)
        self.ax.add_artist(a)

    def show(self, save=False, directory=None, filename=None):
        """Plot the Bloch Sphere in a figure.
        :param save: save the figure (default: False
        :type save: bool
        :param directory: directory in which the save the figure (default: None)
            If None, it will save in the current directory.
        :type directory: None|str
        :param filename: string to prepend in front for 'Bloch_sphere.png' for a filename
        :type filename: None|str
        """
        if self.fig is None:
            self.draw_bloch_sphere()
        plt.tight_layout()
        if save is True:
            plt.savefig(f'{directory}{filename}Bloch_sphere.png', dpi=300)
        plt.show()