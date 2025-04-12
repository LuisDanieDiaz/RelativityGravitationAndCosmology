import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.ndimage import rotate
import numpy as np
from .newton import Particle

def get_r0_2D(r0, axis):
    if axis == 'xy':
        r0_2D = [r0[0], r0[1]]
    elif axis == 'xz':
        r0_2D = [r0[0], r0[2]]
    elif axis == 'yz':
        r0_2D = [r0[1], r0[2]]
    else:
        raise ValueError("Invalid axis. Choose 'xy', 'xz', or 'yz'.")

    return r0_2D

get_r0_2D_vec = np.vectorize(get_r0_2D, signature='(n),(m)->(n)')

class Frame_2D:
    def __init__(self, figsize:tuple=(6,6), title:str='', show_title:bool=True, xlims:tuple=(-2.5, 2.5), ylims:tuple=(-2.5, 2.5)):
        """
        Create a 2D figure with a specific size and remove the background.
        @param figsize: Tuple with the size of the figure.
        @param title: Title of the figure.
        @param shoe_title: Show the title of the figure.
        """
        self.title = title
        self.show_title = show_title
        self.xlims = xlims
        self.ylims = ylims

        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        self.quit_background()

        # Limit of the axes
        self.ax.set_xlim(*self.xlims)
        self.ax.set_ylim(*self.ylims)
        # self.ax.set_aspect('auto')

        # Quit the grid and ticks
        self.ax.grid(False)

        # title
        if self.show_title: 
            # self.ax.set_title(self.title, fontsize=16)
            self.fig.suptitle(self.title, fontsize=16)

    def quit_background(self):
        """
        Remove the background of the figure.
        """
        # Remove the background color
        self.ax.set_facecolor('white')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        # Remove the ticks
        self.ax.xaxis.set_ticks([])
        self.ax.yaxis.set_ticks([])
        

class Frame_3D:
    def __init__(self, figsize:tuple=(6,6), title:str='', show_title:bool=True):
        """
        Create a 3D figure with a specific size and remove the background.
        @param figsize: Tuple with the size of the figure.
        @param title: Title of the figure.
        @param shoe_title: Show the title of the figure.
        """
        self.title = title
        self.show_title = show_title

        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.quit_background()
        
        # Limit of the axes
        self.ax.set_xlim([0, 2.5])
        self.ax.set_ylim([0, 2.5])
        self.ax.set_zlim([0, 2.5])

        # Quit the grid and ticks
        self.ax.grid(False)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])

        # title
        if self.show_title: self.ax.set_title(self.title, fontsize=16)

    def quit_background(self):
        """
        Remove the background of the figure.
        """
        # Remove the background color
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False

        # Change total transparency of axis
        self.ax.xaxis.line.set_color((1,1,1,0))
        self.ax.yaxis.line.set_color((1,1,1,0))
        self.ax.zaxis.line.set_color((1,1,1,0))

    def show(self):
        """
        Show the figure.
        """
        plt.show()


class Coordinate_system_3D:
    def __init__(self, frame, origin=[0,0,0], time=0, size=1, 
                 name_cs='S', name_axis=['X', 'Y', 'Z', 't'], 
                 show_names=True, show_name_cs=True, alpha=1):
        """
        Create a 3D coordinate system.
        @param frame: Frame_3D object where the coordinate system will be drawn.
        @param origin: Origin of the coordinate system.
        @param size: Size of the coordinate system.
        @param name_cs: Name of the coordinate system.
        @param nombres_ejes: Names of the axes.
        """
        self.frame = frame
        self.origin = origin
        self.time = time
        self.size = size
        self.name_cs = name_cs
        self.name_axis = name_axis
        self.show_names = show_names
        self.show_name_cs = show_name_cs
        self.alpha = alpha

    def drawn(self):
        """
        Draw the coordinate system in the frame.
        """
        # axis
        self.frame.ax.quiver(*self.origin, self.size, 0, 0, color='r', arrow_length_ratio=0.1, alpha=self.alpha)
        self.frame.ax.quiver(*self.origin, 0, self.size, 0, color='g', arrow_length_ratio=0.1, alpha=self.alpha)
        self.frame.ax.quiver(*self.origin, 0, 0, self.size, color='b', arrow_length_ratio=0.1, alpha=self.alpha)

        self.show_name_axis()
        self.show_name_coordinate_system()

    def show_name_axis(self):
        """
        Draw the names of the axes in the frame.
        """
        if self.show_names:
            self.frame.ax.text(self.origin[0] + self.size*1.15, self.origin[1], self.origin[2], self.name_axis[0], color='r', alpha=self.alpha)
            self.frame.ax.text(self.origin[0], self.origin[1] + self.size*1.15, self.origin[2], self.name_axis[1], color='g', alpha=self.alpha)
            self.frame.ax.text(self.origin[0], self.origin[1], self.origin[2] + self.size*1.15, self.name_axis[2], color='b', alpha=self.alpha)
            self.frame.ax.text(self.origin[0] - self.size*0.25, self.origin[1], self.origin[2] + 1, self.name_axis[3], color='k', alpha=self.alpha)
    
    def show_name_coordinate_system(self):
        """
        Draw the name of the coordinate system in the frame.
        """
        if self.show_name_cs:
            self.frame.ax.text(self.origin[0]-0.2, self.origin[1], self.origin[2], self.name_cs, color='black', fontsize=12, ha='center', va='center', alpha=self.alpha)



class Coordinate_system_2D(Particle):
    def __init__(self, frame, mass=1., origin=[0.,0.,0.], time=0., v0=[0.,0.,0.], size=1, 
                 name_cs='S', name_axis=['X', 'Y'], 
                 show_names=True, show_name_cs=True, alpha=1, axis='xy'):
        """
        Create a 2D coordinate system.
        @param frame: Frame_2D object where the coordinate system will be drawn.
        @param origin: Origin of the coordinate system.
        @param size: Size of the coordinate system.
        @param name_cs: Name of the coordinate system.
        @param nombres_ejes: Names of the axes.
        """
        super().__init__(mass=mass, r0=origin, v0=v0)
        
        self.frame = frame
        self.size = size
        self.name_cs = name_cs
        self.name_axis = name_axis
        self.show_names = show_names
        self.show_name_cs = show_name_cs
        self.alpha = alpha
        self.axis = axis
        self.time = time

    def evolution(self, dt=0.01):
        self.time += dt
        return super().evolution(dt)

    def drawn(self):
        """
        Draw the coordinate system in the frame.
        """
        # axis
        r0_2D = get_r0_2D(self.r0, self.axis)

        arrow_x = FancyArrow(*r0_2D, self.size, 0, color='r', alpha=self.alpha, width=0.02, zorder=1)
        arrow_y = FancyArrow(*r0_2D, 0, self.size, color='b', alpha=self.alpha, width=0.02, zorder=1)

        self.frame.ax.add_patch(arrow_x)
        self.frame.ax.add_patch(arrow_y)

        self.show_name_axis()
        self.show_name_coordinate_system()
    
    def drawn_grid(self, xlims=(-2.5, 2.5), ylims=(-2.5, 2.5), alpha=0.3, space=0.5):
        """
        Draw the coordinate system in the frame.
        """
        # axis
        r0_2D = get_r0_2D(self.r0, self.axis)

        side_x = xlims[1] - xlims[0]
        side_y = ylims[1] - ylims[0]

        for i in np.arange(r0_2D[0] - side_x, r0_2D[0] + side_x, space*self.size):
            self.frame.ax.plot([i, i], [ylims[0] - side_y, ylims[1] + side_y], color='k', alpha=alpha, lw=0.5)
            self.frame.ax.plot([i, i], [-self.size/16, self.size/16], color='k', alpha=alpha, lw=0.5)
            # number
            self.frame.ax.text(i, -self.size/16, str(round(i, 2)), color='k', alpha=alpha, fontsize=8, ha='center', va='top')


        for i in np.arange(r0_2D[1] - side_y, r0_2D[1] + side_y, space*self.size):
            self.frame.ax.plot([xlims[0] - side_x, xlims[1] + side_x], [i, i], color='k', alpha=alpha, lw=0.5)
            self.frame.ax.plot([-self.size/16, self.size/16], [i, i], color='k', alpha=alpha, lw=0.5)
            # number
            self.frame.ax.text(-self.size/16, i, str(round(i, 2)), color='k', alpha=alpha, fontsize=8, ha='right', va='center')

        # black lines in the main
        self.frame.ax.plot([xlims[0] - side_x, xlims[1] + side_x], [0, 0], color='k', alpha=alpha, lw=1.5)
        self.frame.ax.plot([0, 0], [ylims[0]-side_y, ylims[1] + side_y], color='k', alpha=alpha, lw=1.5)

            

    def show_name_axis(self):
        """
        Draw the names of the axes in the frame.
        """
        r0_2D = get_r0_2D(self.r0, self.axis)
        if self.show_names:
            self.frame.ax.text(r0_2D[0] + self.size*1.15, r0_2D[1], self.name_axis[0], color='r', alpha=self.alpha)
            self.frame.ax.text(r0_2D[0], r0_2D[1] + self.size*1.15, self.name_axis[1], color='g', alpha=self.alpha)
            self.frame.ax.text(r0_2D[0] - 0.2, r0_2D[1], 't', color='k', alpha=self.alpha)

    def show_name_coordinate_system(self):
        """
        Draw the name of the coordinate system in the frame.
        """
        r0_2D = get_r0_2D(self.r0, self.axis)
        if self.show_name_cs:
            self.frame.ax.text(r0_2D[0] - 0.2, r0_2D[1] + self.size, self.name_cs, color='black', fontsize=12, ha='center', va='center', alpha=self.alpha)



class Image_2D:
    def __init__(self, image_path:str, r0=[0,0], zoom=0.1, alpha=1, axis='xy'):
        """
        Create a 2D image.
        @param image_path: Path of the image.
        @param r0: Origin of the image.
        @param size: Size of the image.
        """
        self.image_path = image_path
        self.r0 = np.array(r0)  # Convertir a numpy array
        self.zoom = zoom
        self.alpha = alpha
        self.axis = axis

        self.img = matplotlib.image.imread(self.image_path)
        self.artist = None

        self.rotate_images = {}

    def drawn(self, frame, zorder=-1, angle=0):
        """
        Draw the image in the frame, rotated by a given angle.
        """
        r0_2D = get_r0_2D(self.r0, self.axis)

        img_rotated = self.rotate_images.get(angle)
        if img_rotated is None:
            # Rotar la imagen en sí (array)
            img_rotated = rotate(self.img, angle, reshape=True)

            # guardamos temporalmente la imagen rotada
            self.rotate_images[angle] = img_rotated

        else:
            # Si la imagen ya está rotada, no la rotamos de nuevo
            pass


        # Recortar al rango permitido (float en [0, 1], o int en [0, 255])
        if img_rotated.dtype == np.uint8:
            img_rotated = np.clip(img_rotated, 0, 255)
        else:
            img_rotated = np.clip(img_rotated, 0.0, 1.0)

        # Crear OffsetImage con la imagen ya rotada
        imgBox = OffsetImage(img_rotated, zoom=self.zoom)

        self.artist = AnnotationBbox(
            imgBox, (r0_2D[0], r0_2D[1]), frameon=False, pad=0.0
        )
        self.artist.set_zorder(zorder)

        frame.ax.add_artist(self.artist)



class Body_2D(Particle, Image_2D):
    def __init__(self, image_path:str, mass=1, r0=[0,0,0], v0=[0,0,0], zoom=0.1, alpha=1, axis='xy'):
        """
        Create a body with a mass, position and velocity.
        @param mass: Mass of the body.
        @param r0: Initial position of the body.
        @param v0: Initial velocity of the body.
        @param image_path: Path of the image.
        @param zoom: Zoom of the image.
        @param alpha: Alpha of the image.
        """
            
        Particle.__init__(self, mass=mass, r0=r0, v0=v0)
        Image_2D.__init__(self, image_path=image_path, r0=r0, zoom=zoom, alpha=alpha, axis=axis)

        self.times = np.zeros(0)
        self.rs = np.zeros((0, 3))
        self.vs = np.zeros((0, 3))


    def evolution(self, coordinate_system, dt=0.01):
        self.times = np.append(self.times, coordinate_system.time)
        self.rs = np.append(self.rs, [self.r0], axis=0)
        self.vs = np.append(self.vs, [self.v0], axis=0)
        return super().evolution(dt)

    def drawn_path(self, frame, coordinate_system):
        """
        Draw the path of the body in the frame.
        @param frame: Frame_2D object where the path will be drawn.
        @param coordinate_sistem: Coordinate_system_2D object where the path will be drawn.
        """
        # Get the path of the body in the coordinate system

        rs_prime = galilean_transformation(self.rs, coordinate_system.v0, self.times)
        rs_prime = galilean_transformation(rs_prime, coordinate_system.v0, coordinate_system.time, inverse=True)

        if self.axis == 'xy':
            r0_new_2D = rs_prime[:, [0, 1]]
        elif self.axis == 'xz':
            r0_new_2D = rs_prime[:, [0, 2]]
        elif self.axis == 'yz':
            r0_new_2D = rs_prime[:, [1, 2]]
        else:
            raise ValueError("Invalid axis. Choose 'xy', 'xz', or 'yz'.")

        # Draw the path
        frame.ax.plot(r0_new_2D[:, 0], r0_new_2D[:, 1], color='blue', alpha=0.5, zorder=-1)
        