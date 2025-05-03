import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
import numpy as np
from ..newton import *
from importlib import resources
from . import images
from ..relativity import *
from scipy.ndimage import zoom as zoom_
from PIL import Image
from matplotlib.patches import Wedge

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

def get_r0_2D_vec(r0, axis):
    if axis == 'xy':
        r0_2D = r0[:, [0, 1]]
    elif axis == 'xz':
        r0_2D = r0[:, [0, 2]]
    elif axis == 'yz':
        r0_2D = r0[:, [1, 2]]
    else:
        raise ValueError("Invalid axis. Choose 'xy', 'xz', or 'yz'.")

    return r0_2D


def plot_line_between_2_events(ax, event1, event2, color='k', alpha=0.5, axis='tx', lw=0.5, **kwargs):
    """
    Plot a line between two events in the frame.
    @param frame: Frame_2D object where the line will be drawn.
    @param event1: First event.
    @param event2: Second event.
    @param color: Color of the line.
    @param alpha: Transparency of the line.
    @param axis: Axis to plot on.
    """
    x1, y1 = get_rmu0_2D(event1, axis)
    x2, y2 = get_rmu0_2D(event2, axis)

    # Draw the line
    ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, lw=lw, **kwargs)

def plot_spring(ax, x0, y0, x1, y1, k=6, amplitude=0.1, color='black', lw=2, **kwargs):
    # Asegurarse de que k sea par
    if k % 2 != 0:
        k += 1

    ts = np.linspace(0, 1, 4 * k + 1)  # puntos fijos que aseguran los extremos en 0
    ys_base = np.sin(2 * np.pi * k * ts) * amplitude  # nulo en extremos
    xs_base = ts

    

    # Vector dirección
    dx = x1 - x0
    dy = y1 - y0
    L = np.hypot(dx, dy)
    angle = np.arctan2(dy, dx)

    # Escalar y rotar
    xs_rot = xs_base * L
    ys_rot = ys_base

    x_final = xs_rot * np.cos(angle) - ys_rot * np.sin(angle) + x0
    y_final = xs_rot * np.sin(angle) + ys_rot * np.cos(angle) + y0

    ax.plot(x_final, y_final, color=color, lw=lw, **kwargs)


class Frame_3D:
    def __init__(self, figsize:tuple=(6,6), title:str='', show_title:bool=True):
        """
        Create a 3D figure with a specific size and remove the background.
        @param figsize: Tuple with the size of the figure.
        @param title: Title of the figure.
        @param show_title: Show the title of the figure.
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
    def __init__(self, ax, mass=1., origin=[0.,0.,0.], time=0., v0=[0.,0.,0.], size=1, 
                 name_cs='S', name_axis=['X', 'Y'], 
                 show_names=True, show_name_cs=True, alpha=1, axis='xy'):
        """
        Create a 2D coordinate system.
        @param ax: ax object where the coordinate system will be drawn.
        @param origin: Origin of the coordinate system.
        @param size: Size of the coordinate system.
        @param name_cs: Name of the coordinate system.
        @param nombres_ejes: Names of the axes.
        """
        super().__init__(mass=mass, r0=origin, v0=v0)
        
        self.ax = ax
        self.size = size
        self.name_cs = name_cs
        self.name_axis = name_axis
        self.show_names = show_names
        self.show_name_cs = show_name_cs
        self.alpha = alpha
        self.axis = axis
        self.time = time

    def evolution(self, *particles, dt=0.01):
        """
        Evolve the coordinate system in the frame.
        @param dt: Time step for the evolution.
        @param particles: Particles to be evolved.
        """
        self.time += dt

        for particle in particles:
            particle.evolution(self, dt=dt)

        return super().evolution(dt)

    def drawn(self, *particles, zorder=1):
        """
        Draw the coordinate system in the frame.
        """
        # axis
        r0_2D = get_r0_2D(self.r0, self.axis)

        arrow_x = FancyArrow(*r0_2D, self.size, 0, color='r', alpha=self.alpha, width=0.02, zorder=zorder)
        arrow_y = FancyArrow(*r0_2D, 0, self.size, color='b', alpha=self.alpha, width=0.02, zorder=zorder)

        self.ax.add_patch(arrow_x)
        self.ax.add_patch(arrow_y)

        self.show_name_axis()
        self.show_name_coordinate_system()
    
    def drawn_children(self, *children):
        """
        Draw the children of the coordinate system in the frame.
        """
        for child in children:
            child.drawn(frame=self)

    def drawn_axis(self, alpha=0.3, space=0.5, space_x=0, space_y=0):
        if space_x == 0:
            space_x = space
        if space_y == 0:
            space_y = space

        # black lines in the main
        self.ax.plot([self.ax.xlims[0], self.ax.xlims[1]], [0, 0], color='k', alpha=alpha, lw=1.5)
        self.ax.plot([0, 0], [self.ax.ylims[0], self.ax.ylims[1]], color='k', alpha=alpha, lw=1.5)


    def drawn_grid(self, alpha=0.3, space=0.5, space_x=0, space_y=0):
        """
        Draw the coordinate system in the frame.
        """
        if space_x == 0:
            space_x = space
        if space_y == 0:
            space_y = space
        # axis
        for i in np.arange(self.ax.xlims[0], self.ax.xlims[1], space_x*self.size):
            self.ax.plot([i, i], [self.ax.ylims[0], self.ax.ylims[1]], color='k', alpha=alpha, lw=0.5)
            self.ax.plot([i, i], [-self.size/16, self.size/16], color='k', alpha=alpha, lw=0.5)
            # number
            self.ax.text(i, -self.size/16, str(round(i, 2)), color='k', alpha=alpha, fontsize=8, ha='center', va='top')


        for i in np.arange(self.ax.ylims[0], self.ax.ylims[1], space_y*self.size):
            self.ax.plot([self.ax.xlims[0], self.ax.xlims[1]], [i, i], color='k', alpha=alpha, lw=0.5)
            self.ax.plot([-self.size/16, self.size/16], [i, i], color='k', alpha=alpha, lw=0.5)
            # number
            self.ax.text(-self.size/16, i, str(round(i, 2)), color='k', alpha=alpha, fontsize=8, ha='right', va='center')

        # black lines in the main
        self.ax.plot([self.ax.xlims[0], self.ax.xlims[1]], [0, 0], color='k', alpha=alpha, lw=1.5)
        self.ax.plot([0, 0], [self.ax.ylims[0], self.ax.ylims[1]], color='k', alpha=alpha, lw=1.5)
       

    def show_name_axis(self):
        """
        Draw the names of the axes in the frame.
        """
        r0_2D = get_r0_2D(self.r0, self.axis)
        if self.show_names:
            self.ax.text(r0_2D[0] + self.size*1.15, r0_2D[1], self.name_axis[0], color='r', alpha=self.alpha)
            self.ax.text(r0_2D[0], r0_2D[1] + self.size*1.15, self.name_axis[1], color='g', alpha=self.alpha)
            self.ax.text(r0_2D[0] - 0.2, r0_2D[1], 't', color='k', alpha=self.alpha)

    def show_name_coordinate_system(self):
        """
        Draw the name of the coordinate system in the frame.
        """
        r0_2D = get_r0_2D(self.r0, self.axis)
        if self.show_name_cs:
            self.ax.text(r0_2D[0] - 0.2, r0_2D[1] + self.size, self.name_cs, color='black', fontsize=12, ha='center', va='center', alpha=self.alpha)


    

class Coordinate_system_Lorentz_2D:
    def __init__(self, ax, origin=[0,0,0,0], velocity=[0,0,0], axis='tx', name_cs='S', name_axis=['x', 't'],
                 show_names=True, show_name_cs=True, size=1, alpha=1):
        """
        Create a 2D coordinate system.
        @param ax: Frame_2D object where the coordinate system will be drawn.
        @param velocity: Velocity of the coordinate system.
        @param origin: Origin of the coordinate system.
        @param axis: Axis of the coordinate system.        
        """
        self.ax = ax
        self.origin = np.array(origin)
        self.velocity = np.array(velocity)
        self.axis = axis
        self.name_cs = name_cs
        self.name_axis = name_axis
        self.show_names = show_names
        self.show_name_cs = show_name_cs
        self.size = size
        self.alpha = alpha

        self.unit_t = np.array([1, 0, 0, 0])
        self.unit_x = np.array([0, 1, 0, 0])
        self.unit_y = np.array([0, 0, 1, 0])
        self.unit_z = np.array([0, 0, 0, 1])

        self.speed = np.linalg.norm(self.velocity)

    def get_axis(self, axis):
        if axis == 't': return self.unit_t
        elif axis == 'x': return self.unit_x
        elif axis == 'y': return self.unit_y
        elif axis == 'z': return self.unit_z
        else: raise ValueError("Invalid axis. Choose 't', 'x', 'y', or 'z'.")

    def get_transformation(self):
        origin = Lorentz_transformation(rmu0=self.origin, v0=self.velocity)

        i, j = self.axis
        self.rmux_prime = Lorentz_transformation(rmu0=self.get_axis(i), v0=self.velocity)
        self.rmuy_prime = Lorentz_transformation(rmu0=self.get_axis(j), v0=self.velocity)

        self.rx_2D = get_rmu0_2D(self.rmux_prime, self.axis)
        self.ry_2D = get_rmu0_2D(self.rmuy_prime, self.axis)
        self.origin_transform = get_rmu0_2D(origin, self.axis)


    def drawn(self):
        """
        Draw the coordinate system in the frame.
        """
        self.get_transformation()

        arrow_x = FancyArrow(*self.origin_transform, *self.rx_2D, color='r', width=0.02, zorder=1, alpha=self.alpha)
        arrow_y = FancyArrow(*self.origin_transform, *self.ry_2D, color='b', width=0.02, zorder=1, alpha=self.alpha)

        self.ax.add_patch(arrow_x)
        self.ax.add_patch(arrow_y)

        self.show_name_axis()

    def drawn_grid(self, alpha=0.3, space_x=0.5, space_y=0.5):
        """
        Draw the coordinate system in the frame.
        """
        speed = np.linalg.norm(self.velocity)
        xlims = np.array(self.ax.xlims)*(1 + int(speed*10))
        ylims = np.array(self.ax.ylims)*(1 + int(speed*10))
        # Eje x--------------------------------------
        kwargs = {
            f'{self.axis[0]}_lims': xlims,
            f'{self.axis[1]}_lims': ylims,
            f'{self.axis[0]}_space': space_y,
            f'{self.axis[1]}_space': ylims[1] - ylims[0],
        }
        grid = grid_events(**kwargs)
        grid = Lorentz_transformation_vec(rmu0s=grid, v0=self.velocity)

        for i in range(0,len(grid)//2):
            plot_line_between_2_events(self.ax, grid[i], grid[i + len(grid)//2], color='k', alpha=alpha, axis=self.axis)

        # Eje y--------------------------------------
        kwargs = {
            f'{self.axis[0]}_lims': xlims,
            f'{self.axis[1]}_lims': ylims,
            f'{self.axis[0]}_space': xlims[1] - xlims[0],
            f'{self.axis[1]}_space': space_x,
        }

        grid = grid_events(**kwargs)
        grid = Lorentz_transformation_vec(rmu0s=grid, v0=self.velocity)

        for i in range(0,len(grid),2):
            plot_line_between_2_events(self.ax, grid[i], grid[i+1], alpha=alpha, axis=self.axis)

    def plot_event(self, event, *args, axis="xt", line_color='k',plot_projection=False, **kwargs):
        """ 
        Plot an event in the frame.
        @param event: Event to be plotted.
        @param args: Arguments for the plot.
        @param axis: Axis to plot on.
        @param kwargs: Keyword arguments for the plot.
        """
        # Transform the event to the new frame
        event = Lorentz_transformation(rmu0=event, v0=self.velocity)

        # Get the coordinates of the event in the new frame
        event_2D = get_rmu0_2D(event, axis)

        # Plot the event
        self.ax.plot(event_2D[0], event_2D[1], marker='o', *args, **kwargs)


        if plot_projection:
            
            self.ax.plot([event_2D[0], event_2D[0]], [0, event_2D[1]], '--', color=line_color, zorder=-3, alpha=0.5)
            self.ax.plot([0, event_2D[0]], [event_2D[1], event_2D[1]], '--', color=line_color, zorder=-3, alpha=0.5)
            
    
    def plot_events(self, events, *args, axis="xt", **kwargs):
        """
        Plot a list of events in the frame.
        @param events: List of events to be plotted.
        @param kwargs: Keyword arguments for the plot.
        """
        # Transform the events to the new frame
        events = Lorentz_transformation_vec(rmu0s=events, v0=self.velocity)

        # Get the coordinates of the events in the new frame
        events_2D = get_rmu0_2D_vec(events, axis)

        # Plot the events
        self.ax.plot(events_2D[:, 0], events_2D[:, 1], *args, **kwargs)


    def drawn_axis(self, alpha=0.3, space_x=0.5, space_y=0.5):
        """
        Draw the coordinate system in the frame.
        """
        ############# marcas de distancia ############
        # Eje x--------------------------------------
        kwargs = {
            f'{self.axis[0]}_lims': self.ax.xlims,
            f'{self.axis[0]}_space': space_x,
        }

        grid = grid_events(**kwargs)
        grid = Lorentz_transformation_vec(rmu0s=grid, v0=self.velocity)
        grid_2D = get_rmu0_2D_vec(grid, self.axis)

        between = lambda x, min_, max_: min_<= x and x <= max_ 
        for i in range(len(grid_2D)):
            x, y = grid_2D[i][0], grid_2D[i][1]
            if between(x, *self.ax.xlims) and between(y, *self.ax.ylims):
                self.ax.plot(x, y, color='k', alpha=alpha, lw=0.5)
                self.ax.text(x, y - 0.1, str(i*space_x + self.ax.xlims[0]), color='k', alpha=alpha, fontsize=8, ha='center', va='top')
        # Eje y -------------------------------------
        kwargs = {
            f'{self.axis[1]}_lims': self.ax.ylims,
            f'{self.axis[1]}_space': space_y,
        }
        grid = grid_events(**kwargs)
        grid = Lorentz_transformation_vec(rmu0s=grid, v0=self.velocity)
        grid_2D = get_rmu0_2D_vec(grid, self.axis)

        

        for i in range(len(grid_2D)):
            x, y = grid_2D[i][0], grid_2D[i][1]
            if between(x, *self.ax.xlims) and between(y, *self.ax.ylims):
                self.ax.plot(x, y, color='k', alpha=alpha, lw=0.5)
                self.ax.text(x - 0.1, y, str(i*space_y + self.ax.ylims[0]), color='k', alpha=alpha, fontsize=8, ha='right', va='center')

        #############################################

        # Ejes
        top = np.zeros(4)
        buttom = np.zeros(4)
        left = np.zeros(4)
        right = np.zeros(4)

        top[equivalences[self.axis[0]]] = self.ax.xlims[1]
        buttom[equivalences[self.axis[0]]] = self.ax.xlims[0]
        left[equivalences[self.axis[1]]] = self.ax.ylims[0]
        right[equivalences[self.axis[1]]] = self.ax.ylims[1]

        top = Lorentz_transformation(rmu0=top, v0=self.velocity)
        buttom = Lorentz_transformation(rmu0=buttom, v0=self.velocity)
        left = Lorentz_transformation(rmu0=left, v0=self.velocity)
        right = Lorentz_transformation(rmu0=right, v0=self.velocity)
        
        plot_line_between_2_events(self.ax, top, buttom, color='k', alpha=alpha, axis=self.axis)
        plot_line_between_2_events(self.ax, left, right, color='k', alpha=alpha, axis=self.axis)


    def show_name_axis(self):
        """
        Draw the names of the axes in the frame.
        """
        self.get_transformation()

        if self.show_names:
            posx = self.rx_2D[0]*self.size*1.15, self.rx_2D[1]*self.size*1.15
            posy = self.ry_2D[0]*self.size*1.15, self.ry_2D[1]*self.size*1.15
            if posx[0] > self.ax.xlims[0] and posx[0] < self.ax.xlims[1]:
                self.ax.text(*posx, self.name_axis[0], color='r', alpha=self.alpha)
            if posy[0] > self.ax.ylims[0] and posy[0] < self.ax.ylims[1]:
                self.ax.text(*posy, self.name_axis[1], color='b', alpha=self.alpha)

    def drawn_angle(self, sign=1, r=0.4, color='r', alpha=0.5, color_secondary='b', plot_secondary=False, angle_name=r"$\beta$", show_angle_name=True, show_angle_name_secondary=False):
        """
        Draw the angle between the two axes in the frame.
        """
        speed = np.linalg.norm(self.velocity)
        if speed == 0:
            return 0
        beta = np.arctan(speed)

        index_1 = int((sign + 1)/2)
        index_2 = int((sign - 1)/2)

        thetas = np.array([0, beta*180/np.pi])
        thetas = thetas[::sign]*sign
        wedge = Wedge(
            center=self.origin_transform,
            r=r,
            theta1=thetas[0],
            theta2=thetas[1],
            color=color,
            alpha=alpha,
            lw=0.5,
        )
        self.ax.add_patch(wedge)
        if show_angle_name:
            self.ax.text(self.origin_transform[0] + r*1.5*np.cos(np.deg2rad(thetas[index_1]/2)), 
                         self.origin_transform[0] + r*1.5*np.sin(np.deg2rad(thetas[index_1]/2)), 
                         angle_name, 
                         color=color, 
                         alpha=alpha, 
                         fontsize=16, 
                         ha='center', 
                         va='center')

        if plot_secondary:
            thetas = np.array([beta*180/np.pi, 0])
            thetas = thetas[::sign]
            thetas[0] = 90 - sign*thetas[0]
            thetas[1] = 90 - sign*thetas[1] 
            wedge = Wedge(
                center=self.origin_transform,
                r=r,
                theta1=thetas[0],
                theta2=thetas[1],
                color=color_secondary,
                alpha=alpha,
                lw=0.5,
            )
            self.ax.add_patch(wedge)
            if show_angle_name_secondary:
                self.ax.text(self.origin_transform[0] + r*1.5*np.cos(np.deg2rad((thetas[index_2]-90)/2 + 90)), 
                             self.origin_transform[0] + r*1.5*np.sin(np.deg2rad((thetas[index_2]-90)/2 + 90)), 
                             angle_name, 
                             color=color_secondary, 
                             alpha=alpha, 
                             fontsize=16, 
                             ha='center', 
                             va='center')


class Axis_2D(plt.Axes):
    """
    Class to create a 2D axis.
    """
    def quit_ticks(self):
        # Limit of the axes
        self.set_xlim(*self.xlims)
        self.set_ylim(*self.ylims)

        # Quit the grid and ticks
        self.grid(False)

    def quit_background(self):
        """
        Remove the background of the axis.
        """
        # Remove the background color
        self.set_facecolor('white')
        self.spines['top'].set_visible(False)
        self.spines['right'].set_visible(False)
        self.spines['left'].set_visible(False)
        self.spines['bottom'].set_visible(False)
        # Remove the ticks
        self.xaxis.set_ticks([])
        self.yaxis.set_ticks([])
    
    def set_background_image(self, path, zoom=0.2):
        """
        Set the background image of the axis.
        @param path: Path of the image.
        """
        # si no existe self.image_file, lo crea
        self.image_file = resources.files(images).joinpath(path)
        self.img = plt.imread(self.image_file)
        self.img = zoom_(self.img, (zoom, zoom, 1))

        if self.img.dtype == np.uint8:
            self.img = np.clip(self.img, 0, 255)
        else:
            self.img = np.clip(self.img, 0.0, 1.0)

    def draw_background(self, **kwargs):
        self.imshow(self.img, aspect='auto', extent=[self.get_xlim()[0], self.get_xlim()[1], self.get_ylim()[0], self.get_ylim()[1]], zorder=-2, **kwargs)
        
        
        

class Frame_2D:
    def __init__(self,*args, figsize:tuple=(6,6), title:str='', show_title:bool=True, xlims:tuple=(-2.5, 2.5), ylims:tuple=(-2.5, 2.5)):
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

        self.fig, self.axs = plt.subplots(*args, figsize=figsize,  subplot_kw={'axes_class': Axis_2D})
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)


        try:
            self.ax = self.axs[0]
        except:
            # También quitar el padding total en la figura
            self.fig.patch.set_visible(False)
            self.ax = self.axs
            self.axs = [self.ax]

        for ax in self.axs:
            ax.xlims = self.xlims
            ax.ylims = self.ylims
            ax.quit_background()
            ax.quit_ticks()


        # title
        if self.show_title: 
            # ax.set_title(self.title, fontsize=16)
            self.fig.suptitle(self.title, fontsize=16)
    
    def set_background_image(self, path, zoom=1):
        """
        Add a background image to the figure.
        @param path: Path of the image.
        """
        image_file = resources.files(images).joinpath(path)
        bg_img = plt.imread(image_file)
        # zoom
        bg_img = zoom_(bg_img, (zoom, zoom, 1))
        if bg_img.dtype == np.uint8:
            bg_img = np.clip(bg_img, 0, 255)
        else:
            bg_img = np.clip(bg_img, 0.0, 1.0)
        w_px, h_px = self.fig.get_size_inches() * self.fig.dpi
        w_px, h_px = int(w_px), int(h_px)
        bg_img_resized = Image.fromarray((bg_img * 255).astype('uint8')).resize((w_px, h_px))
        bg_img_resized = np.asarray(bg_img_resized) / 255 

        self.fig.figimage(bg_img_resized, xo=0, yo=0, zorder=-1)
        self.ax.patch.set_alpha(0)