import matplotlib.pyplot as plt
import numpy as np

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

        # arial
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

