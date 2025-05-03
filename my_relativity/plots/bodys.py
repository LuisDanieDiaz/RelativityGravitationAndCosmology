from importlib import resources
import matplotlib
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.ndimage import rotate
from .frames import get_r0_2D, get_r0_2D_vec
from ..newton import *
from . import images
from .. import fonts
from scipy.ndimage import zoom as zoom_
from matplotlib.patches import Wedge
from matplotlib import font_manager

class Image_2D:
    def __init__(self, image_path:str, r0=[0,0], zoom=0.1, alpha=1, axis='xy', angle=0, origin='cc', reflex=False):
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
        self.angle = angle
        self.origin = origin

        self.img = matplotlib.image.imread(self.image_path)
        if reflex == 'x':
            self.img = np.flip(self.img, axis=1)
        elif reflex == 'y':
            self.img = np.flip(self.img, axis=0)
        elif reflex == 'xy' or reflex == 'yx':
            self.img = np.flip(self.img, axis=(0, 1))
        
        if angle != 0:
            self.rotate_image(angle)

        if self.zoom < 1.0:
            factor = self.zoom*2
            self.img = zoom_(self.img, (factor, factor, 1))
            self.zoom = 0.5

        self.artist = None
        self.rotate_images = {}

        self.img_original = self.img.copy()

    def rotate_image(self, angle):
        """
        Rotate the image by a given angle.
        @param angle: Angle in degrees.
        """
        self.img = rotate(self.img, angle, reshape=True)
        self.angle = angle

    def zoom_(self, zoom):
        """
        Zoom the image by a given factor.
        @param zoom: Zoom factor.
        """
        self.img = zoom_(self.img, (zoom, zoom, 1))
        self.zoom = 0.5

    def drawn(self, frame, zorder=-1, angle=0):
        """
        Draw the image in the frame, rotated by a given angle.
        
        @param frame: Frame_2D object where the image will be drawn.
        @param zorder: Z-order of the image.
        @param angle: Angle in degrees.
        @param origin: Position origin: 
                    - First char: 'c' (center), 't' (top), 'b' (bottom)
                    - Second char: 'c' (center), 'l' (left), 'r' (right)
                    e.g. 'cc' (center-center), 'tl' (top-left), 'br' (bottom-right)
        """
        r0_2D = get_r0_2D(self.r0 + frame.r0, self.axis)
        
        img_rotated = self.rotate_images.get(angle)
        if img_rotated is None:
            img_rotated = rotate(self.img, angle, reshape=True)
            self.rotate_images[angle] = img_rotated
        else:
            pass
        
        # Clip to allowed range (float in [0, 1], or int in [0, 255])
        if img_rotated.dtype == np.uint8:
            img_rotated = np.clip(img_rotated, 0, 255)
        else:
            img_rotated = np.clip(img_rotated, 0.0, 1.0)
        
        imgBox = OffsetImage(img_rotated, zoom=self.zoom, alpha=self.alpha)
        
        # Map the origin code to box_alignment tuple
        alignment_map = {
            'tl': (0, 1),  # top-left
            'tc': (0.5, 1),  # top-center
            'tr': (1, 1),  # top-right
            'cl': (0, 0.5),  # center-left
            'cc': (0.5, 0.5),  # center-center (default)
            'cr': (1, 0.5),  # center-right
            'bl': (0, 0),  # bottom-left
            'bc': (0.5, 0),  # bottom-center
            'br': (1, 0)   # bottom-right
        }
        
        # Use default if origin is not in the map
        box_align = alignment_map.get(self.origin, (0.5, 0.5))
        
        self.artist = AnnotationBbox(
            imgBox, (r0_2D[0], r0_2D[1]), 
            frameon=False, pad=0.0,
            box_alignment=box_align,
            clip_on=True
        )
        
        self.artist.set_zorder(zorder)
        self.artist.set_clip_on(True) 
        
        frame.ax.add_artist(self.artist)
        
    def resize(self, xside=1, yside=1):
        """
        Resize the body.
        @param xside: Side of the body in x direction.
        @param yside: Side of the body in y direction.
        """
        self.img = zoom_(self.img_original, (yside, xside, 1))
        self.rotate_images = {}


class Body_2D(Particle, Image_2D):
    def __init__(self, image_path:str, mass=1, r0=[0,0,0], v0=[0,0,0], zoom=0.1, alpha=1, axis='xy', angle=0, origin='cc', reflex=False):
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
        Image_2D.__init__(self, image_path=image_path, r0=r0, zoom=zoom, alpha=alpha, axis=axis, angle=angle, origin=origin, reflex=reflex)

        self.times = np.zeros(0)
        self.rs = np.zeros((0, 3))
        self.vs = np.zeros((0, 3))


    def evolution(self, coordinate_system, dt=0.01):
        """
        Evolve the body in the coordinate system.
        @param coordinate_system: Coordinate_system_2D object where the body will evolve.
        @param dt: Time step for the evolution.
        """
        self.times = np.append(self.times, coordinate_system.time)
        self.rs = np.append(self.rs, [self.r0], axis=0)
        self.vs = np.append(self.vs, [self.v0], axis=0)
        return super().evolution(dt)

    def drawn_path(self, coordinate_system):
        """
        Draw the path of the body in the frame.
        @param frame: Frame_2D object where the path will be drawn.
        @param coordinate_sistem: Coordinate_system_2D object where the path will be drawn.
        """
        # Get the path of the body in the coordinate system

        rs_prime = galilean_transformation(self.rs, coordinate_system.v0, self.times)
        r0_new_2D = get_r0_2D_vec(rs_prime + coordinate_system.r0, self.axis)

        # Draw the path
        coordinate_system.ax.plot(r0_new_2D[:, 0], r0_new_2D[:, 1], color='blue', alpha=0.5, zorder=-1)


class Preloaded_Images:
    """
    Preloaded images for the 2D bodies.
    """
    def __init__(self, type="paper"):
        self.type = type

    def get_image(self, name, **kwargs):
        """
        Get the image of the body.
        @param name: Name of the image.
        """
        image_name = f'{name}.png'
        try:
            image_file = resources.files(images).joinpath(self.type).joinpath(image_name)
            return Body_2D(image_path=str(image_file), **kwargs)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image {image_name} not found in images/{self.type}/.")





class Clock:
    def __init__(self, r0=[0,0,0], time=0, zoom=0.15, units=['s', 60], zoom_text=0, position_text=[0,-1,0]):
        """
        @param time: time in any units
        @param zoom: zoom factor for the clock
        @param units: ['s', 60] units for the clock, and the number of units in each step
        @param zoom_text: zoom factor for the text
        """
        self.time = time
        self.zoom = zoom
        self.units = units
        self.zoom_text = zoom_text
        if zoom_text == 0:
            self.zoom_text = zoom
        self.position_text = np.array(position_text) * self.zoom_text


        preload = Preloaded_Images()
        self.clock = preload.get_image('clock', zoom=zoom, r0=r0)
        self.clock_hand = preload.get_image('clock_hand', zoom=zoom, r0=r0)
        self.shelf = preload.get_image('paper_shelf_gray', zoom=self.zoom_text, r0=self.position_text + r0)
        self.r0 = np.array(r0)

        titan_path = "TitanOne-Regular.ttf"  # ruta al archivo .ttf
        font_path = resources.files(fonts).joinpath(titan_path)
        self.titan_font = font_manager.FontProperties(fname=font_path)


    def drawn(self, S, zorder=-1, color='r', t='t', axis='xy'):
        hour = -self.time * 360/self.units[1]  # Convert to degrees

        r0 = self.r0 + S.r0
        r0_2D = get_r0_2D(r0, axis)

        self.clock.drawn(S, zorder=2 + zorder)
        
        r0_display = S.ax.transData.transform(r0_2D)  # de data a display (pixeles)
        wedge = Wedge(r0_display, 200*self.zoom, theta1=90 + hour, theta2=90,
                    facecolor=color, alpha=0.5, zorder=3 + zorder)
        wedge.set_transform(S.ax.transData.inverted() + S.ax.transData)

        self.clock_hand.drawn(S, zorder=4 + zorder, angle=hour)
        self.shelf.drawn(S, zorder=5 + zorder)
        S.ax.add_patch(wedge)

        

        r0_text = self.position_text + r0
        r0_2D_text = get_r0_2D(r0_text, axis)

        S.ax.text(*r0_2D_text, f"{t} = {self.time:.2f} {self.units[0]}", fontsize=14 * self.zoom_text/0.2, fontproperties=self.titan_font, ha='center', va='center', color=(1,1,1,0.8), zorder=5 + zorder, family='monospace')


class Break_Point:
    def __init__(self, img_index=0, time=0, r0=[0,0,0], zoom=None, alpha=None, zorder=5):
        """
        Create a break point with the time and position.
        @param time: Time of the break point.
        @param r0: Position of the break point.
        """
        self.img_index = img_index
        self.time = time
        self.r0 = np.array(r0)
        self.zoom = zoom
        self.alpha = alpha
        self.zorder = zorder
    
    def drawn(self, S, animation):
        """
        Draw the break point in the animation.
        @param animation: Animation object where the break point will be drawn.
        """
        image = animation.images[self.img_index]
        if self.alpha != None: image.alpha = self.alpha
        if self.zoom != None: image.zoom_(self.zoom)
        image.drawn(S, zorder=self.zorder)
        
        

class Animation:
    def __init__(self, *images_names, time=0, zoom=0.15, alpha=1):
        """
        Create an animation with the images.
        """
        self.index = 0
        self.time = time
        self.break_points = [Break_Point()]
        preload = Preloaded_Images()
        self.images = [ 
            preload.get_image(name, zoom=zoom, alpha=alpha) for name in images_names 
            ]
        self.break_point = self.break_points[0]
        

    def set_break_point(self, img_index=0, time=0, r0=[None], zoom=None, alpha=None):
        """
        Set the break points for the animation.
        """
        if self.break_point.time >= time:
            raise ValueError(f"The time of the break point must be greater than the previous one. t={time}")
        break_point = Break_Point(img_index=img_index, time=time, r0=r0, zoom=zoom, alpha=alpha)
        self.break_points.append(break_point)
    
    def set_break_points(self, n, img_index:list, times:list, r0s=[None], zoom=[None], alpha=[None]):
        r0s = [None] * n if r0s == [None] else r0s
        zoom = [None] * n if zoom == [None] else zoom
        alpha = [None] * n if alpha == [None] else alpha

        if len(img_index) != n:
            raise ValueError("The number of the image must be equal to the number of break points. ")
        if len(times) != n:
            raise ValueError("The number of the time must be equal to the number of break points.")
        if len(r0s) != n:
            raise ValueError("The number of the r0 must be equal to the number of break points.")
        if len(zoom) != n:
            raise ValueError("The number of the zoom must be equal to the number of break points.")
        if len(alpha) != n :
            raise ValueError("The number of the alpha must be equal to the number of break points.")
        
        for i in range(n):
            self.set_break_point(img_index[i], times[i], r0s[i], zoom[i], alpha[i])

    def drawn(self, S):
        """
        Draw the animation in the frame.
        @param S: Frame_2D object where the animation will be drawn.
        @param zorder: Z-order of the image.
        @param angle: Angle in degrees.
        """
        next_break_point = self.break_points[self.index + 1] if self.index + 1 < len(self.break_points) else None
        
        if next_break_point is not None and self.time >= next_break_point.time:
            self.break_point = next_break_point
            self.index += 1
        self.break_point.drawn(S, self)
    
    def restart(self):
        """
        Restart the animation.
        """
        self.index = 0
        self.break_point = self.break_points[0]
        self.time = 0
