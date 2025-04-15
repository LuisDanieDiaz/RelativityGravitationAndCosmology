from importlib import resources
import matplotlib
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.ndimage import rotate
from .frames import get_r0_2D, get_r0_2D_vec
from ..newton import *
from . import images

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

        r0_new_2D = get_r0_2D_vec(rs_prime, self.axis)

        # Draw the path
        frame.ax.plot(r0_new_2D[:, 0], r0_new_2D[:, 1], color='blue', alpha=0.5, zorder=-1)

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
            print(image_file)
            return Body_2D(image_path=str(image_file), **kwargs)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image {image_name} not found in images/{self.type}/.")





        