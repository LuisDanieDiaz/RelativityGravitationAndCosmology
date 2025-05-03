import numpy as np

equivalences = {
    "t": 0,
    "x": 1,
    "y": 2,
    "z": 3,
}

def Gamma(v0):
    """
    Calculate the Lorentz factor (gamma) for a given velocity vector.
    @param v0: Velocity vector.
    @return: Lorentz factor (gamma).
    """
    speed = np.linalg.norm(v0)
    if abs(speed) >= 1:
        raise ValueError("Velocity must be less than the speed of light (c=1).")
    return 1 / np.sqrt(1 - speed**2)

def get_rmu0_2D(rmu0, axis="tx"):
    """
    Get the 2D representation of the 4-vector rmu0.
    @param rmu0: 4-vector.
    @param axis: Axis to plot on.
    @return: 2D representation of the 4-vector.
    """
    if len(axis) != 2:
        raise ValueError("Axis must be a string of length 2.")

    try:
        i, j = equivalences[axis[0]], equivalences[axis[1]]
        return rmu0[i], rmu0[j]
    except KeyError:
        raise ValueError(f"Invalid axis: {axis}. Valid axes are 't', 'x', 'y', 'z'.")


def get_rmu0_2D_vec(rmu0, axis="tx"):
    """
    Get the 2D representation of the 4-vector rmu0.
    @param rmu0: 4-vector.
    @param axis: Axis to plot on.
    @return: 2D representation of the 4-vector.
    """
    if len(axis) != 2:
        raise ValueError("Axis must be a string of length 2.")

    try:
        i, j = equivalences[axis[0]], equivalences[axis[1]]
        return rmu0[:, [i, j]]
    except KeyError:
        raise ValueError(f"Invalid axis: {axis}. Valid axes are 't', 'x', 'y', 'z'.")


def get_rmu0_3D(rmu0, axis="txy"):
    """
    Get the 3D representation of the 4-vector rmu0.
    @param rmu0: 4-vector.
    @param axis: Axis to plot on.
    @return: 3D representation of the 4-vector.
    """
    if len(axis) != 3:
        raise ValueError("Axis must be a string of length 3.")

    try:
        i, j, k = equivalences[axis[0]], equivalences[axis[1]], equivalences[axis[2]]
        return np.array(rmu0[i], rmu0[j], rmu0[k])
    except KeyError:
        raise ValueError(f"Invalid axis: {axis}. Valid axes are 't', 'x', 'y', 'z'.")


def get_rmu0_3D_vec(rmu0, axis="txy"):
    """
    Get the 3D representation of the 4-vector rmu0.
    @param rmu0: 4-vector.
    @param axis: Axis to plot on.
    @return: 3D representation of the 4-vector.
    """
    if len(axis) != 3:
        raise ValueError("Axis must be a string of length 3.")

    try:
        i, j, k = equivalences[axis[0]], equivalences[axis[1]], equivalences[axis[2]]
        return rmu0[:, [i, j, k]]
    except KeyError:
        raise ValueError(f"Invalid axis: {axis}. Valid axes are 't', 'x', 'y', 'z'.")


def Lorentz_transformation(rmu0=np.array([0, 0, 0, 0]), v0=np.array([0, 0, 0])):
    """
    Perform Lorentz transformation on a 4-vector rmu0 with velocity v0.
    Take from: TH. Transformaciones de Herglotz - Problem set 
    @param rmu0: 4-vector.
    @param v0: Velocity vector.
    @return: Transformed 4-vector.
    """
    speed = np.linalg.norm(v0)
    if speed >= 1:
        raise ValueError("Velocity must be less than the speed of light (c=1).")
    elif speed == 0:
        return rmu0
    # Define the Lorentz transformation matrix
    t, r0 = rmu0[0], rmu0[1:]
    gamma = 1 / np.sqrt(1 - np.dot(v0, v0))

    t_prime = gamma * (t - np.dot(v0, r0))
    r_prime = r0 + (((gamma - 1) / np.dot(v0, v0)) * np.dot(v0, r0) - gamma * t) * v0

    rmu_prime = np.array([t_prime, *r_prime])

    return rmu_prime


def grid_events(
    t_lims=[0],
    x_lims=[0],
    y_lims=[0],
    z_lims=[0],
    t_space=0,
    x_space=0,
    y_space=0,
    z_space=0,
    include_t=True,
    include_x=True,
    include_y=True,
    include_z=True,
):
    """
    Crea una cuadrícula de eventos en un espacio tridimensional.
    @pars t_lims: límites de tiempo (t_min, t_max)
    @pars x_lims: límites en el eje x (x_min, x_max)
    @pars y_lims: límites en el eje y (y_min, y_max)
    @pars z_lims: límites en el eje z (z_min, z_max)
    @pars t_space: espacio entre eventos en el eje temporal
    @pars x_space: espacio entre eventos en el eje x
    @pars y_space: espacio entre eventos en el eje y
    @pars z_space: espacio entre eventos en el eje z
    """
    epsilon = 0.0001
    epsilon_t = epsilon if include_t else 0
    epsilon_x = epsilon if include_x else 0
    epsilon_y = epsilon if include_y else 0
    epsilon_z = epsilon if include_z else 0

    # Crear una cuadrícula de eventos en el espacio tridimensional
    t = np.arange(t_lims[0], t_lims[1] + epsilon_t, t_space) if t_space != 0 else t_lims
    x = np.arange(x_lims[0], x_lims[1] + epsilon_x, x_space) if x_space != 0 else x_lims
    y = np.arange(y_lims[0], y_lims[1] + epsilon_y, y_space) if y_space != 0 else y_lims
    z = np.arange(z_lims[0], z_lims[1] + epsilon_z, z_space) if z_space != 0 else z_lims

    # Crear una malla tridimensional
    T, X, Y, Z = np.meshgrid(t, x, y, z)

    # Convertir a una lista de eventos
    events = np.array([T.flatten(), X.flatten(), Y.flatten(), Z.flatten()]).T

    return events


def Lorentz_transformation_vec(rmu0s, v0=np.array([0, 0, 0])):
    """
    Perform Lorentz transformation on a 4-vector rmu0 with velocity v0.
    @param rmu0s: 4-vector.
    @param v0: Velocity vector.
    @return: Transformed 4-vector.
    """
    speed = np.linalg.norm(v0)
    if speed >= 1:
        raise ValueError("Velocity must be less than the speed of light (c=1).")
    elif speed == 0:
        return rmu0s
    # Define the Lorentz transformation matrix
    t, r0 = rmu0s[:, 0], rmu0s[:, 1:]
    gamma = 1 / np.sqrt(1 - np.dot(v0, v0))

    t_prime = gamma * (t - r0 @ v0)
    r_prime = ((gamma - 1) / np.dot(v0, v0)) * np.dot(r0, v0) - gamma * t
    r_prime = r_prime[:, None] * v0
    rmu_prime = np.column_stack((t_prime, r0 + r_prime))

    return rmu_prime
