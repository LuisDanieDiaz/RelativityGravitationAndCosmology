import numpy as np
import numbers

class Force:
    def __init__(self, function=lambda mass, r, v: np.array([0,0,0])):
        """
        Initialize a Force object with a vector.
        @param vector: A list or array-like object representing the force vector.
        """
        self.function = function

    def apply(self, mass, r, v):
        """
        Aply the force to the particle.

        """
        return self.function(mass, r, v)
    
    def __add__(self, other):
        """
        Add two Force objects together.
        @param other: Another Force object to add.
        @return: A new Force object representing the sum of the two forces.
        """
        return Force(lambda mass, r, v: self.apply(mass, r, v) + other.apply(mass, r, v))
    
class Particle:
    def __init__(self, mass=0., r0=[0,0,0], v0=[0,0,0]):
        """
        Initialize a Particle object with mass, initial position, and initial velocity.
        @param mass: Mass of the particle.
        @param r0: Initial position of the particle.
        @param v0: Initial velocity of the particle.
        """
        self.mass = mass
        self.r0 = np.array(r0)
        self.v0 = np.array(v0)
        self.force = Force()
        
    def add_force(self, force):
        """
        Add a force to the particle.
        @param force: A Force object representing the force to be added.
        """
        self.force += force

    def evolution(self, dt=0.01):
        """
        Evolves the position and velocity of the particle using the Leapfrog method.
        @param dt: Time step for the evolution.
        """
        # Aceleración actual
        a0 = self.force.apply(self.mass, self.r0, self.v0) / self.mass

        # Paso medio de velocidad
        v_half = self.v0 + 0.5 * a0 * dt

        # Paso completo de posición
        self.r0 = self.r0 + v_half * dt

        # Nueva aceleración
        a1 = self.force.apply(self.mass, self.r0, v_half) / self.mass

        # Paso medio final de velocidad
        self.v0 = v_half + 0.5 * a1 * dt

        
def galilean_transformation(r0, v0, t, inverse=False):
    """
    Galilean transformation.
    @param r0: Initial position.
    @param v0: Velocity.
    @param t: Time.
    """

    if not isinstance(t, numbers.Number):
        t = t[:, np.newaxis] 

    if inverse:
        return r0 + v0*t
    else:
        rs_prime = r0 - v0*t
        return rs_prime
