from newton import *

def test_Particle():
    r0 = np.array([0, 0, 0])
    v0 = np.array([1, 0, 0])
    t = 0.01
    P1 = Particle(mass=1, r0=r0, v0=v0)
    F1 = Force(lambda mass, r0, v0: np.array([0, 0, 0]))
    
    # Add force
    P1.add_force(F1)
    P1.evolution(dt=t)
    assert np.isclose(P1.r0, r0 + v0*t).all(), f"Expected position {r0 + v0*t}, but got {P1.r0}"
    assert np.isclose(P1.v0, v0).all(), f"Expected velocity {v0}, but got {P1.v0}"

    v0 = np.array([0, 1, 0])
    t = 0.01
    P1 = Particle(mass=1, r0=r0, v0=v0)
    F1 = Force(lambda mass, r0, v0: np.array([0, 0, 0]))
    
    # Add force
    P1.add_force(F1)
    P1.evolution(dt=t)
    assert np.isclose(P1.r0, r0 + v0*t).all(), f"Expected position {r0 + v0*t}, but got {P1.r0}"
    assert np.isclose(P1.v0, v0).all(), f"Expected velocity {v0}, but got {P1.v0}"

    r0 = np.array([1, 0, 0])
    v0 = np.array([0, 1, 0])
    t = 0.01
    P1 = Particle(mass=1, r0=r0, v0=v0)
    F1 = Force(lambda mass, r0, v0: np.array([0, 0, 0]))
    
    # Add force
    P1.add_force(F1)
    P1.evolution(dt=t)
    assert np.isclose(P1.r0, r0 + v0*t).all(), f"Expected position {r0 + v0*t}, but got {P1.r0}"
    assert np.isclose(P1.v0, v0).all(), f"Expected velocity {v0}, but got {P1.v0}"

    r0 = np.array([1, 0, 0])
    v0 = np.array([0, 1, 0])
    t = 0.01
    g = 9.81
    P1 = Particle(mass=1, r0=r0, v0=v0)
    F1 = Force(lambda mass, r0, v0: np.array([0, -g, 0]))
    
    # Add force
    P1.add_force(F1)
    P1.evolution(dt=t)
    assert np.isclose(P1.r0, r0 + v0*t + np.array( [0,-g,0])*t**2 / 2).all(), f"Expected position {r0 + v0*t - g*t**2 / 2}, but got {P1.r0}"
    assert np.isclose(P1.v0, v0 +np.array( [0,-g,0])*t).all(), f"Expected velocity {v0 - g*t}, but got {P1.v0}"

if __name__ == "__main__":
    test_Particle()
    print("All tests passed!")