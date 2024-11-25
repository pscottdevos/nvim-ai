import numpy as np
import matplotlib.pyplot as plt

class PressureGravitySimulation:
    def __init__(self, M1, M2, v1, v2, dt=0.01, duration=10):
        """
        Initialize simulation with point masses and initial conditions
        
        Args:
        M1, M2: Masses of two bodies
        v1, v2: Initial velocity vectors
        dt: Time step
        duration: Simulation duration
        """
        self.M1 = M1
        self.M2 = M2
        self.pos1 = np.array([0.0, 0.0])  # Initial position of mass 1
        self.pos2 = np.array([1.0, 0.0])  # Initial position of mass 2
        self.vel1 = np.array(v1)
        self.vel2 = np.array(v2)
        self.dt = dt
        self.duration = duration
        self.G_PRESSURE = 1.0  # Pressure gradient constant

    def pressure_gradient_force(self, pos1, pos2):
        """
        Calculate pressure gradient force
        Follows inverse square law based on distance
        """
        r = np.linalg.norm(pos2 - pos1)
        direction = (pos2 - pos1) / r
        force_magnitude = self.G_PRESSURE * (self.M1 * self.M2) / (r**2)
        return force_magnitude * direction

    def simulate(self):
        """
        Simulate orbital dynamics using pressure gradient model
        """
        steps = int(self.duration / self.dt)
        trajectory1 = [self.pos1]
        trajectory2 = [self.pos2]

        for _ in range(steps):
            # Calculate pressure gradient force
            force12 = self.pressure_gradient_force(self.pos1, self.pos2)
            force21 = -force12

            # Update velocities (F = ma)
            acc1 = force12 / self.M1
            acc2 = force21 / self.M2

            self.vel1 += acc1 * self.dt
            self.vel2 += acc2 * self.dt

            # Update positions
            self.pos1 += self.vel1 * self.dt
            self.pos2 += self.vel2 * self.dt

            trajectory1.append(self.pos1.copy())
            trajectory2.append(self.pos2.copy())

        return np.array(trajectory1), np.array(trajectory2)

    def plot_orbits(self, traj1, traj2):
        """
        Plot orbital trajectories
        """
        plt.figure(figsize=(10, 6))
        plt.plot(traj1[:, 0], traj1[:, 1], label='Mass 1')
        plt.plot(traj2[:, 0], traj2[:, 1], label='Mass 2')
        plt.title('Pressure Gradient Gravity Orbital Simulation')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

# Example usage
def main():
    # Initial conditions: two masses, initial velocities
    M1, M2 = 1.0, 1.0
    v1 = [0.1, 0.1]   # Initial velocity for mass 1
    v2 = [-0.1, -0.1] # Initial velocity for mass 2

    sim = PressureGravitySimulation(M1, M2, v1, v2)
    traj1, traj2 = sim.simulate()
    sim.plot_orbits(traj1, traj2)

if __name__ == "__main__":
    main()
