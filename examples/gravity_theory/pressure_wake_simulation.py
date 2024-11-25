import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class PressureWakeSimulation:
    def __init__(self, 
                 m1=1.0, 
                 m2=1.0, 
                 k_wake=0.1,  # Pressure relief constant
                 initial_sep=1.0):
        """
        Initialize two-body pressure wake simulation
        
        Args:
        m1, m2: Masses
        k_wake: Pressure wake coefficient
        initial_sep: Initial separation
        """
        self.m1 = m1
        self.m2 = m2
        self.k_wake = k_wake
        self.initial_sep = initial_sep

    def pressure_wake_dynamics(self, state, t):
        """
        Compute dynamics based on pressure wake model
        
        State vector: [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
        """
        x1, y1, vx1, vy1, x2, y2, vx2, vy2 = state
        
        # Compute relative position and distance
        dx = x2 - x1
        dy = y2 - y1
        r = np.sqrt(dx**2 + dy**2)
        
        # Pressure wake force calculation
        wake_coefficient1 = self.k_wake * self.m1
        wake_coefficient2 = self.k_wake * self.m2
        
        # Compute acceleration components
        ax1 = (wake_coefficient2 / r**2) * (dx/r)
        ay1 = (wake_coefficient2 / r**2) * (dy/r)
        ax2 = -(wake_coefficient1 / r**2) * (dx/r)
        ay2 = -(wake_coefficient1 / r**2) * (dy/r)
        
        return [
            vx1, vy1,  # Velocity components
            ax1, ay1,  # Acceleration components
            vx2, vy2,  # Second body velocities
            ax2, ay2   # Second body accelerations
        ]

    def simulate(self, duration=10, dt=0.01):
        """
        Run full simulation
        
        Returns trajectory data
        """
        # Initial conditions
        initial_state = [
            0.0, 0.0,    # Initial position of mass 1
            0.1, 0.1,    # Initial velocity of mass 1
            self.initial_sep, 0.0,  # Initial position of mass 2
            -0.1, -0.1   # Initial velocity of mass 2
        ]
        
        # Time array
        t = np.linspace(0, duration, int(duration/dt))
        
        # Solve differential equations
        solution = odeint(
            self.pressure_wake_dynamics, 
            initial_state, 
            t
        )
        
        return t, solution

    def plot_trajectories(self, t, solution):
        """
        Visualize simulation results
        """
        plt.figure(figsize=(12, 8))
        
        # Plot positions
        plt.subplot(2, 1, 1)
        plt.plot(solution[:, 0], solution[:, 1], label='Mass 1 Path')
        plt.plot(solution[:, 4], solution[:, 5], label='Mass 2 Path')
        plt.title('Pressure Wake Trajectory')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True)
        
        # Plot velocities
        plt.subplot(2, 1, 2)
        plt.plot(t, solution[:, 2], label='Mass 1 X Velocity')
        plt.plot(t, solution[:, 3], label='Mass 1 Y Velocity')
        plt.plot(t, solution[:, 6], label='Mass 2 X Velocity')
        plt.plot(t, solution[:, 7], label='Mass 2 Y Velocity')
        plt.title('Velocity Evolution')
        plt.xlabel('Time')
        plt.ylabel('Velocity')
        plt.legend()
        plt.tight_layout()
        plt.show()

def main():
    # Create simulation instance
    sim = PressureWakeSimulation(
        m1=1.0,      # Mass of first body
        m2=1.0,      # Mass of second body
        k_wake=0.1,  # Pressure wake coefficient
        initial_sep=1.0  # Initial separation
    )
    
    # Run simulation
    t, solution = sim.simulate(duration=10, dt=0.01)
    
    # Visualize results
    sim.plot_trajectories(t, solution)

if __name__ == "__main__":
    main()
