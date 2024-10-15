from rocket import Rocket
from simulation import *

# Define and create rocket
rocket = Rocket(rocket_mass=9.032,
                projected_surface=0.008878, 
                motor_name="Pro54-5G WT",
                drag_coefficient=0.7)

# Create simulation
sim = SimulationEuler(rocket, simulation_duration=30)
# sim = SimulationQuaternion(rocket, simulation_duration=30)

# Run the simulation
sim.run_simulation()

sim.plot_trajectory()
sim.export_data()


# Vitesse moyenne de montée
# v=np.array([(np.linalg.norm(sim.trajectory[i])-np.linalg.norm(sim.trajectory[i-1]))/sim.h for i in range(1,len(sim.trajectory)//2)])
# print(v.mean())