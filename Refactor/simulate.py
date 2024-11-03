from rocket import Rocket
from simulation import *
import matplotlib.pyplot as plt

# Prepare le graphe
fig = plt.figure()
fig.suptitle("Trajectory Simulation")
ax = fig.add_subplot(projection='3d')

for Cx in [0.6,0.65,0.7,0.75,0.8]:
    # Define and create rocket
    # Hellfire
    rocket = Rocket(rocket_mass=7.900,
                    projected_surface=0.008854,
                    motor_name="Pro54-5G WT",
                    drag_coefficient=Cx)
    # Karlavagnen
    # rocket = Rocket(rocket_mass=7.730,
    #                 projected_surface=0.008854,
    #                 motor_name="Pro54-5G C",
    #                 drag_coefficient=0.85)

    # Create simulation
    sim = SimulationEuler(rocket, simulation_duration=30)
    # sim = SimulationQuaternion(rocket, simulation_duration=30)

    # Run the simulation
    sim.run_simulation()

    sim.plot_trajectory(ax)
    sim.export_data()

    # Altitude maximale
    apogee=max(sim.trajectory[:,2])
    print(apogee)
    # Vitesse moyenne de montée
    # v=np.array([(np.linalg.norm(sim.trajectory[i])-np.linalg.norm(sim.trajectory[i-1]))/sim.h for i in range(1,len(sim.trajectory)//2)])
    # print(v.mean())

# Montrer le graphe
plt.show()