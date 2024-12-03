from rocket import Rocket
from simulation import *
import matplotlib.pyplot as plt

# Prepare le graphe
fig = plt.figure()
fig.suptitle("Trajectory Simulation")
ax = fig.add_subplot(projection='3d')

# for Cx in [0.6,0.65,0.7,0.75,0.8]:
for Cx in [0.7]:
    # Define and create rocket
    # Hellfire
    rocket = Rocket(rocket_mass=7.400,
                    motor_name="Pro54-5G WT",
                    open_para=16,
                    rocket_projected_surface=0.008854,
                    para_projected_surface=1.99,
                    rocket_drag_coefficient=Cx)
    # Karlavagnen
    # rocket = Rocket(rocket_mass=7.730,
    #                 motor_name="Pro54-5G C",
    #                 open_para=16.5,
    #                 rocket_projected_surface=0.008854,
    #                 para_projected_surface=1.99,
    #                 rocket_drag_coefficient=0.85)
    # Mistral
    # rocket = Rocket(rocket_mass=1.500,
    #                 motor_name="Pro24-6G BS",
    #                 open_para=8,
    #                 rocket_projected_surface=0.003167,
    #                 para_projected_surface=0.44,
    #                 rocket_drag_coefficient=0.6)

    # Create simulation
    sim = SimulationEuler(rocket, simulation_duration=200)
    # sim = SimulationQuaternion(rocket, simulation_duration=30)

    # Run the simulation
    sim.run_simulation()

    sim.plot_trajectory(ax)
    plt.show()
    sim.export_data()

    # Altitude maximale
    apogee=max(sim.trajectory[:,2])
    print("Apogée :", apogee)
    # Vitesse moyenne de montée
    # v=np.array([(np.linalg.norm(sim.trajectory[i])-np.linalg.norm(sim.trajectory[i-1]))/sim.h for i in range(1,len(sim.trajectory)//2)])
    # print(v.mean())
    # Vitesse
    # v=np.array([(np.linalg.norm(sim.trajectory[i])-np.linalg.norm(sim.trajectory[i-1]))/sim.h for i in range(1,len(sim.trajectory))])
    # Vitesse z
    # vz=np.array([(sim.trajectory[i][2]-sim.trajectory[i-1][2])/sim.h for i in range(1,len(sim.trajectory))])
    # plt.plot(range(len(vz)),vz)
    # plt.show()