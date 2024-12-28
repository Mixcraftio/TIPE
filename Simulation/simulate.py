from rocket import Rocket
from simulation import *
import matplotlib.pyplot as plt

# Prepare le graphe
fig = plt.figure()
fig.suptitle("Trajectory Simulation")
ax = fig.add_subplot(projection='3d')

# for Cx in [0.6,0.65,0.7,0.75,0.8]:
for Cx in [0.7]:
    # Definition et creation de rocket
    # Hellfire
    rocket = Rocket(rocket_mass=7.400,
                    motor_name="Pro54-5G WT",
                    open_para=16,
                    rocket_projected_surface=0.008854,
                    para_projected_surface=1.99,
                    # para_projected_surface=1.99/2,
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

    # Creation de l'objet simulation
    sim = SimulationEuler(rocket, simulation_duration=200)
    # sim = SimulationQuaternion(rocket, simulation_duration=30)

    # Execution de la simulation
    sim.run_simulation()

    sim.plot_trajectory(ax)
    plt.show()
    sim.export_data()


    # ---------- Calculs ----------
    # Altitude maximale
    apogee=max(sim.trajectory[:,2])
    print("Apogée :", apogee)
    # Vitesse moyenne de montée
    print("Vitesse de montée moyenne :", sim.velocity[:np.argmax([np.linalg.norm(e) for e in sim.velocity])].mean())
    # Vitesse
    # v=np.array([(np.linalg.norm(sim.trajectory[i])-np.linalg.norm(sim.trajectory[i-1]))/sim.h for i in range(1,len(sim.trajectory))])
    # Vitesse z
    # plt.plot(range(len(sim.velocity)),sim.velocity)
    # plt.show()