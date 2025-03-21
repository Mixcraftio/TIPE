
"""Runs a simulation using the simulation module"""

from math import ceil
import simulation as sim
import numpy as np
import matplotlib.pyplot as plt

# Prepare le graphe
fig = plt.figure()
fig.suptitle("Trajectory Simulation")
ax = fig.add_subplot(projection='3d')

# for Cx in [0.6,0.65,0.7,0.75,0.8]:
for Cx in [0.7]:
    # Definition et creation de rocket
    # Hellfire
    motor = sim.Motor(motor_name="Pro54-5G WT")
    aero = sim.Aerodynamics(rocket_surface=0.008854, para_surface=1.99)
    rocket = sim.Rocket(rocket_mass=7.400, motor=motor, open_para=16, aerodynamics=aero)
    # # Karlavagnen
    # motor = sim.Motor(motor_name="Pro54-5G C")
    # aero = sim.Aerodynamics(rocket_surface=0.008854, para_surface=1.99, rocket_drag=0.85)
    # rocket = sim.Rocket(rocket_mass=7.730, motor=motor, open_para=16.5, aerodynamics=aero)
    # Mistral
    # motor = sim.Motor(motor_name="Pro24-6G BS")
    # aero = sim.Aerodynamics(rocket_surface=0.003167, para_surface=0.44, rocket_drag=0.6)
    # rocket = sim.Rocket(rocket_mass=1.500, motor=motor, open_para=8, aerodynamics=aero)

    # Creation de l'objet simulation
    simulation = sim.SimulationEuler(rocket, simulation_duration=200)
    # sim = SimulationQuaternion(rocket, simulation_duration=30)

    # Execution de la simulation
    simulation.run_simulation()

    # Plot trajectory
    x, y, z = simulation.trajectory.T
    thrust_end = ceil(simulation.rocket.motor.thrust_time[-1] / simulation.h)
    ax.plot3D(x[:thrust_end], y[:thrust_end], z[:thrust_end], 'r')
    ax.plot3D(x[thrust_end:], y[thrust_end:], z[thrust_end:], 'g')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    simulation.export_data()

    # ---------- Calculs ----------
    # Altitude maximale
    apogee=max(simulation.trajectory[:,2])
    print("Apogée :", apogee)
    # Vitesse moyenne de montée
    index_apogee = np.argmax([np.linalg.norm(e) for e in simulation.velocity])
    print("Vitesse de montée moyenne :", simulation.velocity[:index_apogee].mean())
    # Vitesse
    v=np.array([(np.linalg.norm(simulation.trajectory[i])-np.linalg.norm(simulation.trajectory[i-1]))/simulation.h for i in range(1,len(simulation.trajectory))])
    # Vitesse z
    vz=simulation.velocity.T[2]
plt.show()

time=[simulation.h*i for i in range(simulation.simuNPoints)]
plt.title("Vitesse simulée de la fusée")
plt.plot(time[:-1],v,label="Vitesse 3axes")
plt.plot(time,vz,label="Vitesse en z")
plt.xlabel("Vitesse (m.s-1)")
plt.ylabel("Temps (ms)")
plt.legend()
plt.grid()
plt.show()
