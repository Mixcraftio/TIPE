
"""Runs a simulation using the simulation module"""

import simulation as sim
import matplotlib.pyplot as plt

# for Cx in [0.6,0.65,0.7,0.75,0.8]:
# for Cx in [0.7]:
x=[]; res=[]
for fps in [10*i for i in range(1,10)]+[100*i for i in range(1,10)]:
# for fps in [10]:
    rocket_drag = 0.7
    # Definition et creation de rocket
    # Hellfire
    motor = sim.Motor(motor_name="Pro54-5G WT")
    aero = sim.Aerodynamics(rocket_surface=0.008854, para_surface=1.99, rocket_drag=rocket_drag)
    rocket = sim.Rocket(rocket_mass=7.400, motor=motor, aerodynamics=aero, open_para=16)
    # # Karlavagnen
    # motor = sim.Motor(motor_name="Pro54-5G C")
    # aero = sim.Aerodynamics(rocket_surface=0.008854, para_surface=1.99, rocket_drag=0.85)
    # rocket = sim.Rocket(rocket_mass=7.730, motor=motor, aerodynamics=aero, open_para=16.5)
    # Mistral
    # motor = sim.Motor(motor_name="Pro24-6G BS")
    # aero = sim.Aerodynamics(rocket_surface=0.003167, para_surface=0.44, rocket_drag=0.6)
    # rocket = sim.Rocket(rocket_mass=1.500, motor=motor, aerodynamics=aero, open_para=8)

    # Creation de l'objet simulation
    simulation = sim.SimulationEuler(rocket, integrator="RK4-SecondOrder", simulation_duration=200, fps=fps) # Fusex
    # sim = SimulationQuaternion(rocket, simulation_duration=30) # Minif

    # Execution de la simulation
    simulation.run_simulation()

    # Analyse des résultats
    analysis = sim.DataAnalysis(simulation)
    apogee, vnorm, vmean = analysis.simulation_report()
    # analysis.graph_trajectory()
    # analysis.export_trajectory()

    x.append(1/fps)
    res.append(apogee)

plt.title("apogee=f(h)")
plt.xlabel("h (s)")
plt.ylabel("apogee (m)")
plt.plot(x, res, ".-", label="apogée")
plt.show()
# time=[simulation.h*i for i in range(simulation.simulation_steps)]
# plt.title("Vitesse simulée de la fusée")
# plt.plot(time,vnorm,label="Vitesse 3axes")
# plt.plot(time,simulation.velocity.T[2],label="Vitesse en z")
# plt.xlabel("Vitesse (m.s-1)")
# plt.ylabel("Temps (ms)")
# plt.legend()
# plt.grid()
# plt.show()
