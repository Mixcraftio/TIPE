
"""Runs a simulation using the simulation module"""

import simulation as sim
import matplotlib.pyplot as plt



par=[]; res=[]; param="Cx"
# for h in [0.001 + 0.001 * i for i in range(1000)]:
# for Cx in [0.5 + 0.05 * i for i in range(8)]:
# for elevation in [75 + 1 * i for i in range(10)]:
for v in [True]:
    # Definition et creation de rocket
    # Hellfire
    motor = sim.Motor(motor_name="Pro54-5G WT")
    aero = sim.Aerodynamics(rocket_surface=0.008878, para_surface=1.99)
    rocket = sim.Rocket(rocket_mass=7.400, motor=motor, aerodynamics=aero, open_para=16)
    # Karlavagnen
    # motor = sim.Motor(motor_name="Pro54-5G C")
    # aero = sim.Aerodynamics(rocket_surface=0.008878, para_surface=1.99, rocket_drag=0.85)
    # rocket = sim.Rocket(rocket_mass=7.730, motor=motor, aerodynamics=aero, open_para=16.5)
    # Mistral
    # motor = sim.Motor(motor_name="Pro24-6G BS")
    # aero = sim.Aerodynamics(rocket_surface=0.003167, para_surface=0.44, rocket_drag=0.6)
    # rocket = sim.Rocket(rocket_mass=1.500, motor=motor, aerodynamics=aero, open_para=8)

    # Creation de l'objet simulation
    simulation = sim.SimulationEuler(rocket, integrator="RK4-SecondOrder", simulation_duration=200) # Fusex
    # sim = SimulationQuaternion(rocket, simulation_duration=30) # Minif

    # Execution de la simulation
    simulation.run_simulation()

    # Analyse des resultats
    analysis = sim.DataAnalysis(simulation)
    apogee, vnorm, vmean = analysis.simulation_report()
    analysis.graph_trajectory()
    # analysis.export_trajectory()

    # par.append(h)
    # par.append(Cx)
    # par.append(elevation)
    # res.append(apogee)

# plt.title(f"apogee=f({param})")
# plt.xlabel(param)
# plt.ylabel("apogee (m)")
# plt.plot(par, res, ".-", label="apogée")
# plt.show()