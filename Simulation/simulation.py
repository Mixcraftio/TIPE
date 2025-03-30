
"""Simulation module for model rockets"""

import json
from math import ceil
import numpy as np
import quaternion as quat
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# Variables liees a l'environement
g = 9.81  # Intensité de la pesanteur (m/s^2)
rho = 1.225  # Masse volumique de l'air (kg/m^3)

with open("Motors/motors.json", encoding="UTF-8") as f:
    rocket_motors = json.loads(f.read())


class Motor:
    """Gère les propriétés du moteur de la fusée."""
    def __init__(self, motor_name: str):
        if motor_name not in rocket_motors:
            raise ValueError(f"Le moteur '{motor_name}' n'est pas disponible. Moteurs disponibles : {list(rocket_motors.keys())}")

        motor_data = rocket_motors[motor_name]
        self.thrust_time = motor_data["thrust_time"]
        self.thrust_force = motor_data["thrust_force"]
        self.thrust = interp1d(self.thrust_time, self.thrust_force)  # Interpolation de la courbe de poussée
        self.mass = motor_data["mass"]
        self.casing_mass = motor_data["casing_mass"]
        self.propellant_mass = motor_data["propellant_mass"]
        self.propellant = interp1d(self.thrust_time, self.propellant_mass)  # Interpolation de la courbe de masse du carburant

    def get_thrust(self, t: float) -> float:
        """Renvoie la poussée au temps t."""
        return self.thrust(t) if t <= self.thrust_time[-1] else 0

    def get_propellant_mass(self, t: float) -> float:
        """Renvoie la masse de carburant restante au temps t."""
        return self.propellant(t) if t <= self.thrust_time[-1] else 0


class Aerodynamics:
    """Gère les propriétés aérodynamiques de la fusée."""
    def __init__(self, rocket_surface: float, para_surface: float, rocket_drag: float = 0.7, para_drag: float = 0.8):
        self.rocket_surface = rocket_surface  # Surface projetée de la fusée
        self.para_surface = para_surface  # Surface projetée du parachute
        self.rocket_drag = rocket_drag  # Coefficient de traînée de la fusée
        self.para_drag = para_drag  # Coefficient de traînée du parachute


class Rocket:
    """Définit une fusée pour une simulation."""
    def __init__(self, rocket_mass: float, motor: Motor, aerodynamics: Aerodynamics, open_para: float, gisement: float = 45, site: float = 80):
        self.rocket_mass = rocket_mass  # Masse à vide de la fusée
        self.motor = motor  # Objet moteur
        self.m = rocket_mass + motor.mass  # Masse totale (fusée + moteur)
        self.open_para = open_para  # Temps d'ouverture du parachute
        self.aerodynamics = aerodynamics  # Propriétés aérodynamiques
        self.gisement = gisement  # Angle d'azimut initial (degrés)
        self.site = site  # Angle d'élévation initial (degrés)

    def Thrust(self, t: float) -> float:
        """Renvoie la valeur de la poussée au temps t."""
        return self.motor.get_thrust(t)

    def Mass(self, t: float) -> float:
        """Renvoie la masse totale au temps t."""
        return self.rocket_mass + self.motor.casing_mass + self.motor.get_propellant_mass(t)



class SimulationEuler:
    def __init__(self, rocket: Rocket, integrator = "RK4-SecondOrder", simulation_duration: float = 200, fps: int = 60):
        self.rocket = rocket # Une instance de l'objet rocket
        self.integrator = integrator # La méthode d'intégration utilisée
        self.simulation_duration = simulation_duration
        self.h = 1 / fps # Calcul du pas d'intégration
        self.simulation_steps = ceil(simulation_duration / self.h) # Nombre de points de la simulation
        self.time = 0
        self.time_index = 0

        self.trajectory = np.zeros((self.simulation_steps, 3), dtype=np.float64)
        self.velocity = np.zeros((self.simulation_steps, 3), dtype=np.float64)
        self.euler_angles = np.zeros((self.simulation_steps, 3), dtype=np.float64)

        self.theta = self.rocket.gisement * np.pi / 180
        self.phi = (90 - self.rocket.site) * np.pi / 180
        self.euler_angles[0] = np.array([self.theta, self.phi, 0])

    def RK4_SingleStep(self, accel, velocity, t, h):
        f1 = accel(t, velocity)
        f2 = accel(t + h / 2, velocity + f1 * h / 2)
        f3 = accel(t + h / 2, velocity + f2 * h / 2)
        f4 = accel(t + h, velocity + f3 * h)
        final_velocity = velocity + (f1 + 2 * f2 + 2 * f3 + f4) * (h / 6)
        return final_velocity

    def Euler_SingleStep(self, velocity, position, h):
        final_position = position + velocity * h
        return final_position

    def RK4_SingleStep_SecondOrder(self, accel, velocity, position, t, h):
        k1 = accel(t, velocity)
        k2 = accel(t + h / 2, velocity + k1 * h / 2)
        k3 = accel(t + h / 2, velocity + k2 * h / 2)
        k4 = accel(t + h, velocity + k3 * h)
        final_position = position + velocity * h + (k1 + k2 + k3) * (h ** 2 / 6)
        final_velocity = velocity + (k1 + 2 * k2 + 2 * k3 + k4) * (h / 6)
        return final_position, final_velocity

    def acceleration(self, t, velocity):
        euler_angles = self.euler_angles[self.time_index - 1]
        theta = euler_angles[0]
        phi = euler_angles[1]
        # Poussée propulseur
        thrust_z = self.rocket.Thrust(t) * np.cos(phi)
        thrust_xy = self.rocket.Thrust(t) * np.sin(phi)
        thrust_y = thrust_xy * np.cos(theta)
        thrust_x = thrust_xy * np.sin(theta)

        thrust = np.array([thrust_x, thrust_y, thrust_z])

        # Résistance de l'air
        resistance = -0.5 * rho * self.rocket.aerodynamics.rocket_surface * self.rocket.aerodynamics.rocket_drag * np.square(velocity)
        if t >= self.rocket.open_para:
            resistancePara = 0.5 * rho * self.rocket.aerodynamics.para_surface * self.rocket.aerodynamics.para_drag * np.array([0,0,velocity[2]**2])
        else:
            resistancePara = 0

        # Poids
        # weight = np.array([0, 0, -self.rocket.m * g])
        weight = np.array([0, 0, -self.rocket.Mass(t) * g])

        # Résultante des forces
        forces = thrust + resistance + resistancePara + weight
        # accel = forces / self.rocket.m
        accel = forces / self.rocket.Mass(t)
        return accel

    def updateRotation(self, i, theta, phi):
        def dist(a, b, ref="xyz"):
            ax, ay, az = a[0], a[1], a[2]
            bx, by, bz = b[0], b[1], b[2]
            if ref == "xy":
                return np.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
            return np.sqrt((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2)

        if dist(self.trajectory[i-1], self.trajectory[i], "xy") != 0:
            cos_theta = (self.trajectory[i][0] - self.trajectory[i-1][0]) / dist(self.trajectory[i-1], self.trajectory[i], "xy")
            theta = np.arccos(cos_theta) if self.trajectory[i][2] >= 0 else -np.arccos(cos_theta)

            cos_phi = (self.trajectory[i][2] - self.trajectory[i-1][2]) / dist(self.trajectory[i-1], self.trajectory[i])
            phi = np.arccos(cos_phi)
        return np.array([theta, phi, 0])

    def run_simulation(self):
        # Loop principal
        for i in range(1, self.simulation_steps):
            self.time_index = i
            if self.integrator == "RK4-SecondOrder":
                position, self.velocity[i] = self.RK4_SingleStep_SecondOrder(self.acceleration, self.velocity[i-1], self.trajectory[i-1], self.time, self.h)
            elif self.integrator == "RK4-Euler":
                self.velocity[i] = self.RK4_SingleStep(self.acceleration, self.velocity[i-1], self.time, self.h)
                position = self.Euler_SingleStep(self.velocity[i], self.trajectory[i-1], self.h)
            else:
                position, self.velocity[i] = self.RK4_SingleStep_SecondOrder(self.acceleration, self.velocity[i-1], self.trajectory[i-1], self.time, self.h)
            if position[2] < 0:
                self.velocity[i] = np.array([0, 0, 0])
                self.trajectory[i] = np.array([self.trajectory[i-1][0], self.trajectory[i-1][1], 0])
            else:
                self.trajectory[i] = position
                self.euler_angles[i] = self.updateRotation(i, self.theta, self.phi)
            self.time += self.h

        return self.trajectory, self.euler_angles


class SimulationQuaternion:
    def __init__(self, rocket: Rocket, simulation_duration: float = 200, fps: int = 60):
        self.rocket = rocket # Une instance de l'objet rocket
        self.simulation_duration = simulation_duration
        self.h = 1 / fps # Calcul du pas d'intégration
        self.simulation_steps = ceil(simulation_duration / self.h) # Nombre de points de la simulation
        self.time = 0
        self.time_index = 0

        self.trajectory = np.zeros((self.simulation_steps, 3), dtype=np.float64)
        self.velocity = np.zeros((self.simulation_steps, 3), dtype=np.float64)
        self.q = np.zeros(self.simulation_steps, dtype=np.quaternion)

        theta = self.rocket.gisement * np.pi / 180
        phi = (90 - self.rocket.site) * np.pi / 180
        self.q[0] = quat.from_euler_angles([theta, phi, 0])
        self.wi = np.array([0.0, 0.0, 0.0])

    def RK4_SingleStep(self, accel, velocity, t, h):
        f1 = accel(t, velocity)
        f2 = accel(t + h / 2, velocity + f1 * h / 2)
        f3 = accel(t + h / 2, velocity + f2 * h / 2)
        f4 = accel(t + h, velocity + f3 * h)
        final_velocity = velocity + (f1 + 2 * f2 + 2 * f3 + f4) * (h / 6)
        return final_velocity

    def Euler_SingleStep(self, velocity, position, h):
        final_position = position + velocity * h
        return final_position

    def acceleration(self, t, velocity):
        euler_angles = quat.as_euler_angles(self.q[self.time_index - 1])
        theta = euler_angles[0]
        phi = euler_angles[1]
        # Poussée propulseur
        thrust_z = self.rocket.Thrust(t) * np.cos(phi)
        thrust_xy = self.rocket.Thrust(t) * np.sin(phi)
        thrust_y = thrust_xy * np.cos(theta)
        thrust_x = thrust_xy * np.sin(theta)

        thrust = np.array([thrust_x, thrust_y, thrust_z])

        # Résistance de l'air
        resistance = -0.5 * rho * self.rocket.S * self.rocket.Cx * np.square(velocity)

        # Poids
        # weight = np.array([0, 0, -self.rocket.m * g])
        weight = np.array([0, 0, -self.rocket.Mass(t) * g])

        # Résultante des forces
        forces = thrust + resistance + weight
        # accel = forces / self.rocket.m
        accel = forces / self.rocket.Mass(t)
        return accel

    def update_rotation(self, eq, qi, wi, h):
        w = self.RK4_SingleStep(eq, wi, 0, h)
        wx=w[0]; wy=w[1]; wz=w[2]
        OMEGA=np.array([[0,-wx,-wy,-wz],[wx,0,wz,-wy],[wy,-wz,0,wx],[wz,wy,-wx,0]])
        qi = quat.as_float_array(qi)
        q_dot = 0.5 * np.matmul(OMEGA, qi)
        q = self.Euler_SingleStep(q_dot, qi, h)
        return q, w

    def self_acceleration(self, t, velocity):
        return np.array([0.0, 0.0, 0.0])

    def run_simulation(self):
        # Loop principal
        for i in range(1, self.simulation_steps):
            self.time_index = i
            self.velocity[i] = self.RK4_SingleStep(self.acceleration, self.velocity[i-1], self.time, self.h)
            position = self.Euler_SingleStep(self.velocity[i], self.trajectory[i-1], self.h)
            if position[2] < 0:
                self.velocity[i] = np.array([0, 0, 0])
                self.trajectory[i] = np.array([self.trajectory[i-1][0], self.trajectory[i-1][1], 0])
            else:
                self.trajectory[i] = position
                self.q[i], self.wi = self.update_rotation(self.self_acceleration, self.q[i-1], self.wi, self.h)
            self.time += self.h
        return self.trajectory, self.q


class DataAnalysis:
    def __init__(self, simulation):
        self.simulation = simulation

    def simulation_report(self):
        # Altitude maximale
        apogee = max(self.simulation.trajectory[:,2])
        print("Apogée :", apogee)
        # Norme de la vitesse
        vnorm = np.linalg.norm(self.simulation.velocity, axis=1)
        # Vitesse moyenne de montée
        index_apogee = np.argmax(vnorm)
        vmean = vnorm[:index_apogee].mean()
        print("Vitesse de montée moyenne :", vmean)
        return apogee, vnorm, vmean

    def graph_trajectory(self, filename="./OUT/Trajectory.svg"):
        """Trace, sauvegarde et montre le graphe de la trajectoire de la fusée."""
        fig = plt.figure()
        fig.suptitle("Trajectory Simulation")
        ax = fig.add_subplot(projection='3d')

        x, y, z = self.simulation.trajectory.T
        thrust_end = ceil(self.simulation.rocket.motor.thrust_time[-1] / self.simulation.h)
        ax.plot3D(x[:thrust_end], y[:thrust_end], z[:thrust_end], 'r', label="Trajectoire de la fusée")
        ax.plot3D(x[thrust_end:], y[thrust_end:], z[thrust_end:], 'g', label="Phase de poussée")
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')

        ax.legend()
        fig.savefig(filename)
        plt.show()

    def export_trajectory(self, filename="./OUT/SimulationTrajectory.csv"):
        """Exporte la trajectoire de la fusée en un csv."""
        export = ""

        if isinstance(self.simulation, SimulationEuler):
            export += "# timestamp (ms), traj_x (m), traj_y (m), traj_z (m), theta (rad), phi (rad), psi (rad) \n"
            rot = [np.array([i[2], i[1], i[0]]) for i in self.simulation.euler_angles]
        elif isinstance(self.simulation, SimulationQuaternion):
            export += "# timestamp (ms), traj_x (m), traj_y (m), traj_z (m), q1, q2, q3, q4 \n"
            rot = [quat.as_float_array(qi) for qi in self.simulation.q]
        else:
            rot = []

        for r in range(self.simulation.simulation_steps):
            # Export des timestamps
            export += str(self.simulation.h*r) + ","

            # Export de la trajectoire
            export += ",".join(map(str, self.simulation.trajectory[r])) + ","

            # Export des quaternions
            export += ",".join(map(str, rot[r])) + "\n"

        # Ecriture dans le fichier
        with open(filename, "w", encoding="UTF-8") as f:
            f.write(export)
