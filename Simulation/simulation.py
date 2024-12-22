from math import ceil
import numpy as np
import quaternion as quat

# Environment variables
g = 9.81  # Intensité de la pesanteur (m/s^2)
rho = 1.225  # Masse volumique de l'air (kg/m^3)

class SimulationEuler:
    def __init__(self, rocket, simulation_duration, fps=60):
        self.rocket = rocket # Une instance de l'objet rocket
        self.simulation_duration = simulation_duration
        self.fps = fps
        self.h = 1 / self.fps # Calcul du pas d'intégration
        self.simuNPoints = ceil(simulation_duration / self.h) # Nombre de points de la simulation
        self.time = 0

        self.trajectory = np.zeros((self.simuNPoints, 3), dtype=np.float64)
        self.velocity = np.zeros((self.simuNPoints, 3), dtype=np.float64)
        self.euler_angles = np.zeros((self.simuNPoints, 3), dtype=np.float64)

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
        resistance = -0.5 * rho * self.rocket.S * self.rocket.Cx * np.square(velocity)
        if t >= self.rocket.t_open_para:
            resistancePara = 0.5 * rho * self.rocket.Sp * self.rocket.Cp * np.array([0,0,velocity[2]**2])
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
        for i in range(1, self.simuNPoints):
            self.time_index = i
            self.velocity[i] = self.RK4_SingleStep(self.acceleration, self.velocity[i-1], self.time, self.h)
            position = self.Euler_SingleStep(self.velocity[i], self.trajectory[i-1], self.h)
            if position[2] < 0:
                self.velocity[i] = np.array([0, 0, 0])
                self.trajectory[i] = np.array([self.trajectory[i-1][0], self.trajectory[i-1][1], 0])
            else:
                self.trajectory[i] = position
                self.euler_angles[i] = self.updateRotation(i, self.theta, self.phi)
            self.time += self.h

        return self.trajectory, self.euler_angles

    def plot_trajectory(self, ax):
        x, y, z = self.trajectory.T
        thrust_end = ceil(self.rocket.thrust_time[-1] / self.h)

        ax.plot3D(x[:thrust_end+1], y[:thrust_end+1], z[:thrust_end+1], 'r')
        ax.plot3D(x[thrust_end:], y[thrust_end:], z[thrust_end:], 'g')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    
    def export_data(self, filename="SIM-EULER.txt"):
        export = ""
        export += f"{self.simulation_duration};{self.h};{self.rocket.thrust_time[-1]}\n"

        # Reformatage des angles d'Euler
        e2 = [np.array([i[2], i[1], i[0]]) for i in self.euler_angles]

        for r in range(self.simuNPoints):
            # Export de la trajectoire
            export += ";".join(map(str, self.trajectory[r])) + ";"

            # Export des angles d'Euler
            export += ";".join(map(str, e2[r])) + "\n"

        # Ecriture dans le fichier
        with open(filename, "w") as f:
            f.write(export)



class SimulationQuaternion:
    def __init__(self, rocket, simulation_duration, fps=60):
        self.rocket = rocket # Une instance de l'objet rocket
        self.simulation_duration = simulation_duration
        self.fps = fps
        self.h = 1 / self.fps # Calcul du pas d'intégration
        self.simuNPoints = ceil(simulation_duration / self.h) # Nombre de points de la simulation
        self.time = 0

        self.trajectory = np.zeros((self.simuNPoints, 3), dtype=np.float64)
        self.velocity = np.zeros((self.simuNPoints, 3), dtype=np.float64)
        self.q = np.zeros(self.simuNPoints, dtype=np.quaternion)

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
        for i in range(1, self.simuNPoints):
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
    
    def plot_trajectory(self,ax):
        x, y, z = self.trajectory.T
        thrust_end = ceil(self.rocket.thrust_time[-1] / self.h)

        ax.plot3D(x[:thrust_end], y[:thrust_end], z[:thrust_end], 'r')
        ax.plot3D(x[thrust_end:], y[thrust_end:], z[thrust_end:], 'g')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    def export_data(self, filename="SIM-QUAT.txt"):
        export = ""
        export += f"{self.simulation_duration};{self.h};{self.rocket.thrust_time[-1]}\n"

        # Conversion des quaternions en tableaux de float
        q2 = [quat.as_float_array(qi) for qi in self.q]

        for r in range(self.simuNPoints):
            # Export de la trajectoire
            export += ";".join(map(str, self.trajectory[r])) + ";"

            # Export des quaternions
            export += ";".join(map(str, q2[r])) + "\n"

        # Ecriture dans le fichier
        with open(filename, "w") as f:
            f.write(export)

