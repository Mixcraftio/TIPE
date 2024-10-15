from scipy.interpolate import interp1d

rocket_motors = {
    "Pro54-5G WT": {
        "mass": 1.685,
        "thrust_time": [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.55, 1.6, 1.62, 1.64, 1.66, 1.67, 1.68, 1.69, 1.7],
        "thrust_force": [0, 492.25, 1369.46, 1236.01, 1279.47, 1311.39, 1331.39, 1304.08, 1280.62, 1249.86, 1217.94, 1199.29, 1158.77, 1112.56, 941.81, 726.07, 559.17, 399.95, 317.66, 247.28, 198.05, 67.3, 0]
    },
    "Pro54-5G C": {
        "mass": 1.685,
        "thrust_time": [0, 0.01, 0.1, 0.12, 0.26, 0.71, 1.28, 2.05, 2.41, 2.83, 3.25, 3.65, 3.8, 4, 4.1, 4.19, 4.31, 4.41, 4.52, 4.6, 4.65, 4.67, 4.68],
        "thrust_force": [27, 402.4, 1286, 1257, 1042, 1027, 998.4, 901.4, 849.6, 763.5, 707.1, 655.1, 651.7, 624.1, 601.3, 536.2, 415.7, 270.2, 140.2, 76.9, 54.9, 40.2, 0]
    }
}


class Rocket:
    def __init__(self, rocket_mass, projected_surface, motor_name, drag_coefficient=0.7, gisement=45, site=80):
        if motor_name not in rocket_motors:
            raise ValueError(f"Motor '{motor_name}' not found in the available motor list : {list(rocket_motors.keys())}")
        self.thrust_time = rocket_motors[motor_name]["thrust_time"]
        self.thrust_force = rocket_motors[motor_name]["thrust_force"]
        self.thrust = interp1d(self.thrust_time, self.thrust_force)  # Interpolate the thrust curve
        self.m = rocket_mass+rocket_motors[motor_name]["mass"] # Mass of the rocket
        self.Cx = drag_coefficient  # Drag coefficient
        self.S = projected_surface  # Cross-sectional area
        self.gisement = gisement  # Initial azimuth angle (in degrees)
        self.site = site  # Initial elevation angle (in degrees)

    def Thrust(self, t):
        if t <= self.thrust_time[-1]:
            return self.thrust(t)
        return 0
