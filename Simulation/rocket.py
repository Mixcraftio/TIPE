from scipy.interpolate import interp1d
import json

with open("Motors/motors.json") as f:
    rocket_motors = json.loads(f.read())

class Rocket:
    def __init__(self, rocket_mass, motor_name, open_para, rocket_projected_surface, para_projected_surface, rocket_drag_coefficient=0.7, para_drag_coefficient=1, gisement=45, site=80):

        if motor_name not in rocket_motors:
            raise ValueError(f"Motor '{motor_name}' not found in the available motor list : {list(rocket_motors.keys())}")
        self.thrust_time = rocket_motors[motor_name]["thrust_time"]
        self.thrust_force = rocket_motors[motor_name]["thrust_force"]
        self.thrust = interp1d(self.thrust_time, self.thrust_force) # Interpolate the thrust curve

        self.rocket_mass = rocket_mass # Mass of the motorless rocket
        self.m = rocket_mass + rocket_motors[motor_name]["mass"] # Initial mass of the rocket
        self.casing_mass = rocket_motors[motor_name]["casing_mass"]
        self.propellant_mass = rocket_motors[motor_name]["propellant_mass"]
        self.propellant = interp1d(self.thrust_time, self.propellant_mass) # Interpolate the mass curve

        self.S = rocket_projected_surface # Cross-sectional area
        self.Cx = rocket_drag_coefficient # Drag coefficient
        self.t_open_para = open_para
        self.Sp = para_projected_surface # Cross-sectional parachute area
        self.Cp = para_drag_coefficient # Parachute drag coefficient

        self.gisement = gisement # Initial azimuth angle (in degrees)
        self.site = site # Initial elevation angle (in degrees)

    def Thrust(self, t):
        if t <= self.thrust_time[-1]:
            return self.thrust(t)
        return 0
    
    def Mass(self, t):
        if t <= self.thrust_time[-1]:
            return self.rocket_mass + self.casing_mass + self.propellant(t)
        return self.rocket_mass + self.casing_mass
