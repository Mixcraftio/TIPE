from scipy.interpolate import interp1d
import json

with open("Motors/motors.json") as f:
    rocket_motors = json.loads(f.read())

class Rocket:
    def __init__(self, rocket_mass, motor_name, open_para, rocket_projected_surface, para_projected_surface, rocket_drag_coefficient=0.7, para_drag_coefficient=0.8, gisement=45, site=80):

        if motor_name not in rocket_motors:
            raise ValueError(f"Motor '{motor_name}' not found in the available motor list : {list(rocket_motors.keys())}")
        self.thrust_time = rocket_motors[motor_name]["thrust_time"]
        self.thrust_force = rocket_motors[motor_name]["thrust_force"]
        self.thrust = interp1d(self.thrust_time, self.thrust_force) # Interpolatation de la courbe de poussée

        self.m = rocket_mass + rocket_motors[motor_name]["mass"] # Masse de la fusee + propu plein
        self.rocket_mass = rocket_mass # Masse de la fusee seule
        self.casing_mass = rocket_motors[motor_name]["casing_mass"]
        self.propellant_mass = rocket_motors[motor_name]["propellant_mass"]
        self.propellant = interp1d(self.thrust_time, self.propellant_mass) # Interpolation de la courbe de masse du combustible

        self.S = rocket_projected_surface # Maitre-couple fusee
        self.Cx = rocket_drag_coefficient # Coefficient de resistance a l'air fusee
        self.t_open_para = open_para
        self.Sp = para_projected_surface # Maitre-couple parachute
        self.Cp = para_drag_coefficient # Coefficient de resistance a l'air parachute

        self.gisement = gisement # Angle azimuth initial (degres)
        self.site = site # Angle elevation initial (degres)

    def Thrust(self, t):
        if t <= self.thrust_time[-1]:
            return self.thrust(t)
        return 0
    
    def Mass(self, t):
        if t <= self.thrust_time[-1]:
            return self.rocket_mass + self.casing_mass + self.propellant(t)
        return self.rocket_mass + self.casing_mass
