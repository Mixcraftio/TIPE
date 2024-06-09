import matplotlib.pyplot as plt
import numpy as np
import quaternion as quat
from scipy.interpolate import interp1d
from math import ceil


# ----- Variables de simulation -----
# Choix temps
# simulationDuration=30 # Temps de simulation (en s)
# h=0.01 # Tranche de temps de la simulation (ti+1 - ti) (en s)
# simuNPoints=ceil(simulationDuration/h) # Nombres de points de simulation
# Choix points
simulationDuration=30 # Temps de simulation (en s)
simuNPoints=3000 # Nombres de points de simulation
h=simulationDuration/simuNPoints # Tranche de temps de la simulation (ti+1 - ti)

# Initialisation des variables python
trajecto=np.array([np.array([0.0,0.0,0.0]) for i in range(simuNPoints)])
q=np.array([np.quaternion(0,0,0,0) for i in range(simuNPoints)])
time=0

# Variables caractéristiques du milieu
g=9.81 # Intensité du champs de pesanteur ()
rho=1.225 # Masse volumique de l'air (en kg/m3)

# Variables caractéristiques de la fusée
speed=np.array([0,0,0]) # Vitesse initiale [(en m.s-1),(en m.s-1),(en m.s-1)]
m=7.8 # Masse totale de la fusée (en kg)
Cx=0.85 # Coefficient de frottements ()
S=0.008854 # Surface projetée du dessus (en m2)
theta=45 # Angle de lancer de la fusée (en deg)
elevation=80 # Angle d'élévation de la rampe (en deg)
phi=90-elevation
theta=theta*np.pi/180
phi=phi*np.pi/180
q[0]=quat.from_euler_angles(np.array([theta,phi,0]))
wi=np.array([0.0,0.0,0.0])

# Courbe de poussée du propulseur (en N)
thrustTime=[0,0.01,0.02,0.05,0.1,0.2,0.4,0.8,0.9,1,1.1,1.2,1.3,1.4,1.55,1.6,1.62,1.64,1.66,1.67,1.68,1.69,1.7]
thrustForce=[0,492.25,1369.46,1236.01,1279.47,1311.39,1331.39,1304.08,1280.62,1249.86,1217.94,1199.29,1158.77,1112.56,941.81,726.07,559.17,399.95,317.66,247.28,198.05,67.3,0]
Thrust=interp1d(thrustTime,thrustForce)
# -----------------------------------


def RK4_SingleStep(accel,vi,t,h):
    f1=accel(t,vi)
    f2=accel(t+h/2,vi+f1*h/2)
    f3=accel(t+h/2,vi+f2*h/2)
    f4=accel(t+h,vi+f3*h)
    vf=vi+(f1+2*f2+2*f3+f4)*(h/6)
    return vf

def Euler_SingleStep(vi,xi,h):
    xf=xi+vi*h
    return xf

def acceleration(t,v):
    # Poussée
    if 0<=t<=thrustTime[-1]:
        euler_angles=quat.as_euler_angles(q[i-1])
        theta=euler_angles[0]; phi=euler_angles[1]
        Pkz=Thrust(t)*np.cos(phi)
        Ppxy=Thrust(t)*np.sin(phi)
        Pky=Ppxy*np.cos(theta)
        Pkx=Ppxy*np.sin(theta)
    else:
        Pkx=0; Pky=0; Pkz=0
    Pk=np.array([Pkx,Pky,Pkz])
    # Résistance de l'air
    R=-1/2*rho*S*Cx*(np.square(v))
    # Poids
    P=np.array([0,0,-m*g])

    # Résultante
    forces = Pk + R + P
    accel= 1/m*forces
    return accel

def dist(a,b,ref="xyz"):
    ax=a[0]; ay=a[1]; az=a[2]
    bx=b[0]; by=b[1]; bz=b[2]
    if ref=="xy":
        return np.sqrt((bx-ax)**2+(by-ay)**2)
    return np.sqrt((bx-ax)**2+(by-ay)**2+(bz-az)**2)

def selfAcceleration(t,v):
    accel=np.array([0.0,0.0,0.0])
    return accel

def updateRotation2(eq,qi,wi,h):
    w=RK4_SingleStep(eq,wi,0,h)
    wx=w[0]; wy=w[1]; wz=w[2]
    OMEGA=np.array([[0,-wx,-wy,-wz],[wx,0,wz,-wy],[wy,-wz,0,wx],[wz,wy,-wx,0]])
    qi=quat.as_float_array(qi)
    qDot=1/2*np.matmul(OMEGA,qi)
    q=Euler_SingleStep(qDot,qi,h)
    return q,w

for i in range(1,simuNPoints):
    # LOOP principal
    speed=RK4_SingleStep(acceleration,speed,time,h)
    position=Euler_SingleStep(speed,trajecto[i-1],h)
    if position[2]<0:
        speed=np.array([0,0,0])
        trajecto[i]=trajecto[i-1]
    else:
        trajecto[i]=position
    # Calcul de rotation
    # if dist(trajecto[i-1],trajecto[i],"xy")!=0:
    #     # theta
    #     costheta=(trajecto[i][0]-trajecto[i-1][0])/dist(trajecto[i-1],trajecto[i],"xy")
    #     if trajecto[i][2]>=0:
    #         theta=np.arccos(costheta)
    #     else:
    #         theta=-np.arccos(costheta)
    #     # phi
    #     # cosphi=(dist(trajecto[i],trajecto[i-1],"xy"))/(dist(trajecto[i-1],trajecto[i]))
    #     cosphi=(trajecto[i][2]-trajecto[i-1][2])/(dist(trajecto[i-1],trajecto[i]))
    #     phi=np.arccos(cosphi)
    # q[i]=quat.from_euler_angles(np.array([theta,phi,0]))
    # Calcul de rotation 2
    q[i],wi=updateRotation2(selfAcceleration,q[i-1],wi,h)
    time+=h


# Packing en listes 1d pour tracé
x=np.array([trajecto[i,0] for i in range(simuNPoints)])
y=np.array([trajecto[i,1] for i in range(simuNPoints)])
z=np.array([trajecto[i,2] for i in range(simuNPoints)])
v=np.array([(np.linalg.norm(trajecto[i])-np.linalg.norm(trajecto[i-1]))/h for i in range(1,len(trajecto))])

# Affichage de variables intéressantes
print("Altitude max: ", max(z), " m")
print("Vitesse max: ", max(v), " m.s^-2")
# print(trajecto)

# Tracé graphique de la trajectoire
thrustEnd=ceil(thrustTime[-1]/h) # Colorer la poussée en rouge
fig = plt.figure()
fig.suptitle("Diagrammes")
simplt = fig.add_subplot(projection='3d')
simplt.plot3D(x[:thrustEnd],y[:thrustEnd],z[:thrustEnd],"r")
simplt.plot3D(x[thrustEnd:],y[thrustEnd:],z[thrustEnd:],"g")
simplt.set_xlabel('x')
simplt.set_ylabel('y')
simplt.set_zlabel('z')
simplt.set_title("Simulation")
plt.show()

# Export des données
export=""
export+=str(simulationDuration)+"\n"
q2=[quat.as_float_array(i) for i in q]
for r in range(simuNPoints):
    for i in trajecto[r]:
        export+=str(i)+";"
    for i in q2[r][:-1]:
        export+=str(i)+";"
    export+=str(q2[r][-1])+"\n"

f = open("SIM.txt", "w")
f.write(export)
f.close()
print("Saved")
