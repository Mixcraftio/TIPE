import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from math import ceil


# ----- Variables de simulation -----
# Choix temps
timeOfLaSimulation=30 # Temps de simulation (en s)
h=0.01 # Tranche de temps de la simulation (ti+1 - ti) (en s)
simuNpoints=ceil(timeOfLaSimulation/h) # Nombres de points de simulation
# Choix points
# timeOfLaSimulation=100 # Temps de simulation (en s)
# simuNpoints=1000 # Nombres de points de simulation
# h=timeOfLaSimulation/simuNpoints # Tranche de temps de la simulation (ti+1 - ti)
# Initialisation des variables python
trajecto=np.array([np.array([0.0,0.0,0.0]) for i in range(simuNpoints)])
time=0
# -----------------------------------

# Courbe de poussée du propulseur (en N)
POUST=[0,0.01,0.02,0.05,0.1,0.2,0.4,0.8,0.9,1,1.1,1.2,1.3,1.4,1.55,1.6,1.62,1.64,1.66,1.67,1.68,1.69,1.7]
POUSF=[0,492.25,1369.46,1236.01,1279.47,1311.39,1331.39,1304.08,1280.62,1249.86,1217.94,1199.29,1158.77,1112.56,941.81,726.07,559.17,399.95,317.66,247.28,198.05,67.3,0]
Poussee=interp1d(POUST,POUSF)

# Variables caractéristiques de la fusée
vit=np.array([0,0,0]) # Vitesse initiale [(en m.s-1),(en m.s-1)]
m=7.8 # Masse totale de la fusée (en kg)
Cx=0.85 # Coefficient de frottements ()
S=0.008854 # Surface projetée du dessus (en m2)
alpha=80 # Angle de lancer de la fusée (en deg)
beta=45
alpha=alpha*np.pi/180
beta=beta*np.pi/180

# Variables caractéristiques du milieu
g=9.81 # Intensité du champs de pesanteur ()
rho=1.293 # Masse volumique de l'air (en kg/m3)


def RK4_SingleStep(eq,h,ti,vi):
    f1=eq(ti,vi)
    f2=eq(ti+h/2,vi+f1*h/2)
    f3=eq(ti+h/2,vi+f2*h/2)
    f4=eq(ti+h,vi+f3*h)
    vf=vi+(f1+2*f2+2*f3+f4)*(h/6)
    return vf

def Euler_SingleStep(vi,h,xi):
    xf=xi+vi*h
    return xf

def equa(t,v):
    # Poussée
    if 0<=t<=POUST[-1]:
        Pkz=Poussee(t)*np.sin(alpha)
        Ppxy=Poussee(t)*np.cos(alpha) # Projeté de la poussée sur xy
        Pky=Ppxy*np.cos(beta)
        Pkx=Ppxy*np.sin(beta)
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

for i in range(1,simuNpoints):
    # Nouvelle vitesse et nouveau point
    vit=RK4_SingleStep(equa,h,time,vit)
    newPosition=Euler_SingleStep(vit,h,trajecto[i-1])
    if newPosition[2]<0:
        vit=[0,0,0]
        trajecto[i]=trajecto[i-1]
    else:
        trajecto[i]=newPosition
    
    # Calcul de rotation
    if dist(trajecto[i-1],trajecto[i],"xy")!=0:
        # alpha
        cosa=(dist(trajecto[i],trajecto[i-1],"xy"))/(dist(trajecto[i-1],trajecto[i]))
        alpha=np.arccos(cosa)
        # beta
        cosb=(trajecto[i][0]-trajecto[i-1][0])/dist(trajecto[i-1],trajecto[i],"xy")
        if trajecto[i][2]>=0:
            beta=np.arccos(cosb)
        else:
            beta=-np.arccos(cosb)
    
    time+=h


# Packing en listes 1d pour tracé
x=[trajecto[i,0] for i in range(simuNpoints)]
y=[trajecto[i,1] for i in range(simuNpoints)]
z=[trajecto[i,2] for i in range(simuNpoints)]

# Affichage de variables intéressantes
print("Altitude max: ", max(z), " m")
print(trajecto)
l=ceil(POUST[-1]/h) # Colorer la poussée en rouge

# Tracé graphique de la trajectoire
fig, (simplt)=plt.subplots(1)
fig.suptitle("Diagrammes")
fig.tight_layout()

simplt=plt.axes(projection='3d')
# simplt.view_init(30, 45)
simplt.plot3D(x[:l],y[:l],z[:l],"r")
simplt.plot3D(x[l:],y[l:],z[l:],"g")
simplt.set_xlabel('x')
simplt.set_ylabel('y')
simplt.set_zlabel('z')
simplt.set_title("Simulation")

plt.show()

t2=""
for r in trajecto:
    for i in r[:-1]:
        t2+=str(i)+";"
    t2+=str(i)+"\n"
# print(t2)

f = open("SIM.txt", "a")
f.write(t2)
f.close()
