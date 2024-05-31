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
trajecto=[[0,0] for i in range(simuNpoints)]
time=0
# -----------------------------------

# Courbe de poussée du propulseur (en N)
POUST=[0,0.01,0.02,0.05,0.1,0.2,0.4,0.8,0.9,1,1.1,1.2,1.3,1.4,1.55,1.6,1.62,1.64,1.66,1.67,1.68,1.69,1.7]
POUSF=[0,492.25,1369.46,1236.01,1279.47,1311.39,1331.39,1304.08,1280.62,1249.86,1217.94,1199.29,1158.77,1112.56,941.81,726.07,559.17,399.95,317.66,247.28,198.05,67.3,0]
Poussee=interp1d(POUST,POUSF)

# Variables caractéristiques de la fusée
vit=[0,0] # Vitesse initiale [(en m.s-1),(en m.s-1)]
m=7.8 # Masse totale de la fusée (en kg)
Cx=0.85 # Coefficient de frottements ()
S=0.008854 # Surface projetée du dessus (en m2)
alpha=80 # Angle de lancer de la fusée (en deg)
alpha=alpha*np.pi/180

# Variables caractéristiques du milieu
g=9.81 # Intensité du champs de pesanteur ()
rho=1.293 # Masse volumique de l'air (en kg/m3)


def newVit(v,t):
    vx=v[0]; vy=v[1]
    # va=np.sqrt(vx**2+vy**2)
    # Poussée
    if 0<=t<=POUST[-1]:
        Pkx=Poussee(t)*np.cos(alpha)
        Pky=Poussee(t)*np.sin(alpha)
    else:
        Pkx=0; Pky=0
    # Résistance de l'air
    Rx=-1/2*rho*S*Cx*(vx**2)
    Ry=-1/2*rho*S*Cx*(vy**2)
    # Poids
    P=-m*g
    # Nouvelle vitesse
    vx+=1/m*(Pkx+Rx)*h
    vy+=1/m*(Pky+Ry+P)*h
    return [vx,vy]

def newPos(pos,v):
    x=pos[0]; y=pos[1]
    vx=v[0]; vy=v[1]
    x+=vx*h
    y+=vy*h
    return [x,y]

def dist(a,b):
    ax=a[0]; ay=a[1]
    bx=b[0]; by=b[1]
    return np.sqrt((bx-ax)**2+(by-ay)**2)

for i in range(1,simuNpoints):
    # Nouvelle vitesse et nouveau point
    vit=newVit(vit,time)
    newPosition=newPos(trajecto[i-1],vit)
    # if newPosition[1]<0: break
    # trajecto[i]=newPosition
    if newPosition[1]<0:
        vit=[0,0]
        trajecto[i]=trajecto[i-1]
    else:
        trajecto[i]=newPosition
    if dist(trajecto[i-1],trajecto[i])!=0:
        alpha=np.arccos((trajecto[i][0]-trajecto[i-1][0])/dist(trajecto[i-1],trajecto[i]))
        # alphax=dist([trajecto[i-1][0],0],[trajecto[i][0],0])
        # alphay=dist([0,trajecto[i-1][1]],[0,trajecto[i][1]])
    time+=h


# Packing en listes 1d pour tracé
x=[0 for i in range(simuNpoints)]
y=[0 for i in range(simuNpoints)]
for i in range(len(trajecto)):
    x[i]=trajecto[i][0]; y[i]=trajecto[i][1]

# Affichage de variables intéressantes
print("Altitude max: ", max(y), " m")
print(trajecto)
l=ceil(POUST[-1]/h) # Colorer la poussée en rouge

# Tracé graphique de la trajectoire
fig, (simplt)=plt.subplots(1)
fig.suptitle("Diagrammes")
fig.tight_layout()

simplt.plot(x[:l],y[:l],"ro")
simplt.plot(x[l:],y[l:],"go")
simplt.axline([0,0],[1,0],c="r")
simplt.set_title("Simulation")

plt.show()
