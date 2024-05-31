import matplotlib.pyplot as plt
import numpy as np


LeIteratOflaSimu=120
LeTrajectoir=[[0,0] for i in range(LeIteratOflaSimu)]
LeVitesse=(5,50)
LeTrancheOfTime=0.1

def LeNouvelAccelerat():
    m=7
    g=9.81
    POIDS=[0,-m*g]
    return [POIDS[0]/m,POIDS[1]/m]

def LeNouvelVitesse(v,F):
    vx=v[0]; vy=v[1]
    Fx=F[0]; Fy=F[1]
    return [vx+(Fx*LeTrancheOfTime),vy+(Fy*LeTrancheOfTime)]

def LeNouvelPosition(pos,v):
    x=pos[0]; y=pos[1]
    vx=v[0]; vy=v[1]
    return [x+(vx*LeTrancheOfTime),y+(vy*LeTrancheOfTime)]

for i in range(1,LeIteratOflaSimu):
    # print(LeResultanteDesForces)
    LeAccelerat=LeNouvelAccelerat()
    # print(LeVitesse)
    LeVitesse=LeNouvelVitesse(LeVitesse,LeAccelerat)
    # print(LeTrajectoir[i])
    NouvelPoint=LeNouvelPosition(LeTrajectoir[i-1],LeVitesse)
    LeTrajectoir[i]=NouvelPoint

x=[0 for i in range(LeIteratOflaSimu)]
y=[0 for i in range(LeIteratOflaSimu)]
for i in range(len(LeTrajectoir)):
    x[i]=LeTrajectoir[i][0]; y[i]=LeTrajectoir[i][1]

print(max(y))
plt.plot(x,y,"bo")
plt.axline([0,0],[1,0],c="r")
plt.show()
