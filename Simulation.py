import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from math import ceil


LeNumberOfPoints=1000 #int
LeTimeOfLaSimulation=100 #s
LeTrancheOfTime=LeTimeOfLaSimulation/LeNumberOfPoints
LeTrajectoir=[[0,0] for i in range(LeNumberOfPoints)]
LeVitesse=[0,0]
LeTime=0

POUST=[0,0.01,0.02,0.05,0.1,0.2,0.4,0.8,0.9,1,1.1,1.2,1.3,1.4,1.55,1.6,1.62,1.64,1.66,1.67,1.68,1.69,1.7]
POUSF=[0,492.25,1369.46,1236.01,1279.47,1311.39,1331.39,1304.08,1280.62,1249.86,1217.94,1199.29,1158.77,1112.56,941.81,726.07,559.17,399.95,317.66,247.28,198.05,67.3,0]
CurveOfLaPoussee=interp1d(POUST,POUSF)

def LeNouvelAccelerat(v,t):
    vx=v[0]; vy=v[1]
    m=7.8
    g=9.81
    POIDS=[0,-m*g]
    Cx=0.85
    rho=1.293
    S=0.008854
    RESIST=[-(1/2)*rho*Cx*(vx**2)*S,-(1/2)*rho*Cx*(vy**2)*S]
    if 0<=t<=POUST[-1]:
        POUSSEE=[1,CurveOfLaPoussee(t)]
    else:
        POUSSEE=[0,0]
    return ((POIDS[0]+RESIST[0]+POUSSEE[0])/m,(POIDS[1]+RESIST[1]+POUSSEE[1])/m)

def LeNouvelVitesse(v,F):
    vx=v[0]; vy=v[1]
    Fx=F[0]; Fy=F[1]
    return [vx+(Fx*LeTrancheOfTime),vy+(Fy*LeTrancheOfTime)]

def LeNouvelPosition(pos,v):
    x=pos[0]; y=pos[1]
    vx=v[0]; vy=v[1]
    return [x+(vx*LeTrancheOfTime),y+(vy*LeTrancheOfTime)]

for i in range(1,LeNumberOfPoints):
    # print(LeResultanteDesForces)
    LeAccelerat=LeNouvelAccelerat(LeVitesse,LeTime)
    # print(LeVitesse)
    LeVitesse=LeNouvelVitesse(LeVitesse,LeAccelerat)
    # print(LeTrajectoir[i])
    NouvelPoint=LeNouvelPosition(LeTrajectoir[i-1],LeVitesse)
    if NouvelPoint[1]<=0:
        LeVitesse=[0,0]
        LeAccelerat=[0,0]
        LeTrajectoir[i]=LeTrajectoir[i-1]
    else:
        LeTrajectoir[i]=NouvelPoint
    LeTime+=LeTrancheOfTime

x=[0 for i in range(LeNumberOfPoints)]
y=[0 for i in range(LeNumberOfPoints)]
for i in range(len(LeTrajectoir)):
    x[i]=LeTrajectoir[i][0]; y[i]=LeTrajectoir[i][1]


print("Altitude max: ", max(y), " m")
print(LeTrajectoir)
l=ceil(POUST[-1]/LeTrancheOfTime)

fig, (simplt,thrstplt)=plt.subplots(2)
fig.suptitle("Diagrammes")
fig.tight_layout()

simplt.plot(x[:l],y[:l],"ro")
simplt.plot(x[l:],y[l:],"go")
simplt.axline([0,0],[1,0],c="r")
simplt.set_title("Simulation")

thrstplt.plot(POUST,CurveOfLaPoussee(POUST))
thrstplt.set_title("Courbe de la poussée")

plt.show()
