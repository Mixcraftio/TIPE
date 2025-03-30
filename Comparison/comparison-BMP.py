import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

g=9.81; Cp=1006; T0=25+273.15; R=8.314; M=29e-3
altitude = lambda p: -((R*T0)/(M*g))*np.log(p/p0)
# altitude = lambda p: 3146-(p/10)


#------------------- Data Import -------------------
# Hellfire
# df = pd.read_csv("../DATA/TENSIO.TXT", sep=";", usecols=[0, 2], engine="python")
# dataFile = pd.read_csv("../DATA/TENSIO-TRIM.TXT", sep=";", usecols=[0, 2], engine="python")
dataFile = pd.read_csv("../DATA/bmp-compensated.csv", sep=",", usecols=[0, 2], engine="python")
dataFile = dataFile.dropna()
t_brut, bmp_brut = dataFile.iloc[:, 0].values, dataFile.iloc[:, 1].values
decollageTensio=1752882
tzm=t_brut[3450:3750]-decollageTensio;bmp=bmp_brut[3450:3750] # zoom vol
p0=bmp_brut[3450]
montee=t_brut[3500:3542]-decollageTensio;calib_montee=altitude(bmp_brut[3500:3542])
descente=t_brut[3538:3720]-decollageTensio;calib_descente=altitude(bmp_brut[3538:3720])

# Karlavagnen
# dataFile = pd.read_csv("../DATA/DATA_Tensio-Alti.TXT", sep="\t", usecols=[0, 2], engine="python")
# dataFile = dataFile.dropna()
# t_brut, bmp_brut = dataFile.iloc[:, 0].values, dataFile.iloc[:, 1].values
# decollageTensio=1090500
# tzm=t_brut[21400:24400]-decollageTensio;bmp=bmp_brut[21400:24400] # zoom vol
# p0=bmp_brut[21400]
# montee=t_brut[21400:24400]-decollageTensio;calib_montee=altitude(bmp_brut[21400:24400])
# descente=t_brut[21400:24400]-decollageTensio;calib_descente=altitude(bmp_brut[21400:24400])


calib=altitude(bmp)
print("Altitude maximale (apogee): " + str(max(calib)) + " m")

#------------- Simulation Data Import --------------
#Perso
dataFilePerso = pd.read_csv("../Simulation/OUT/SIM-EULER.csv", comment="#", sep=",", engine="python")
dataFilePerso = dataFilePerso.dropna()
t_sim_Perso, alti_sim_Perso = dataFilePerso.iloc[:, 0].values, dataFilePerso.iloc[:, 3].values
vites_sim_Perso = np.array([(alti_sim_Perso[i]-alti_sim_Perso[i-1])/(t_sim_Perso[i]-t_sim_Perso[i-1]) for i in range(1,len(t_sim_Perso))] + [(alti_sim_Perso[len(t_sim_Perso)-1]-alti_sim_Perso[len(t_sim_Perso)-2])/(t_sim_Perso[len(t_sim_Perso)-1]-t_sim_Perso[len(t_sim_Perso)-2])])
t_sim_Perso*=1000 # s -> ms
#StabTraj
dataFileStabTraj = pd.read_csv("../DATA/StabTraj-Simulation-SansVent.csv", comment="#", sep=",", engine="python")
dataFileStabTraj = dataFileStabTraj.dropna()
t_sim_StabTraj, alti_sim_StabTraj = dataFileStabTraj.iloc[:, 0].values, dataFileStabTraj.iloc[:, 1].values
vites_sim_StabTraj = np.array([(alti_sim_StabTraj[i]-alti_sim_StabTraj[i-1])/(t_sim_StabTraj[i]-t_sim_StabTraj[i-1]) for i in range(1,len(t_sim_StabTraj))] + [(alti_sim_StabTraj[len(t_sim_StabTraj)-1]-alti_sim_StabTraj[len(t_sim_StabTraj)-2])/(t_sim_StabTraj[len(t_sim_StabTraj)-1]-t_sim_StabTraj[len(t_sim_StabTraj)-2])])
t_sim_StabTraj*=1000 # s -> ms
#OpenRocket
dataFileOpenRocket = pd.read_csv("../DATA/OpenRocket-Simulation-SansVent.csv", comment="#", sep=",", engine="python")
dataFileOpenRocket = dataFileOpenRocket.dropna()
t_sim_OpenRocket, alti_sim_OpenRocket, vites_sim_OpenRocket = dataFileOpenRocket.iloc[:, 0].values, dataFileOpenRocket.iloc[:, 1].values, dataFileOpenRocket.iloc[:, 2].values
t_sim_OpenRocket*=1000 # s -> ms
#RocketPy
dataFileRocketPy = pd.read_csv("../DATA/RocketPy-Simulation-SansVent.csv", comment="#", sep=",", engine="python")
dataFileRocketPy = dataFileRocketPy.dropna()
t_sim_RocketPy, alti_sim_RocketPy = dataFileRocketPy.iloc[:, 0].values, dataFileRocketPy.iloc[:, 3].values
vites_sim_RocketPy = np.array([(alti_sim_RocketPy[i]-alti_sim_RocketPy[i-1])/(t_sim_RocketPy[i]-t_sim_RocketPy[i-1]) for i in range(1,len(t_sim_RocketPy))] + [(alti_sim_RocketPy[len(t_sim_RocketPy)-1]-alti_sim_RocketPy[len(t_sim_RocketPy)-2])/(t_sim_RocketPy[len(t_sim_RocketPy)-1]-t_sim_RocketPy[len(t_sim_RocketPy)-2])])
t_sim_RocketPy*=1000 # s -> ms
#---------------------------------------------------


#------------------- Brut data plot -------------------
fig,bmp_plot=plt.subplots()
bmp_plot.set_title("Valeurs brutes du BMP180")
bmp_plot.set_xlabel("Temps (ms)")
bmp_plot.set_ylabel("Valeur brute (Pa)")
bmp_plot.plot(tzm, bmp, ".")
fig.savefig("./OUT/BMP.svg")
plt.show()


#------------------- Altitude plot -------------------
fig2,altitude_plot=plt.subplots()
altitude_plot.set_title("Valeurs du BMP180")
altitude_plot.set_xlabel("Temps (ms)")
altitude_plot.set_ylabel("Altitude (m)")

altitude_plot.plot(tzm,calib, label="Altitude mesurée")

altitude_plot.plot(t_sim_Perso,alti_sim_Perso,color="tab:orange",label="Altitude z de simulation Perso")
altitude_plot.plot(t_sim_StabTraj,alti_sim_StabTraj,color="tab:green",label="Altitude z de simulation StabTraj")
altitude_plot.plot(t_sim_OpenRocket,alti_sim_OpenRocket,color="tab:red",label="Altitude z de simulation OpenRocket")
altitude_plot.plot(t_sim_RocketPy,alti_sim_RocketPy,color="tab:grey",label="Altitude z de simulation RocketPy")

altitude_plot.grid()
altitude_plot.legend()
fig2.savefig("./OUT/Altitude.svg")
plt.show()



#------------------- Vitesse plot -------------------
vit=[(calib[i+1] - calib[i])/(tzm[i+1] - tzm[i])*1e3 for i in range(len(calib)-1)]
vit_montee=[(calib_montee[i+1] - calib_montee[i])/(montee[i+1] - montee[i])*1e3 for i in range(len(calib_montee)-1)]
vit_descente=[(calib_descente[i] - calib_descente[i-1])/(descente[i] - descente[i-1])*1e3 for i in range(1,len(calib_descente))]
print("Vitesse moyenne de descente: " + str(np.mean(vit_descente))+ " m.s-1")

fig3,vites=plt.subplots()
vites.set_title("Vitesse z calculée des données du BMP180")
vites.set_xlabel("Temps (ms)")
vites.set_ylabel("Vitesse (m.s-1)")

vites.plot(tzm[:-1],vit, label="Vitesse z calculée")
vites.hlines(np.mean(vit_descente), xmin=descente[0], xmax=descente[-1], color="tab:purple", label="Moyenne de la vitesse z de descente")
vites.set_ylim((-50,205))

vites.plot(t_sim_Perso,vites_sim_Perso,color="tab:orange",label="Vitesse z de simulation Perso")
vites.plot(t_sim_StabTraj,vites_sim_StabTraj,color="tab:green",label="Vitesse z de simulation StabTraj")
vites.plot(t_sim_OpenRocket,vites_sim_OpenRocket,color="tab:red",label="Vitesse z de simulation OpenRocket")
vites.plot(t_sim_RocketPy,vites_sim_RocketPy,color="tab:grey",label="Vitesse z de simulation RocketPy")

vites.grid()
vites.legend()
fig3.savefig("./OUT/Vitesse.svg")
plt.show()