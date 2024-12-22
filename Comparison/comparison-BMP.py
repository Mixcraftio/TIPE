import numpy as np
import matplotlib.pyplot as plt

g=9.81; Cp=1006; T0=30+273.15; R=8.314; M=29e-3
altitude = lambda p: -((R*T0)/(M*g))*np.log(p/p0)
# altitude = lambda p: 3146-(p/10)

# Hellfire
# t_brut,bmp_brut=np.genfromtxt("../DATA/TENSIO.TXT", delimiter=";", usecols=[0,2], unpack=True, invalid_raise=False)
t_brut,bmp_brut=np.genfromtxt("../DATA/TENSIO-TRIM.TXT", delimiter=";", usecols=[0,2], unpack=True, invalid_raise=False)
decollageTensio=1752882
tzm=t_brut[3450:3750]-decollageTensio;bmp=bmp_brut[3450:3750] # zoom vol
p0=bmp_brut[3450]
montee=t_brut[3500:3542]-decollageTensio;calib_montee=altitude(bmp_brut[3500:3542])
descente=t_brut[3538:3720]-decollageTensio;calib_descente=altitude(bmp_brut[3538:3720])

# Karlavagnen
# t_brut,bmp_brut=np.genfromtxt("../DATA/DATA_Tensio-Alti.TXT", delimiter="\t", usecols=[0,2], unpack=True, invalid_raise=False)
# decollageTensio=1070000
# tzm=t_brut[21400:24400]-decollageTensio;bmp=bmp_brut[21400:24400] # zoom vol
# p0=bmp_brut[21400]
# montee=t_brut[21400:24400]-decollageTensio;calib_montee=altitude(bmp_brut[21400:24400])
# descente=t_brut[21400:24400]-decollageTensio;calib_descente=altitude(bmp_brut[21400:24400])


calib=altitude(bmp)

tx,ty,tz,alpha,phi,theta=np.genfromtxt("../Simulation/SIM-EULER.txt", delimiter=";", skip_header=1, unpack=True, invalid_raise=False)




fig,bmp_plot=plt.subplots()
bmp_plot.set_title("Valeurs brutes du BMP180")
bmp_plot.set_xlabel("Temps (ms)")
bmp_plot.set_ylabel("Valeur brute (Pa)")
bmp_plot.plot(tzm, bmp, ".-")
fig.savefig("./BMP.svg")
plt.show()


# t_brut,bmp_brut=np.genfromtxt("../DATA/TENSIO.TXT", delimiter=";", usecols=[0,2], unpack=True, invalid_raise=False)
# decollageTensio=1752882
# tzm1=t_brut[3450:3750]-decollageTensio;bmp1=bmp_brut[3450:3750] # zoom vol
# p0=bmp_brut[3450]
# t_brut,bmp_brut=np.genfromtxt("../DATA/DATA_Tensio-Alti.TXT", delimiter="\t", usecols=[0,2], unpack=True, invalid_raise=False)
# decollageTensio=1070000
# tzm2=t_brut[21400:24400]-decollageTensio;bmp2=bmp_brut[21400:24400] # zoom vol
# p0=bmp_brut[21400]
# fig,bmp_plot=plt.subplots()
# bmp_plot.set_title("Valeurs brutes du BMP180")
# bmp_plot.set_xlabel("Temps (ms)")
# bmp_plot.set_ylabel("Valeur brute (Pa)")
# bmp_plot.plot(tzm1, bmp1, ".-")
# bmp_plot.plot(tzm2, bmp2, ".-")


fig2,altitude_plot=plt.subplots()
altitude_plot.set_title("Valeurs du BMP180")
altitude_plot.set_xlabel("Temps (ms)")
altitude_plot.set_ylabel("Altitude (m)")
altitude_plot.plot(tzm,calib, label="Altitude mesurée")
altitude_plot.grid()


t=np.linspace(0,200000,len(tx))
altitude_plot.plot(t,tz,color="tab:red",label="Altitude z de simulation")
plt.legend()
fig2.savefig("./Altitude.svg")
plt.show()



vit=[(calib[i+1] - calib[i])/(tzm[i+1] - tzm[i])*1e3 for i in range(len(calib)-1)]

vit_montee=[(calib_montee[i+1] - calib_montee[i])/(montee[i+1] - montee[i])*1e3 for i in range(len(calib_montee)-1)]
vit_descente=[(calib_descente[i] - calib_descente[i-1])/(descente[i] - descente[i-1])*1e3 for i in range(1,len(calib_descente))]

print("\nVitesse moyenne de descente: " + str(np.mean(vit_descente))+ " m.s-1")

fig3,vites=plt.subplots()
vites.set_title("Vitesse z calculée des données du BMP180")
vites.set_xlabel("Temps (ms)")
vites.set_ylabel("Vitesse (m.s-1)")
vites.plot(tzm[:-1],vit, label="Vitesse z calculée")
vites.hlines(np.mean(vit_descente), xmin=descente[0], xmax=descente[-1], color="tab:orange", label="Moyenne de la vitesse z de descente")
vites.set_ylim((-50,200))
vites.grid()


vz=np.array([(tz[i]-tz[i-1])/0.016666666666666666 for i in range(1,len(tx))])
t=np.linspace(0,200000,len(tx)-1)
plt.plot(t,vz,color="tab:red",label="Vitesse z de simulation")
plt.legend()
fig3.savefig("./Vitesse.svg")
plt.show()