from cmath import sqrt
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

#Volume cilindro

diametro_cilindo = np.loadtxt("cilindri.txt", usecols=0)
altezza_cilindro = np.loadtxt("cilindri.txt", usecols=1)
massa_cilindro = np.loadtxt("cilindri.txt", usecols=2)

sigma_d_c = np.full(diametro_cilindo.shape, 0.01)
sigma_h_c = np.array([0.01, 0.01, 0.01, 0.02])
sigma_m_c = np.full(massa_cilindro.shape, 0.001)

V_cilindro = (diametro_cilindo / 2)**2 * np.pi * altezza_cilindro

sigma_V_cilindro = np.sqrt((diametro_cilindo/2 * np.pi * altezza_cilindro * sigma_d_c)**2 + ((diametro_cilindo**2/2) * np.pi * sigma_h_c)**2)

#Volume parallelepipedo

lato_1_parallelepipedo = np.loadtxt("parallelepipedo.txt", usecols=0)
lato_2_parallelepipedo = np.loadtxt("parallelepipedo.txt", usecols=1)
lato_3_parallelepipedo = np.loadtxt("parallelepipedo.txt", usecols=2)
massa_parallelepipedo = np.loadtxt("parallelepipedo.txt", usecols=3)

sigma_p1, sigma_p2 = np.full(lato_1_parallelepipedo.shape, 0.01)
sigma_p3 = np.array([0.01, 0.01, 0.02])
sigma_m_pa = np.full(massa_cilindro.shape, ...)

V_parallelepipedo = lato_1_parallelepipedo * lato_2_parallelepipedo * lato_3_parallelepipedo

sigma_V_parallelepipedo = np.sqrt((lato_2_parallelepipedo * lato_3_parallelepipedo * sigma_p1) + (lato_1_parallelepipedo * lato_3_parallelepipedo * sigma_p2) + (lato_1_parallelepipedo * lato_2_parallelepipedo * sigma_p3))

#Volume prisma

apotema = 9.95/2 # Media delle tre facce dell'esagono
altezza_prisma = np.loadtxt("prisma.txt", usecols=0)[0] 
massa_prisma = np.loadtxt("prisma.txt", usecol=1)

sigma_a = 0.01/2
sigma_al_p = 0.01
sigma_m_pr = ...

V_esagono = 2 * sqrt(3) * (apotema**2)

sigma_V_esagono = np.sprt((4*sqrt(3) * apotema * altezza_prisma * sigma_a) + (2*sqrt(3) * (apotema**2) * sigma_al_p))

#Volume sfera

raggio_sfera = np.loadtxt("sfere.txt", usecol=0)
massa_sfera = np.loadtxt("sfere.txt", usecol=1)

sigma_r = (raggio_sfera.shape, 0.01)
sigma_m_s = ...

V_sfera = 4/3 (np.pi * raggio_sfera**3)

sigma_V_sfera = np.sqrt(4 * np.pi * (raggio_sfera**2) * sigma_r)

# Volumi e masse

Volume = np.concatenate(V_cilindro, V_parallelepipedo, V_esagono, V_sfera)

sigma_V = np.concatenate(sigma_V_cilindro, sigma_V_parallelepipedo, sigma_V_esagono, sigma_V_sfera)

massa = np.concatenate(massa_cilindro, massa_parallelepipedo, massa_prisma, massa_sfera)

sigma_massa = np.concatenate(sigma_m_c, sigma_m_pa, sigma_m_pr, sigma_m_s)

# Funzione e legge di potenza

def retta(x, m, q):
    return m * x + q

def legge_di_potenza(x, norm, indice):
    return norm * (x**indice)

# Plot

plt.figure("Grafico massa-volume")
plt.errorbar(Volume, massa, sigma_V, sigma_massa)
popt, pcov = curve_fit(retta, V_cilindro, massa_cilindro)
m_hat, q_hat = popt
sigma_m, sigma_q = np.sqrt(pcov.diagonal())
print(m_hat, sigma_m, q_hat, sigma_q)

# Grafico modello di best fit

x = np.linspace(0, 4000, 100)
plt.plot(x, retta(x, m_hat, q_hat))
plt.xlabel("Volume [mm$^3$")
plt.ylabel("Massa [g]")
plt.grid(which="both", ls="dashes", color="gray")

plt.figure("Grafico massa-raggio")
plt.errorbar(raggio_sfera, massa_sfera, sigma_m_s, sigma_r, fmt="o")
popt, pcov = curve_fit(legge_di_potenza, raggio_sfera, massa_sfera)
norm_hat, indice_hat = popt
sigma_norm, sigma_indice = np.sqrt(pcov.diagonal())
print(norm_hat, sigma_norm, indice_hat, sigma_indice)
x = np.linspace(4, 10, 100)
plt.plot(x, legge_di_potenza(x, norm_hat, indice_hat))
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Raggio [mm]")
plt.ylabel("Massa [g]")
plt.grid(which="both", ls="dashed", color="gray")
plt.savefig("massa_raggio.pdf")