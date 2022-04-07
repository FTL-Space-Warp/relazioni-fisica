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

V_cilindro = np.full(diametro_cilindo / 2)**2 * np.pi * altezza_cilindro

sigma_V_cilindro = (diametro_cilindo/2 * np.pi * altezza_cilindro * sigma_d_c)**2 + ((diametro_cilindo**2/2) * np.pi * sigma_h_c)**2

#Volume parallelepipedo

lato_1_parallelepipedo = np.loadtxt("prismi.txt", usecols=0)
lato_2_parallelepipedo = np.loadtxt("prismi.txt", usecols=1)
lato_3_parallelepipedo = np.loadtxt("prismi.txt", usecols=2)

sigma_p1, sigma_p2 = np.full(lato_1_parallelepipedo.shape, 0.01)
sigma_p3 = np.array([0.01, 0.01, 0.02])

V_parallelepipedo = lato_1_parallelepipedo * lato_2_parallelepipedo * lato_3_parallelepipedo

sigma_V_parallelepipedo = np.sqrt((lato_2_parallelepipedo * lato_3_parallelepipedo * sigma_p1) + (lato_1_parallelepipedo * lato_3_parallelepipedo * sigma_p2) + (lato_1_parallelepipedo * lato_2_parallelepipedo * sigma_p3))

#Volume prismaesagonale

apotema = 9.95/2 # Media delle tre facce dell'esagono

sigma_a = 0.01/2

altezza_prismaesagonale = np.loadtxt("esagono.txt", usecols=1)[0]

sigma_al_es = 0.01

V_esagono = 2 * sqrt(3) * (apotema**2)

sigma_V_esagono = np.sprt((4*sqrt(3) * apotema * altezza_esagono * sigma_a) + (2*sqrt(3) * (apotema**2) * sigma_al_es))

#Volume sfera

raggio_sfera = np.loadtxt("sfere.txt", usecol=1)

sigma_r = (raggio_sfera.shape, 0.01)

V_sfera = 4/3 (np.pi * raggio_sfera**3)

sigma_V_sfera = np.sqrt(4 * np.pi * (raggio_sfera**2) * sigma_r)

Volume = np.concatenate(V_cilindro, V_parallelepipedo, V_esagono, V_sfera)

sigma_V= np.concatenate(sigma_V_cilindro, sigma_V_parallelepipedo, sigma_V_esagono, sigma_V_sfera)

massa = np.concatenate(sigma_m_c, ...)

def retta(x, m, q):
    return m * x + q

def legge_di_potenza(x, norm, indice):
    return norm * (x**indice)

plt.figure("Grafico massa-volume")
plt.errorbar(V_cilindro, massa_cilindro, sigma_V_cilindro, sigma_m_c)
popt, pcov = curve_fit(retta, V_cilindro, massa_cilindro)
