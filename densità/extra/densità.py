import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

# Volume cilindro

diametro_cilindo = np.loadtxt("cilindri.txt", usecols=0)
altezza_cilindro = np.loadtxt("cilindri.txt", usecols=1)
massa_cilindro = np.loadtxt("cilindri.txt", usecols=2)

sigma_d_c = np.full(diametro_cilindo.shape, 0.01)
sigma_h_c = np.array([0.01, 0.01, 0.01, 0.02])
sigma_m_c = np.full(massa_cilindro.shape, 0.001)

V_cilindro = (diametro_cilindo / 2)**2 * np.pi * altezza_cilindro

sigma_V_cilindro = np.sqrt(
    (diametro_cilindo/2 * np.pi * altezza_cilindro * sigma_d_c)**2 +
    ((diametro_cilindo**2/2) * np.pi * sigma_h_c)**2)

# Volume parallelepipedo

lato_1_parallelepipedo = np.loadtxt("parallelepipedi.txt", usecols=0)
lato_2_parallelepipedo = np.loadtxt("parallelepipedi.txt", usecols=1)
lato_3_parallelepipedo = np.loadtxt("parallelepipedi.txt", usecols=2)
massa_parallelepipedo = np.loadtxt("parallelepipedi.txt", usecols=3)

sigma_p1 = np.full(lato_1_parallelepipedo.shape, 0.01)
sigma_p2 = sigma_p1
sigma_p3 = np.array([0.01, 0.01, 0.02])
sigma_m_pa = np.full(massa_parallelepipedo.shape, 0.001)

V_parallelepipedo = (lato_1_parallelepipedo * lato_2_parallelepipedo *
                     lato_3_parallelepipedo)

sigma_V_parallelepipedo = np.sqrt(
    (lato_2_parallelepipedo * lato_3_parallelepipedo * sigma_p1) +
    (lato_1_parallelepipedo * lato_3_parallelepipedo * sigma_p2) +
    (lato_1_parallelepipedo * lato_2_parallelepipedo * sigma_p3))

# Volume prisma

apotema = 9.95/2  # Media delle tre facce dell'esagono
altezza_prisma = float(np.loadtxt("prisma.txt", usecols=0))
massa_prisma = np.loadtxt("prisma.txt", usecols=1, ndmin=1)

sigma_a = 0.01/2
sigma_al_p = 0.01
sigma_m_pr = np.array([0.001], ndmin=1)

V_prisma = np.array([2 * np.sqrt(3) * (apotema**2)], ndmin=1)

sigma_V_prisma = np.array(
    [np.sqrt((4*np.sqrt(3) * apotema * altezza_prisma * sigma_a) +
             (2*np.sqrt(3) * (apotema**2) * sigma_al_p))],
    ndmin=1)

# Volume sfera

raggio_sfera = np.loadtxt("sfere.txt", usecols=0)
massa_sfera = np.loadtxt("sfere.txt", usecols=1)

sigma_r = np.full(raggio_sfera.shape, 0.01)
sigma_m_s = np.full(massa_sfera.shape, 0.001)

V_sfera = 4/3 * (np.pi * raggio_sfera**3)

sigma_V_sfera = np.sqrt(4 * np.pi * (raggio_sfera**2) * sigma_r)

# Volumi e masse

Volume = np.concatenate((V_cilindro, V_parallelepipedo, V_prisma, V_sfera))

sigma_V = np.concatenate((sigma_V_cilindro, sigma_V_parallelepipedo,
                         sigma_V_prisma, sigma_V_sfera))

massa = np.concatenate((massa_cilindro, massa_parallelepipedo, massa_prisma,
                       massa_sfera))

sigma_massa = np.concatenate((sigma_m_c, sigma_m_pa, sigma_m_pr, sigma_m_s))


# Funzione e legge di potenza


def retta(x, m, q):
    return m * x + q


def legge_di_potenza(x, norm, indice):
    return norm * (x**indice)


# Plot

plt.figure("Grafico massa-volume")

colori = ['blue', 'red', 'green']
label = ['materiale 1', 'materiale 2', 'materiale 3']

# bitmask per i materiali
bits = np.array([2**x for x in range(len(Volume)-1, -1, -1)])
materiali = np.array([(bits & 0b1111000100000) != 0,
                      (bits & 0b0000111000000) != 0,
                      (bits & 0b0000000011111) != 0])

for i in range(len(materiali)):
    plt.errorbar(Volume[materiali[i]], massa[materiali[i]],
                 sigma_V[materiali[i]], sigma_massa[materiali[i]],
                 color=colori[i], label=label[i], fmt='o')

    # Fit dei dati

    popt, pcov = curve_fit(retta, Volume[materiali[i]], massa[materiali[i]])
    m_hat, q_hat = popt
    sigma_m, sigma_q = np.sqrt(pcov.diagonal())
    print(m_hat, sigma_m, q_hat, sigma_q)

    # Grafico modello di best fit

    x = np.linspace(Volume[materiali[i]].min(), Volume[materiali[i]].max(), 2)
    plt.plot(x, retta(x, m_hat, q_hat), ls="--", color=colori[i])

plt.xlabel("Volume [cm$^3$]")
plt.ylabel("Massa [g]")
plt.grid(which="both", ls="-", color='gray')

plt.legend()

plt.figure("Grafico massa-raggio")
plt.errorbar(raggio_sfera, massa_sfera, sigma_m_s, sigma_r, fmt="o")
popt, pcov = curve_fit(legge_di_potenza, raggio_sfera, massa_sfera)
norm_hat, indice_hat = popt
sigma_norm, sigma_indice = np.sqrt(pcov.diagonal())
print(norm_hat, sigma_norm, indice_hat, sigma_indice)
x = np.linspace(10, 30)
plt.plot(x, legge_di_potenza(x, norm_hat, indice_hat))
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Raggio [mm]")
plt.ylabel("Massa [g]")
plt.grid(which="both", ls="dashed", color="gray")
plt.savefig("massa_raggio.pdf")

plt.show()
