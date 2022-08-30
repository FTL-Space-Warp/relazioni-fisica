import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm, binom

line_lengths = np.array([])
letter_count = np.array([])


with open('dante.txt', 'r') as dante:
    for line in dante:
        line = line.strip('\n ').lower()
        if line:
            line_lengths = np.append(line_lengths, len(line))
            letter_count = np.append(letter_count, line.count("a"))

N = len(line_lengths)
m = line_lengths.mean()
s = line_lengths.std(ddof=1)
print(f"Numero di versi: {N}")
print(f"Media campione delle lunghezze: {m}")
print(f"Varianza campione delle lunghezze: {s}")

binning = np.arange(line_lengths.min() - 0.5, line_lengths.max() + 1.5)

plt.figure("Lunghezza dei versi")
o, _, _ = plt.hist(line_lengths, bins=binning, label="Conteggi",
                   histtype="step")
plt.xlabel("Numero di caratteri per verso")
plt.ylabel("Occorrenze")
# print(o)

k = np.arange(line_lengths.min(), line_lengths.max() + 1)
e_poisson = N * poisson.pmf(k, m)
e_gauss = N * (norm.cdf(k+0.5, m, s) - norm.cdf(k-0.5, m, s))

chi2_poisson = ((o - e_poisson)**2. / e_poisson).sum()
dof_poisson = len(k) - 1 - 1
chi2_gauss = ((o - e_gauss)**2. / e_gauss).sum()
dof_gauss = len(k) - 1 - 2

plt.bar(k, e_poisson, width=0.25, color="#ff7f0e", label="Poisson")
k = np.arange(line_lengths.min(), line_lengths.max() + 1, 0.1)
plt.plot(k, N * norm.pdf(k, m, s), color="#2ca02c", label="Gauss")

print(f"chi2 per la Poissoniana: {chi2_poisson} / {dof_poisson} dof \
      (±{(2*dof_poisson)**.5})")
print(f"chi2 per la Gaussiana: {chi2_gauss} / {dof_gauss} dof \
      (±{(2*dof_gauss)**.5})")

plt.legend()
plt.savefig("lunghezze_versi.pdf")

# Analisi delle occorrenze di una singola lettera

p = letter_count.sum() / line_lengths.sum()

binning = np.arange(letter_count.min() - 0.5, letter_count.max() + 1.5)

plt.figure("Frequenza della lettera \"a\"")
o, _, _ = plt.hist(letter_count, bins=binning, label="Conteggi",
                   histtype="step")
plt.xlabel("Occorrenze della lettera \"a\" per verso")
plt.ylabel("Occorrenze")
# print(o)

k = np.arange(letter_count.min(), letter_count.max() + 1)
e_binom = N * binom.pmf(k, m, p)
e_poisson = N * poisson.pmf(k, p * m)

chi2_binom = ((o - e_binom)**2 / e_binom).sum()
dof_binom = len(k) - 1 - 2
chi2_poisson = ((o - e_poisson)**2. / e_poisson).sum()
dof_poisson = len(k) - 1 - 1

plt.bar(k-0.125, e_poisson, width=0.25, color="#ff7f0e", label="Poisson")
plt.bar(k+0.125, e_binom, width=0.25, color="green", label="Binomiale")


print(f"chi2 per la Poissoniana: {chi2_poisson} / {dof_poisson} dof\
      (±{(2*dof_poisson)**.5})")
print(f"chi2 per la Binomiale: {chi2_binom} / {dof_binom} dof\
      (±{(2*dof_binom)**.5})")

plt.legend()
plt.savefig("frequenza_a.pdf")

plt.show()
