import string
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm  # , binom

line_lengths = np.array([])
letter_counts = {}

for i in string.ascii_lowercase:
    letter_counts[i] = []

with open('dante.txt', 'r') as dante:
    for line in dante:
        line = line.replace('\n', '').lower()
        if line:
            line_lengths = np.append(line_lengths, len(line))
            for i in string.ascii_lowercase:
                letter_counts[i].append(line.count(i))

N = len(line_lengths)
m = line_lengths.mean()
s = line_lengths.std(ddof=1)
print(N)
print(m)
print(s)

binning = np.arange(line_lengths.min() - 0.5, line_lengths.max() + 1.5)

plt.figure("Lunghezza dei versi")
o, _, _ = plt.hist(line_lengths, bins=binning, rwidth=0.25, label="Conteggi")
plt.xlabel("Numero di caratteri per verso")
plt.ylabel("Occorrenze")

k = np.arange(line_lengths.min(), line_lengths.max() + 1)
e_poisson = N * poisson.pmf(k, m)
e_gauss = N * norm.pdf(k, m, s)

plt.bar(k - 0.3, e_poisson, width=0.25, color="#ff7f0e", label="Poisson")
plt.plot(k, e_gauss, color="#2ca02c", label="Gauss")

chi2_poisson = ((o - e_poisson)**2. / e_poisson).sum()
chi2_gauss = ((o - e_gauss)**2. / e_gauss).sum()
dof_poisson = len(k) - 1 - 1
dof_gauss = len(k) - 1 - 2
print(f"chi2 per la Poissoniana: {chi2_poisson} / {dof_poisson} dof")
print(f"chi2 per la Gaussiana: {chi2_gauss} / {dof_gauss} dof")

plt.legend()
plt.show()
