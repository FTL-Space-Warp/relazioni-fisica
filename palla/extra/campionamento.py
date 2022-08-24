import wave
import sys
import numpy as np
from matplotlib import pyplot as plt

file_path = sys.argv[1]

# Extract data from wave file
stream = wave.open(file_path)
signal = np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16)
framerate = stream.getframerate()


def const(t):
    return 15000


def sample_file(th, sep):
    """Finds all bounces in an audio signal

    Keyword arguments:
    th -- threshold to identify a datapoint
    sep -- minimun separation bewtween datapoints
    """

    start_point = 0
    end_point = 0
    peak = 0

    sampling = 0

    samples = []
    peaks = []
    sigmas = []

    for i in range(len(signal)):
        # Start sampling
        if abs(signal[i]) >= th and not sampling:
            start_point = end_point = i
            sampling = sep
            peak = i

        elif abs(signal[i]) >= th and sampling:
            sampling = sep
            # Find new peak
            if abs(signal[i]) > abs(signal[peak]):
                peak = i

        elif abs(signal[i]) < th and sampling > 1:
            sampling -= 1
            end_point = i - sep

        elif abs(signal[i]) < th and sampling == 1:
            sigmas.append((end_point-start_point)/(framerate*2))
            samples.append((start_point+end_point)/(framerate*2))
            start_point = end_point = 0
            peaks.append(peak/framerate)
            peak = 0

            sampling = False

    return samples, peaks, sigmas


# Get parameters from cli
threshold = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
separation = int(sys.argv[3]) if len(sys.argv) > 3 else 200
num = int(sys.argv[4]) if len(sys.argv) > 4 else 50


samples, peaks, sigmas = sample_file(threshold, separation)

if num:
    samples = samples[:(num-len(samples))]
    peaks = peaks[:(num-len(peaks))]
    sigmas = sigmas[:(num-len(sigmas))]

sigma = max(sigmas)
print("framerate: %d" % framerate)
print("sigma = %f" % sigma)
print("%d bounces found" % len(samples))

# Plot Result
t = np.arange(len(signal)) / framerate

plt.figure('Rimbalzi pallina')
plt.axhline(y=threshold, color='red', linestyle='--')
plt.axhline(y=-1*threshold, color='red', linestyle='--')
plt.scatter(samples,
            np.zeros(len(samples)),
            c='red',
            zorder=2)
plt.plot(t, signal, zorder=1)
plt.xlabel('Tempo [s]')

# Save as csv
np.savetxt(file_path.rstrip('wav')+'csv', samples, delimiter=',')

plt.show()
