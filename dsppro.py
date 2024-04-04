import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import find_peaks
import pywt

# Inputs
filename = input('Enter the name of the ECG signal file (including .mat extension): ')
Fs = float(input('Enter sampling rate: '))

# Load ECG data
ecg = loadmat(filename)
ecgsig = ecg['val'][0] / 200
t = np.arange(len(ecgsig))
tx = t / Fs

# Wavelet transformation
coeffs = pywt.dwt(ecgsig, 'sym4')
cA, cD = coeffs

# Reconstructing signal without high-frequency components
reconstructed_signal = pywt.idwt(cA, None, 'sym4')

# Square of reconstructed signal
y = np.abs(reconstructed_signal) ** 2
avg = np.mean(y)

# Finding peaks
peaks, _ = find_peaks(y, height=8 * avg, distance=50)
nohb = len(peaks)
timelimit = len(ecgsig) / Fs
hppermin = (nohb * 60) / timelimit
print('Heart Rate =', hppermin)

# Displaying ECG signal and detected R-peaks
plt.subplot(211)
plt.plot(tx, ecgsig)
plt.title('Original ECG Signal')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')

plt.subplot(212)
plt.plot(tx, y)
plt.plot(tx[peaks], y[peaks], 'x')
plt.title('Detected R-Peaks')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()