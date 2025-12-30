import numpy as np
import librosa


def magnitude_spectrum(x, fs):
    """
    x : 1-D signal
    fs: sampling rate (Hz)
    returns freqs, |X(f)|
    """
    X = np.fft.rfft(x)
    mag = np.abs(X)
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    return freqs, mag


def spectrum_descriptor(x, fs):
    freqs, mag = magnitude_spectrum(x, fs)
    # normalize to unit energy (optional but common)
    mag = mag / (np.linalg.norm(mag) + 1e-12)
    return freqs, mag


def spectral_centroid(x, fs):
    freqs, mag = magnitude_spectrum(x, fs)
    num = np.sum(freqs * mag)
    den = np.sum(mag) + 1e-12
    return num / den


def band_energy_ratios(x, fs, bands):
    """
    bands: list of (f_low, f_high)
    returns list of energy ratios per band relative to total
    """
    freqs, mag = magnitude_spectrum(x, fs)
    power = mag**2
    total = np.sum(power) + 1e-12
    ratios = []

    for f0, f1 in bands:
        idx = np.logical_and(freqs >= f0, freqs < f1)
        ratios.append(np.sum(power[idx]) / total)

    return ratios


def mfcc_features(x, fs, n_mfcc=13):
    """
    returns an array shape (n_mfcc, n_frames)
    often averaged over time for a fixed-length vector
    """
    mfcc = librosa.feature.mfcc(y=x, sr=fs, n_mfcc=n_mfcc, center=True)
    # common: average across frames
    return mfcc.mean(axis=1)
