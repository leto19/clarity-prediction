from scipy.signal import resample_poly
import numpy as np
def resample_oct(x, p, q):
    """Resampler that is compatible with Octave"""
    h = _resample_window_oct(p, q)
    window = h / np.sum(h)
    return resample_poly(x, p, q, window=window)

def _resample_window_oct(p, q):
    """Port of Octave code to Python"""

    gcd = np.gcd(p, q)
    if gcd > 1:
        p /= gcd
        q /= gcd

    # Properties of the antialiasing filter
    log10_rejection = -3.0
    stopband_cutoff_f = 1. / (2 * max(p, q))
    roll_off_width = stopband_cutoff_f / 10

    # Determine filter length
    rejection_dB = -20 * log10_rejection
    L = np.ceil((rejection_dB - 8) / (28.714 * roll_off_width))

    # Ideal sinc filter
    t = np.arange(-L, L + 1)
    ideal_filter = 2 * p * stopband_cutoff_f \
        * np.sinc(2 * stopband_cutoff_f * t)

    # Determine parameter of Kaiser window
    if (rejection_dB >= 21) and (rejection_dB <= 50):
        beta = 0.5842 * (rejection_dB - 21)**0.4 \
            + 0.07886 * (rejection_dB - 21)
    elif rejection_dB > 50:
        beta = 0.1102 * (rejection_dB - 8.7)
    else:
        beta = 0.0

    # Apodize ideal filter response
    h = np.kaiser(2 * L + 1, beta) * ideal_filter

    return h

def eb_Resample24kHz(x,fs_x):
    fs_new = 24000
    if fs_x == fs_new:
        return x
    else:
       x = resample_oct(x, fs_new, fs_x)
    return x,fs_new