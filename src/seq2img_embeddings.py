from pyts.image import MarkovTransitionField, RecurrencePlot, GramianAngularField
import cv2
from scipy.signal import stft
import numpy as np


def phase_space_image(ts, tau=64, m=4, bins=64):
    n = len(ts)
    valid = n - (m - 1) * tau
    X = np.column_stack([ts[i : i + valid] for i in range(0, m * tau, tau)])

    H, _, _ = np.histogram2d(X[:, 0], X[:, 1], bins=bins)
    H = H.astype(float)

    return H / (H.max() + 1e-12)  # avoid divide-by-zero


import numpy as np


def s2i_vector(
    a: np.ndarray,
    psi_parameters: list = [(64, 4, 64)],
    markov_bins: int = 32,
    stft_parameters: list = [(128, 64)],
) -> np.ndarray:
    images = []

    for tau, m, bins in psi_parameters:
        img = phase_space_image(a, tau=tau, m=m, bins=bins)
        images.append(img)

    mtf = MarkovTransitionField(n_bins=markov_bins)
    images.append(mtf.fit_transform(a.reshape(1, -1))[0])

    rp = RecurrencePlot()
    images.append(rp.fit_transform(a.reshape(1, -1))[0])

    gaf0 = GramianAngularField(method="summation")
    images.append(gaf0.fit_transform(a.reshape(1, -1))[0])

    gaf1 = GramianAngularField(method="difference")
    images.append(gaf1.fit_transform(a.reshape(1, -1))[0])

    for nperseg, noverlap in stft_parameters:
        _, _, Zxx = stft(a, nperseg=nperseg, noverlap=noverlap)
        images.append(np.abs(Zxx))

    img_plain = np.tile(a, (len(a), 1))[np.newaxis, ...]
    images.append(img_plain[0])

    vector = []

    for img in images:
        img = img.astype(np.float32)

        if img.ndim > 2:
            img = img.squeeze()

        m = cv2.moments(img)
        hu = cv2.HuMoments(m).flatten()

        # hu = np.sign(hu) * np.log1p(np.abs(hu))

        vector.extend(hu.tolist())

    vector = np.array(vector)
    vector = np.nan_to_num(vector, nan=0.0)

    return np.array(vector)
