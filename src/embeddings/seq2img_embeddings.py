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
    max_size: int = 10_000,  # to avoid memory overflow
    log_values: bool = False,
) -> tuple[np.ndarray, list[str]]:
    images = []
    names = []
    final_names = []

    if len(a) > max_size:
        factor = len(a) // max_size
        a = a[: len(a) // factor * factor].reshape(-1, factor).mean(axis=1)

    for tau, m, bins in psi_parameters:
        img = phase_space_image(a, tau=tau, m=m, bins=bins)
        images.append(img)
        names.append(f"PSI_tau{tau}_m{m}_bins{bins}")

    mtf = MarkovTransitionField(n_bins=markov_bins)
    images.append(mtf.fit_transform(a.reshape(1, -1))[0])
    names.append(f"MTF_bins{markov_bins}")

    rp = RecurrencePlot()
    images.append(rp.fit_transform(a.reshape(1, -1))[0])
    names.append("RP")

    gaf0 = GramianAngularField(method="summation")
    images.append(gaf0.fit_transform(a.reshape(1, -1))[0])
    names.append("GAF_sum")

    gaf1 = GramianAngularField(method="difference")
    images.append(gaf1.fit_transform(a.reshape(1, -1))[0])
    names.append("GAF_diff")

    for nperseg, noverlap in stft_parameters:
        _, _, Zxx = stft(a, nperseg=nperseg, noverlap=noverlap)
        images.append(np.abs(Zxx))
        names.append(f"STFT_nperseg{nperseg}_noverlap{noverlap}")

    img_plain = np.tile(a, (len(a), 1))[np.newaxis, ...]
    images.append(img_plain[0])
    names.append("Plain_Tile")

    vector = []

    for i, img in enumerate(images):
        img = img.astype(np.float32)

        if img.ndim > 2:
            img = img.squeeze()

        m = cv2.moments(img)
        hu = cv2.HuMoments(m).flatten()

        if log_values:
            hu = np.sign(hu) * np.log1p(np.abs(hu))

        vector.extend(hu.tolist())
        final_names.extend([f"{names[i]}_Hu{j+1}" for j in range(len(hu))])

    vector = np.array(vector)
    vector = np.nan_to_num(vector, nan=0.0)

    return np.array(vector), final_names
