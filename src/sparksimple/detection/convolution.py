"""sparksimple.detection.convolution — 3D FFT convolution reproduction of MATLAB conv3.m.

Mirrors MATLAB ``mfiles/conv3.m``.
"""

from __future__ import annotations

import math
import numpy as np


def conv3_matlab(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute 3D convolution using FFT matching MATLAB ``mfiles/conv3.m``.

    MATLAB reference (conv3.m):
        [a1,a2,a3] = size(A);
        [b1,b2,b3] = size(B);
        c = ([a1,a2,a3]-[b1,b2,b3])/2;
        B2 = padarray(B, ceil(c));
        dims = ceil(c) == c;
        if(dims(1)==0), B2(end,:,:) = []; end
        if(dims(2)==0), B2(:,end,:) = []; end
        if(dims(3)==0), B2(:,:,end) = []; end
        AA = fftn(A);
        BB = fftn(B2);
        CC = AA .* BB;
        C = real(fftshift(ifftn(CC)));

    Parameters
    ----------
    a : np.ndarray
        3D volume input A of shape ``(Y, X, T)``.
    b : np.ndarray
        3D kernel input B of shape ``(Kb_y, Kb_x, Kb_t)`` where ``size(A) >= size(B)``.

    Returns
    -------
    np.ndarray
        3D convolved volume of shape ``(Y, X, T)`` with dtype ``float64``.
    """
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError(
            f"Both A and B must be 3D arrays, got A.ndim={a.ndim}, B.ndim={b.ndim}"
        )

    a1, a2, a3 = a.shape
    b1, b2, b3 = b.shape

    if a1 < b1 or a2 < b2 or a3 < b3:
        raise ValueError(
            f"Input volume size {a.shape} must be >= kernel size {b.shape}"
        )

    c = [(a1 - b1) / 2.0, (a2 - b2) / 2.0, (a3 - b3) / 2.0]
    p = [int(math.ceil(ci)) for ci in c]

    pad_width = [(p[0], p[0]), (p[1], p[1]), (p[2], p[2])]
    b2 = np.pad(b.astype(np.float64), pad_width, mode="constant", constant_values=0.0)

    # Trim trailing elements if (a_i - b_i) was odd (matching dims(i) == 0 in MATLAB)
    if float(p[0]) != c[0]:
        b2 = b2[:-1, :, :]
    if float(p[1]) != c[1]:
        b2 = b2[:, :-1, :]
    if float(p[2]) != c[2]:
        b2 = b2[:, :, :-1]

    # Ensure shape parity
    assert b2.shape == (a1, a2, a3), f"Padded B shape {b2.shape} != A shape {(a1, a2, a3)}"

    aa = np.fft.fftn(a.astype(np.float64))
    bb = np.fft.fftn(b2)
    cc = aa * bb
    c_out = np.real(np.fft.fftshift(np.fft.ifftn(cc)))

    return c_out
