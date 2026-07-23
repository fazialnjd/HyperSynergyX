import numpy as np
import pandas as pd


def load_incidence(path):
    if path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    names = df.iloc[:, 0].astype(str).tolist()
    H = df.iloc[:, 1:].to_numpy(dtype=float)
    return names, H


def compute_transition_matrix(H, Sim):
    De = np.diag(np.sum(H, axis=0))
    Dv = np.diag(np.sum(H @ De, axis=1))
    P = np.linalg.pinv(Dv) @ H @ np.linalg.pinv(De) @ H.T
    P = P * Sim
    return P


def dbrwh_scores(H, alpha=0.8, Sim=None, restart_bias=True, attraction_bias=True):
    cc = H.shape[0]
    if Sim is None:
        Sim = np.ones((cc, cc), dtype=float)
    P = compute_transition_matrix(H, Sim)
    I = np.eye(cc)

    M = np.linalg.pinv(I - alpha * P.T)

    # tensor 3 dimentional
    scores = np.zeros((cc, cc, cc), dtype=float)

    for k in range(cc):
        rk = np.zeros(cc, dtype=float)
        if attraction_bias:
            rk[k] = 1.0
        for i in range(cc):
            r0 = np.zeros(cc, dtype=float)
            if restart_bias:
                r0[i] = 1.0

            bias_vec = r0 + rk
            count = (1 if restart_bias else 0) + (1 if attraction_bias else 0)

            if count > 0:
                v = (1 - alpha) * (M @ (bias_vec / count))
            else:
                v = np.zeros(cc, dtype=float)  # other drugs fill with zero value
            scores[:, i, k] = (
                v  # vector [0.3,0.2,0.1,0.4], sim of all must be 1. tozie ehtemal medeh chon
            )

    synergy = np.zeros_like(scores)
    for i in range(cc):
        for j in range(cc):
            for k in range(cc):
                synergy[i, j, k] = (
                    scores[i, j, k] + scores[j, k, i] + scores[k, i, j]
                ) / 3.0

    # synergy[i, j, k]
    # = میزان تاثیرپذیری راس j از حفت (i,k) وقتی i نقطه شروع و k نقطه جاذب باشد
    # مقدارش یه اسکالر هست.

    """
    حذف درایه‌های دارای تکرار رأس
    هر جایی که حداقل دو ایندکس برابر باشند، مقدار synergy را صفر می‌کند.
یعنی فقط سه‌تایی‌های با سه عضو متمایز غیرصفر می‌مانند.     
    """
    for ii in range(cc):
        synergy[ii, :, ii] = 0
        synergy[ii, ii, :] = 0
        synergy[:, ii, ii] = 0

    # normalization. between 0,1
    mn = synergy.min()
    mx = synergy.max()
    if mx > mn:
        norm = (synergy - mn) / (mx - mn)
    else:
        norm = synergy
    return norm


def combine_sims(S_list):
    if not S_list:
        return None
    S = None
    for s in S_list:
        if s is None:
            continue
        S = s if S is None else np.maximum(S, s)
    return S
