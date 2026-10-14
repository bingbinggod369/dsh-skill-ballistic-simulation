# -*- coding: utf-8 -*-
"""压缩空气弹射系统内弹道模型求解器（复现《压缩空气弹射系统内弹道特性》）。

模型：兵工学报 2021,42(12)。状态向量 [U, m, m_h, T_h, W_T, V, v_m, l]，RK4 积分。
推断点（已记录）：过载=a/g；发射阀面积 S_e(t)=min(a·t^2, S_e,max)。
"""
from __future__ import annotations
import numpy as np


def solve(params, dt=1e-4):
    cv = params["cv"]; R = params["R"]; k = params["k"]
    S = params["S"]; ST = params["ST"]; phi1 = params["phi1"]; phi = params["phi"]
    g = params["g"]; mDmT = params["mDmT"]; L = params["L"]
    Vh = params["Vh"]; mh0 = params["mh0"]; Th0 = params["Th0"]
    V0 = params["V0"]; m0 = params["m0"]; T0 = params["T0"]
    a = params["a"]; Se_max = params["Se_max"]

    K0 = np.sqrt(k) * (2.0 / (k + 1.0)) ** ((k + 1.0) / (2.0 * (k - 1.0)))
    crit = (2.0 / (k + 1.0)) ** (k / (k - 1.0))
    U0 = m0 * cv * T0
    ph0 = mh0 * R * Th0 / Vh

    def Se(t):
        return min(a * t * t, Se_max)

    def deriv(y, t):
        U, m, mh, Th, WT, V, vm, l = y
        T = U / (cv * m)
        p = m * R * T / V
        ph = mh * R * Th / Vh
        ratio = p / ph
        Se_ = Se(t)
        if ratio <= crit:
            dmv = phi * ph * K0 * Se_ / np.sqrt(R * Th)
        else:
            inner = ratio ** (2.0 / k) - ratio ** ((k + 1.0) / k)
            if inner < 0:
                inner = 0.0
            dmv = phi * ph * Se_ / np.sqrt(R * Th) * np.sqrt(2.0 * k / (k - 1.0) * inner)
        dmh = -dmv
        dm = dmv
        dTh = (k - 1.0) * Th / mh * dmh
        dHi = -cv * Th * dmh - cv * mh * dTh
        dWT = ST * p * vm
        dV = S * vm
        dU = dHi - dWT
        thrust = ST * p
        res = phi1 * mDmT * g
        dvm = 0.0 if thrust < res else (thrust - res) / (phi1 * mDmT)
        dl = vm
        return np.array([dU, dm, dmh, dTh, dWT, dV, dvm, dl]), p, ph, dmv, dvm

    y = np.array([U0, m0, mh0, Th0, 0.0, V0, 0.0, 0.0])
    t = 0.0
    series = {"t": [], "l": [], "v": [], "p": [], "ph": [], "ov": [], "flow": [], "Se": []}
    max_ov = 0.0
    t_start = None

    def record(t, y, p, ph, dmv, dvm):
        ov = dvm / g
        series["t"].append(t); series["l"].append(y[7]); series["v"].append(y[6])
        series["p"].append(p); series["ph"].append(ph); series["ov"].append(ov)
        series["flow"].append(dmv); series["Se"].append(Se(t))

    _, p, ph, dmv, dvm = deriv(y, 0.0)
    record(0.0, y, p, ph, dmv, dvm)

    while y[7] < L and t < 5.0:
        k1, _, _, _, _ = deriv(y, t)
        k2, _, _, _, _ = deriv(y + 0.5 * dt * k1, t + 0.5 * dt)
        k3, _, _, _, _ = deriv(y + 0.5 * dt * k2, t + 0.5 * dt)
        k4, _, _, _, _ = deriv(y + dt * k3, t + dt)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        _, p, ph, dmv, dvm = deriv(y, t)
        record(t, y, p, ph, dmv, dvm)
        ov = dvm / g
        if ov > max_ov:
            max_ov = ov
        if t_start is None and y[6] > 1e-6:
            t_start = t

    t_muzzle = t
    v_muzzle = y[6]
    avg_ov = v_muzzle / (t_muzzle * g) if t_muzzle > 0 else 0.0
    return {
        "t_muzzle": t_muzzle, "v_muzzle": v_muzzle,
        "max_ov": max_ov, "avg_ov": avg_ov, "t_start": t_start,
        "series": {k: np.array(v) for k, v in series.items()},
    }


def base_params():
    return dict(cv=717.6, R=287.053, k=1.4, S=3.2349, ST=3.1416, phi1=1.1, phi=0.9,
                g=9.8, mDmT=57600.0, L=100.0, Vh=450.0, mh0=10521.1728, Th0=298.0,
                V0=4.0, m0=4.7380, T0=298.0, a=0.3, Se_max=0.6750)