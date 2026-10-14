"""经典内弹道（集总参数）求解器 —— 复现张小兵《枪炮内弹道学》P109 模型 + P114 57mm 高射炮数据。

与用户基准代码 internal_ballistics.py 的模型/参数完全一致：
  状态变量 3 个一阶 ODE：y = [Z, l, v]
    dZ/dt = (u1/e1) * p^n        仅 Z < Zk 时燃烧
    dl/dt = v                    行程定义
    dv/dt = S*p / (phi*m)        弹丸运动方程
  形状函数 psi(Z) 分三段（含分裂燃烧）：
    Z < 1      : psi = chi*Z*(1 + lam*Z + mu*Z^2)
    1 <= Z < Zk: psi = chis*(Z/Zk)*(1 + lams*(Z/Zk))
    Z >= Zk    : psi = 1
  分裂点：Zk = (e1 + rou)/e1，rou = 0.2956*(e1b/2 + e1)
  自由容积缩径长：l_psi = l0*[1 - Delta/rho_p - Delta*(alpha - 1/rho_p)*psi]
  压强：p = [f*omega*psi - (theta/2)*phi*m*v^2] / [S*(l + l_psi)]

本实现用固定步长 RK4（仅 numpy），与 scipy solve_ivp RK45（等价 MATLAB ode45）
在足够小步长下数值一致；不依赖 scipy/matplotlib。
"""
from __future__ import annotations

import numpy as np


def solve(params, dt=1e-7, max_steps=2_000_000):
    """求解经典内弹道，返回关键结果与时间序列。

    params（SI 单位）：
      propellant_density  火药密度 rho_p [kg/m^3]
      propellant_mass     装药量 omega [kg]
      impetus_f           火药力 f [J/kg]
      theta               k-1（k 为绝热指数）
      covolume_alpha      余容 alpha [m^3/kg]
      burn_rate_a         燃速系数 u1
      burn_rate_n         燃速指数 n
      half_web            弧厚 e1 [m]
      half_web2           弧厚 e1b [m]
      form_chi            chi
      form_lambda         lam
      form_mu             mu
      form_chi_s          chi_s（分裂燃烧段）
      form_lambda_s       lam_s（分裂燃烧段）
      projectile_mass     弹丸质量 m [kg]
      bore_area           炮膛横截面积 S [m^2]
      chamber_volume      药室容积 V0 [m^3]
      barrel_length       炮管行程长 l_g [m]
      sec_work_phi        次要功系数 phi
      ignition_pressure   点火压力 p0 [Pa]
    """
    rho_p = params["propellant_density"]
    omega = params["propellant_mass"]
    f = params["impetus_f"]
    theta = params["theta"]
    alpha = params["covolume_alpha"]
    u1 = params["burn_rate_a"]
    n = params["burn_rate_n"]
    e1 = params["half_web"]
    e1b = params["half_web2"]
    chi = params["form_chi"]
    lam = params["form_lambda"]
    mu = params["form_mu"]
    chis = params["form_chi_s"]
    lams = params["form_lambda_s"]
    m = params["projectile_mass"]
    S = params["bore_area"]
    V0 = params["chamber_volume"]
    LG = params["barrel_length"]
    phi = params["sec_work_phi"]
    P0 = params["ignition_pressure"]

    rou = 0.2956 * (e1b / 2.0 + e1)          # 药粒初始体积折合的弧厚增量
    Zk = (e1 + rou) / e1                     # 分裂点相对弧厚
    delta = omega / V0                       # 装填密度
    l0 = V0 / S                              # 药室容积缩径长

    def psi_of(Z):
        if Z < 1.0:
            return chi * Z * (1.0 + lam * Z + mu * Z * Z)
        if Z < Zk:
            zk = Z / Zk
            return chis * zk * (1.0 + lams * zk)
        return 1.0

    def l_psi(psi):
        return l0 * (1.0 - delta / rho_p - delta * (alpha - 1.0 / rho_p) * psi)

    def pressure(psi, l, v):
        return (f * omega * psi - theta / 2.0 * phi * m * v * v) / (S * (l + l_psi(psi)))

    def rhs(y):
        Z, l, v = y
        psi = psi_of(Z)
        p = pressure(psi, l, v)
        dZ = (u1 / e1) * (p ** n) if Z < Zk else 0.0
        return np.array([dZ, v, S * p / (phi * m)])

    # 点火瞬间初值
    psi_0 = (1.0 / delta - 1.0 / rho_p) / (f / P0 + (alpha - 1.0 / rho_p))
    Z_0 = (np.sqrt(1.0 + 4.0 * lam * psi_0 / chi) - 1.0) / (2.0 * lam)
    y = np.array([Z_0, 0.0, 0.0])
    t = 0.0

    series = {"t": [t], "Z": [Z_0], "l": [0.0], "v": [0.0]}
    psi = psi_of(Z_0)
    p = pressure(psi, 0.0, 0.0)
    series_psi = [psi]
    series_P = [p]
    pmax = p
    idx_pmax = 0
    t_burnout = None

    steps = 0
    while y[1] < LG and steps < max_steps:
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * dt * k1)
        k3 = rhs(y + 0.5 * dt * k2)
        k4 = rhs(y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt
        steps += 1
        psi = psi_of(y[0])
        p = pressure(psi, y[1], y[2])
        if t_burnout is None and y[0] >= Zk:
            t_burnout = t
        series["t"].append(t)
        series["Z"].append(y[0])
        series["l"].append(y[1])
        series["v"].append(y[2])
        series_psi.append(psi)
        series_P.append(p)
        if p > pmax:
            pmax = p
            idx_pmax = len(series["t"]) - 1

    series["psi"] = series_psi
    series["P"] = series_P

    return {
        "P_max": pmax,
        "l_at_Pmax": series["l"][idx_pmax],
        "t_Pmax": series["t"][idx_pmax],
        "v_muzzle": y[2],
        "P_muzzle": series_P[-1],
        "t_muzzle": t,
        "t_burnout": t_burnout,
        "Z_end": y[0],
        "psi_end": psi,
        "series": series,
    }


def interp_P_at_l(result, l_target: float) -> float:
    """按行程线性插值得到某行程处的压强（用于 checkpoints 对比）。"""
    l = result["series"]["l"]
    P = result["series"]["P"]
    return float(np.interp(l_target, l, P))


def example_params() -> dict:
    """示例参数：张小兵《枪炮内弹道学》P114 的 57mm 高射炮数据（n 由书中 0.83 改为 0.77225）。"""
    return {
        "propellant_density": 1600.0,
        "propellant_mass": 1.16,
        "impetus_f": 950000.0,
        "theta": 0.25,
        "covolume_alpha": 0.001,
        "burn_rate_a": 5.127e-8,
        "burn_rate_n": 0.77225,
        "half_web": 0.00055,
        "half_web2": 0.00055,
        "form_chi": 0.75,
        "form_lambda": 0.12,
        "form_mu": 0.0,
        "form_chi_s": 1.696,
        "form_lambda_s": -0.4104,
        "projectile_mass": 2.8,
        "bore_area": 0.00266,
        "chamber_volume": 0.00151,
        "barrel_length": 3.624,
        "sec_work_phi": 1.168,
        "ignition_pressure": 3.0e7,
    }


if __name__ == "__main__":
    res = solve(example_params())
    print("装填密度 Delta    = %.2f kg/m^3" % (1.16 / 0.00151))
    print("最大膛压 pmax     = %.2f MPa" % (res["P_max"] / 1e6))
    print("最大膛压时刻      = %.3f ms" % (res["t_Pmax"] * 1e3))
    print("最大膛压时行程    = %.3f dm" % (res["l_at_Pmax"] * 10))
    print("炮口速度 vm       = %.2f m/s" % res["v_muzzle"])
    print("炮口压力 pm       = %.2f MPa" % (res["P_muzzle"] / 1e6))
    print("炮口时刻 tm       = %.3f ms" % (res["t_muzzle"] * 1e3))
    print("燃尽时刻          = %.3f ms" % (res["t_burnout"] * 1e3 if res["t_burnout"] else float("nan")))