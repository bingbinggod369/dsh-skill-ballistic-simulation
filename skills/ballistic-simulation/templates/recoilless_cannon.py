# -*- coding: utf-8 -*-
"""无后坐炮（无坐力炮）内弹道求解器 —— numpy 手写 RK4。

模型出处：张小兵《枪炮内弹道学》P275 无后坐炮内弹道模型，参数取 P277 的 82mm 无后坐炮。
与经典内弹道的区别：炮尾开口带拉瓦尔喷管，部分燃气向后喷出抵消后坐力，
故多了两个状态变量 eta(相对流量=喷出质量/装药量) 与 tau(相对温度=膛温/爆温)。

状态变量 [Z, psi, l, v, eta, tau]，6 个一阶 ODE：
  dZ/dt   = (vj/l0)*sqrt(theta/2B)*(p/(f*Delta))^n     燃速定律(仅 Z<1)
  dpsi/dt = chi*(1+2*lambda*Z)*dZ/dt                    形状函数(仅 psi<1)
  dl/dt   = v
  dv/dt   = S*p/(varphi*m)                              弹丸运动方程
  deta/dt = CA*vj*Sj*p/(f*omega*sqrt(tau))              喷管相对质量流量
  dtau/dt = [(1-tau)*dpsi - theta*varphi*m*v*dv/(f*omega) - theta*tau*deta]/(psi-eta)
辅助：p = f*omega*tau*(psi-eta)/(S*(l+l_psi))；l_psi = l0*[1-Delta/rho_p*(1-psi)-alpha*Delta*(psi-eta)]
"""
from __future__ import annotations
import numpy as np


def solve(params, dt=1e-7, max_steps=2_000_000):
    V0 = params["chamber_volume"]; LG = params["barrel_length"]; omega = params["propellant_mass"]
    f = params["impetus_f"]; lam = params["form_lambda"]; varphi = params["sec_work_phi"]
    B = params["loading_param"]; vj = params["limit_velocity"]; S = params["bore_area"]
    Sj = params["nozzle_throat_area"]; m = params["projectile_mass"]; POM = params["nozzle_open_pressure"]
    alpha = params["covolume_alpha"]; chi = params["form_chi"]; n = params["burn_rate_n"]
    CA = params["flow_coeff"]; rho_p = params["propellant_density"]

    delta = omega / V0; l0 = V0 / S
    theta = 2.0 * f * omega / (m * varphi * vj * vj)

    def l_psi(psi, eta):
        return l0 * (1.0 - delta / rho_p * (1.0 - psi) - alpha * delta * (psi - eta))

    def pressure(psi, eta, tau, l):
        return f * omega * tau * (psi - eta) / (S * (l + l_psi(psi, eta)))

    def rhs(y):
        Z, psi, l, v, eta, tau = y
        p = pressure(psi, eta, tau, l)
        dZ = (vj / l0) * np.sqrt(theta / (2.0 * B)) * (p / (f * delta)) ** n if Z < 1.0 else 0.0
        dpsi = chi * (1.0 + 2.0 * lam * Z) * dZ if psi < 1.0 else 0.0
        dl = v
        dv = S * p / (varphi * m)
        deta = CA * vj * Sj * p / (f * omega * np.sqrt(tau))
        dtau = (1.0 / (psi - eta)) * ((1.0 - tau) * dpsi - theta * varphi * m * v * dv / (f * omega) - theta * tau * deta)
        return np.array([dZ, dpsi, dl, dv, deta, dtau])

    psi_0 = (1.0 / delta - 1.0 / rho_p) / (f / POM + alpha - 1.0 / rho_p)
    Z_0 = (np.sqrt(1.0 + 4.0 * lam * psi_0 / chi) - 1.0) / (2.0 * lam)
    y = np.array([Z_0, psi_0, 0.0, 0.0, 0.0, 1.0]); t = 0.0
    pmax = 0.0
    series = {"t": [], "l": [], "v": [], "p": [], "eta": [], "tau": []}
    while y[2] < LG and t < 0.02:
        k1 = rhs(y); k2 = rhs(y + 0.5 * dt * k1); k3 = rhs(y + 0.5 * dt * k2); k4 = rhs(y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4); t += dt
        p = pressure(y[1], y[4], y[5], y[2])
        if p > pmax: pmax = p
        series["t"].append(t); series["l"].append(y[2]); series["v"].append(y[3])
        series["p"].append(p); series["eta"].append(y[4]); series["tau"].append(y[5])
    return {"pmax": pmax, "v_muzzle": y[3], "t_muzzle": t, "p_muzzle": series["p"][-1],
            "eta_muzzle": y[4], "tau_muzzle": y[5], "series": series}


def example_params():
    """82mm 无后坐炮（《枪炮内弹道学》P277）。"""
    return dict(chamber_volume=3.022e-3, barrel_length=1.02, propellant_mass=0.53,
                impetus_f=1000e3, form_lambda=-0.2, sec_work_phi=1.05, loading_param=0.77,
                limit_velocity=1048.9, bore_area=0.5281e-2, nozzle_throat_area=0.389e-2,
                projectile_mass=2.81, nozzle_open_pressure=5.0e6, covolume_alpha=1.0e-3,
                form_chi=1.25, burn_rate_n=0.69, flow_coeff=0.652, propellant_density=1.6e3)


if __name__ == "__main__":
    r = solve(example_params())
    print("pmax=%.2f MPa, v_muzzle=%.2f m/s, t_muzzle=%.3f ms, p_muzzle=%.2f MPa, eta_m=%.3f" % (
        r["pmax"]/1e6, r["v_muzzle"], r["t_muzzle"]*1e3, r["p_muzzle"]/1e6, r["eta_muzzle"]))