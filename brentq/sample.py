from astropy import units as u
from poliastro.bodies import Earth, Sun, Venus
import numpy as np
from poliastro.twobody import Orbit
from poliastro.util import norm

from astropy.time import Time
from poliastro import iod
from poliastro.threebody.flybys import compute_flyby
from brentq import brentq
import matplotlib.pyplot as plt

T_ref = 150 * u.day
k = Sun.k
a_ref = np.cbrt(k * T_ref**2 / (4 * np.pi**2)).to(u.km)
energy_ref = (-k / (2 * a_ref)).to(u.J / u.kg)
flyby_1_time = Time("2018-09-28", scale="tdb")
r_mag_ref = norm(Orbit.from_body_ephem(Venus, epoch=flyby_1_time).r)
v_mag_ref = np.sqrt(2 * k / r_mag_ref - k / a_ref)
d_launch = Time("2018-08-11", scale="tdb")
ss0 = Orbit.from_body_ephem(Earth, d_launch)
ss1 = Orbit.from_body_ephem(Venus, epoch=flyby_1_time)
tof = flyby_1_time - d_launch
(v0, v1_pre), = iod.lambert(Sun.k, ss0.r, ss1.r, tof.to(u.s))
V = Orbit.from_body_ephem(Venus, epoch=flyby_1_time).v
h = 2548 * u.km
d_flyby_1 = Venus.R + h
V_2_v_, delta_ = compute_flyby(v1_pre, V, Venus.k, d_flyby_1)
theta_range = np.linspace(0, 2 * np.pi)

def func(theta):
    V_2_v, _ = compute_flyby(v1_pre, V, Venus.k, d_flyby_1, theta * u.rad)
    ss_1 = Orbit.from_vectors(Sun, ss1.r, V_2_v, epoch=flyby_1_time)
    return (ss_1.period - T_ref).to(u.day).value

plt.plot(theta_range, [func(theta) for theta in theta_range])
plt.axhline(0, color="k", linestyle="dashed");
theta_opt_a = brentq(func, 0, 1) * u.rad
print(theta_opt_a.to(u.deg))
theta_opt_b = brentq(func, 4, 5) * u.rad
print(theta_opt_b.to(u.deg))