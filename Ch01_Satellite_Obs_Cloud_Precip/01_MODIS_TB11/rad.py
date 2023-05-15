import numpy as np

def RAD2BT(wl,rad):
	h = 6.626e-34  # Planck constant    [Js]
	k = 1.3806e-23 # Boltzman constant  [J/K] 
	c = 2.9979e+8  # Speed of light     [m/s]
	rad = rad*1e+6 # Radiance           [W/m^2/m/strad]
	wl = wl*1e-6   # um to m            [m]
	P1 = h * c / (wl * k)
	P2 = np.log(1+ (2 * h * (c**2)) / ((wl**5) * rad ))
	BT = P1 / P2
	return BT


