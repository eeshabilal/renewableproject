import numpy as np
import matplotlib.pyplot as plt
import math

def main():

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""
    " Comment or Uncomment Depending on what we're plotting "
    #N = np.linspace(0, 365, 365)    # Day number where Jan 1st is 1
    N = np.array([55])   # Edit for plots of certain days

    t = np.linspace(0, 24, 96)  # 24-hour day split into 15 minute or .25 hour increments
    #t = np.array([11.5])

    beta = 22 # Panel angle
    gamma = 46 # Panel azimuthal angle
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""

    daily_outputs = np.zeros((np.size(N), np.size(t))) # Note that the day in index 0 is Jan 1
    i = 0
    for day in N:
        daily_outputs[i] = power_gen(day, t, beta, gamma)
        i += 1

    plt.plot(t, daily_outputs[0])

    plt.xlabel('Time (hours)')
    plt.xticks(np.arange(0, 25, 1))

    plt.ylabel('Power (Watts)')
    plt.title('Palmer Event Center: Total Power Output (Feb 24)')

    plt.show()

def power_gen(N, t, beta, gamma):

    # constants
    L = 30.26 # deg Latitude of Austin
    altitude = .149 # km Altitude of Austin
    panel_eff = .157
    A = 1.64 * .99 # m^2
    I_0 = extraterrestrial_radiation(N)
    delta = solar_declination_angle(N)
    array_size = np.size(t)

    solar_times = np.zeros(array_size)
    omega = np.zeros(array_size)
    theta_z = np.zeros(array_size)
    alpha = np.zeros(array_size)
    gamma_s = np.zeros(array_size)
    theta_i = np.zeros(array_size)
    tau_b = np.zeros(array_size)
    tau_d = np.zeros(array_size)
    I_cb = np.zeros(array_size)
    I_cd = np.zeros(array_size)
    I = np.zeros(array_size)
    Wdot_elec = np.zeros(array_size)

    i = 0
    for hour in t:
        solar_times[i] = solar_time(N, hour)
        omega[i] = solar_hour_angle(solar_times[i])

        theta_z[i] = zenith_angle(L, delta, omega[i])

        if theta_z[i] < math.pi/2:
            alpha[i] = 90 - math.degrees(theta_z[i])
            gamma_s[i] = solar_azimuth_angle(delta, omega[i], alpha[i])
            theta_i[i] = angle_of_incidence(alpha[i], beta, gamma, gamma_s[i])

            if theta_i[i] < math.pi/2:
                tau_b[i] = beam_transmissivity(N, theta_z[i], altitude)
                tau_d[i] = diffuse_transmittivity(tau_b[i])
                I_cd[i] = diffuse_radiation(I_0, theta_z[i], tau_d[i], beta)
                I_cb[i] = beam_radiation(I_0, tau_b[i], theta_i[i])
                I[i] = I_cd[i] + I_cb[i]
                Wdot_elec[i] = I[i] * panel_eff * A
            else:
                Wdot_elec[i] = 0

        else:
            Wdot_elec[i] = 0

        i += 1

    return Wdot_elec

def solar_time(N, standard_time):
    # N is day number, standard_time is in hours
    # Returns the solar time in hours

    tau = math.radians( 360 * N / 365)
    long_std = 90 # deg
    long_loc = 97.74 # deg
    ET = (
        - 7.3412 * math.sin(tau) + .4944 * math.cos(tau)
        - 9.3795 * math.sin(2 * tau) - 3.2568 * math.cos(2 * tau)
        - .3179 * math.sin(3 * tau) - .0774 * math.cos(3 * tau)
        - .1739 * math.sin(4 * tau) - .1283 * math.cos(4 * tau)
    )

    return standard_time + (4 * (long_std - long_loc) + ET) / 60  # In hours

def solar_hour_angle(solar_time): # omega
    return solar_time * 15 - 180 # deg

def solar_declination_angle(N): # delta
    theta = math.radians(360 * ((284 + N) / 365))
    return 23.45 * math.sin(theta) # deg

def zenith_angle(L, delta, omega): # theta_z
    L = math.radians(L)
    delta = math.radians(delta)
    omega = math.radians(omega)
    return math.acos(math.sin(L) * math.sin(delta) + math.cos(L) * math.cos(delta) * math.cos(omega)) # radians

def solar_azimuth_angle(delta, omega, alpha): # gamma_s
    delta = math.radians(delta)
    omega = math.radians(omega)
    alpha = math.radians(alpha)
    return math.degrees(math.asin(math.cos(delta) * math.sin(omega) / math.cos(alpha))) # degrees

def angle_of_incidence(alpha, beta, gamma, gamma_s): # theta_i
    alpha = math.radians(alpha)
    beta = math.radians(beta)
    gamma = math.radians(gamma)
    gamma_s = math.radians(gamma_s)
    return math.acos(math.sin(alpha) * math.cos(beta) + math.cos(alpha) * math.sin(beta) * math.cos(gamma - gamma_s)) # rad

def beam_transmissivity(N, theta_z, A): # tau_b
    # theta_z = math.radians(theta_z) # Uncomment this when testing hand calcs with degrees.
    if (N >= 355) or (N <= 79):  # Winter range
        r0, r1, rk = 1.03, 1.01, 1.00
    elif 172 <= N <= 265:  # Summer range
        r0, r1, rk = 0.97, 0.99, 1.02
    else:  # Spring/Fall (Standard)
        r0, r1, rk = 1.00, 1.00, 1.00

    a0_star = 0.4237 - 0.008216 * (6 - A) ** 2
    a1_star = 0.5055 + 0.00595 * (6.5 - A) ** 2
    k_star = 0.2711 + 0.01858 * (2.5 - A) ** 2

    a0 = r0 * a0_star
    a1 = r1 * a1_star
    k = rk * k_star

    return a0 + a1 * math.exp(-k / math.cos(theta_z))

def diffuse_transmittivity(tau_b): # tau_d
    return .271 - .294 * tau_b

def extraterrestrial_radiation(N): # I_0
    solar_constant = 1368 # W/m^2
    return solar_constant * (1 + .034 * math.cos(2 * math.pi * (N - 3) / 365))

def diffuse_radiation(I_0, theta_z, tau_d, beta):
    # theta_z = math.radians(theta_z) # uncomment for testing hand calcs w degrees
    beta = math.radians(beta)
    return I_0 * math.cos(theta_z) * tau_d * ((1 + math.cos(beta)) / 2)

def beam_radiation(I_0, tau_b, theta_i):
    # theta_z = math.radians(theta_z) # uncomment for testing hand calcs w degrees
    return I_0 * tau_b * math.cos(theta_i)

if __name__ == '__main__':
    main()