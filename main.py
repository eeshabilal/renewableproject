import numpy as np
import matplotlib.pyplot as plt
import math

def main():

    """"""""""""""""""""""""""" Control Panel """""""""""""""""""""""""""""

    # Relevant days and times
    # Feb 5, N = 36
    # Feb 24, N = 55
    # Jun 21, N = 172
    # Dec 21, N = 355
    # Solar noon @ 12.5 Austin Local Time

    ## Comment or Uncomment Depending on what we're plotting ##

    # For plots vs time of day on individual days
    N = np.array([36]) # Edit for plots of certain days
    t = np.linspace(0, 24, 96) # 24-hour day split into 15 minute or .25 hour increments
    day_name = 'Feb 5'

    # # For plots vs day of the year at individual times
    # N = np.linspace(0, 365, 365)  # Day number where Jan 1st is 1
    # t = np.array([12.5]) # This is local time
    # day_name = ''

    beta = 22 # Panel angle
    gamma = 46 # Panel azimuthal angle

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    total_system_power = np.zeros((np.size(N), np.size(t))) # W/m^2
    irradiance = np.zeros((np.size(N), np.size(t))) # W/m^2
    bd_ratio = np.zeros((np.size(N), np.size(t)))
    theta_i_noon = np.zeros((np.size(N), np.size(t))) # deg
    total_system_energy = np.zeros((np.size(N), np.size(t))) # kWh
    hours_in_day = np.linspace(0, 24, 96)

    i = 0
    for day in N:
        total_system_power[i] = simulate(day, t, beta, gamma)[0]
        irradiance[i] = simulate(day, t, beta, gamma)[1]
        bd_ratio[i]= simulate(day, t, beta, gamma)[2]
        theta_i_noon[i] = simulate(day, t, beta, gamma)[3]

        total_system_energy[i] = np.trapezoid(simulate(day, hours_in_day, beta, gamma)[0], hours_in_day)
        i += 1

    # simulate function indexing guide
    # simulate[0] = Power output for 1 panel in W/m^2
    # simulate[1] = Irradiance in W/m^2
    # simulate[2] = Ratio of beam irradiance to diffuse irradiance
    # simulate[3] = Angle of incidence in rads
    # simulate[4] = Total daily energy production for 1 panel in kWh

    # IMPORTANT: * Plot shapes are SUPER related to beta and gamma

    # Plots vs time of day
    if day_name:
        plot_solar_data(t, total_system_power[0], irradiance[0], day_name)
        plot_bd_ratio(t, bd_ratio[0], day_name) # *

    if day_name == 'Feb 5':
        plot_power_delivery(t, total_system_power[0], day_name)

    # Plots vs day of the year
    if not day_name:
        plot_theta_i(N, theta_i_noon) # *
        plot_energy(N, total_system_energy)




def simulate(N, t, beta, gamma):

    # constants
    L = 30.26 # deg Latitude of Austin
    altitude = .149 # km Altitude of Austin
    panel_eff = .157
    A = 1.64 * .99 # m^2
    I_0 = extraterrestrial_radiation(N) # W/m^2
    delta = solar_declination_angle(N) # deg
    array_size = np.size(t)

    solar_times = np.zeros(array_size) # h
    omega = np.zeros(array_size) # deg
    theta_z = np.zeros(array_size) # rad
    alpha = np.zeros(array_size) # deg
    gamma_s = np.zeros(array_size) # deg
    theta_i = np.zeros(array_size) # rad
    tau_b = np.zeros(array_size)
    tau_d = np.zeros(array_size)
    I_cb = np.zeros(array_size) # W/m^2
    I_cd = np.zeros(array_size) # W/m^2
    bd_ratio = np.zeros(array_size)
    I = np.zeros(array_size) # W/m^2
    Wdot_elec = np.zeros(array_size) # W/m^2

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
                bd_ratio[i] = I_cb[i] / I_cd[i]

            else:
                Wdot_elec[i] = 0
                bd_ratio[i] = 0

        else:
            Wdot_elec[i] = 0
            bd_ratio[i] = 0

        i += 1

    return Wdot_elec, I, bd_ratio, theta_i

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

def plot_solar_data(t, power_array, irradiance_array, day_name):
    # Plots Total System Power and Irradiance on a dual y-axis vs time of day

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left Axis: Total System Power (kW)
    color_power = 'tab:blue'
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Total System Power (kW/m^2)', color=color_power, fontweight='bold')
    ax1.plot(t, power_array * 960 / 1000, color=color_power, label='Total Power Output', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color_power)
    ax1.set_xticks(np.arange(0, 25, 1))
    ax1.grid(True, alpha=0.6)

    # Right Axis: Irradiance (kW/m2)
    ax2 = ax1.twinx()
    color_irr = 'tab:orange'
    ax2.set_ylabel('Irradiance (kW/m^2)', color=color_irr, fontweight='bold')
    ax2.plot(t, irradiance_array / 1000, color=color_irr, label='Irradiance', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color_irr)

    # Legends and Title
    plt.title(f'Power & Irradiance vs. Time ({day_name})')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)
    fig.tight_layout()

    plt.show()


def plot_bd_ratio(t, ratios, day_name):
    # Plots Beam-to-Diffuse ratio vs time of day

    plt.figure(figsize=(10, 6))
    plt.plot(t, ratios, color='darkgreen', linewidth=2, label='Beam/Diffuse Ratio')

    # Axis Labels and Title
    plt.xlabel('Time (hours)', fontweight='bold')
    plt.ylabel('Ratio', fontweight='bold')
    plt.title(f'Beam-to-Diffuse Radiation Ratio ({day_name})')
    plt.xticks(np.arange(0, 25, 1))

    plt.grid(True, alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plt.show()

def plot_theta_i(N, theta_i_noon):
    # Plots angle of incidence vs day of the year
    theta_i_noon *= 180 / math.pi

    plt.figure(figsize=(10, 6))
    plt.plot(N, theta_i_noon, color='r', linewidth=2, label='Angle of Incidence')

    plt.xlabel('Day', fontweight='bold')
    plt.ylabel('Angle of Incidence (degrees)', fontweight='bold')
    plt.title('Angle of Incidence at Solar Noon vs. Day of the Year')
    plt.xticks(np.arange(0, 366, 30))

    plt.grid(True, alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plt.show()

def plot_energy(N, energy):

    plt.figure(figsize=(10, 6))
    plt.plot(N, energy, color='c', linewidth=2, label='Daily Energy Production')

    plt.xlabel('Day', fontweight='bold')
    plt.ylabel('Energy Production (kWh)', fontweight='bold')
    plt.title('Daily Energy Production vs. Day of the Year')
    plt.xticks(np.arange(0, 366, 30))

    plt.grid(True, alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plt.show()

def plot_power_delivery(t, power_output, day_name):

    plt.figure(figsize=(10, 6))
    plt.plot(t, power_output * 960 / 1e6, color='m', linewidth=2, label='Power Output')

    plt.xlabel('Time (hours)', fontweight='bold')
    plt.ylabel('Power Output (MW)', fontweight='bold')
    plt.title(f'Power Output vs. Time of Day ({day_name})')
    plt.xticks(np.arange(0, 25, 1))

    plt.grid(True, alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()