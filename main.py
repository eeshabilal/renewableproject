import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize_scalar
from datetime import datetime, timedelta

def main():

    """"""""""""""""""""""""""" Control Panel """""""""""""""""""""""""""""

    # Relevant days and times
    # Feb 5, N = 36
    # Feb 24, N = 55
    # Jun 21, N = 172
    # Dec 21, N = 355
    t_15min = np.linspace(0, 24, 96)
    t_5min = np.linspace(0, 24, 288) # 24 hours*12 increments per hour = 288 increments
    days_in_year = np.arange(1,366)

    ## Comment or Uncomment Depending on what we're plotting ##

    # For plots vs time of day on individual days
    N = np.array([36]) # Edit for plots of certain days
    day_name = 'Feb 5'

    # # For plots vs day of the year at individual times
    # N = np.linspace(0, 365, 365)  # Day number where Jan 1st is 1
    # day_name = ''

    beta = 22 # Panel angle for case 1 and 2
    gamma = 46 # Panel azimuthal angle

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    days_needed = np.unique(np.append(N, [172, 355])).tolist() # Adding june 21 and dec 21 for case 2 and 3 plots
    cleaned_2019_data = get_cleaned_solar_power_arrays('PEC 15 minute data for 2019.csv', days_needed)

    cleaned_2019_data = get_power_outputs_2019('PEC 15 minute data for 2019.csv', N)
    annual_actual_energy = get_annual_daily_energy_array('PEC 15 minute data for 2019.csv')
    cleaned_feb_data = get_power_outputs_2026('pec 15 minute data for 2.5.2026.csv')

    total_system_energy = np.zeros(len(N))  # kWh
    if not day_name:
        t = np.zeros(len(N))
        for i, day in enumerate(N):
            t[i] = local_time(day, 12)
        theta_i_noon = np.zeros(len(N)) # deg
    else:
        t = t_5min
        total_system_power = np.zeros((np.size(N), np.size(t))) # W
        irradiance = np.zeros((np.size(N), np.size(t))) # W/m^2
        bd_ratio = np.zeros((np.size(N), np.size(t)))

    i = 0
    for day in N:
        if day_name:
            # Plots against time of day
            res = simulate(day, t, beta, gamma)
            total_system_power[i] = res[0]
            irradiance[i] = res[1]
            bd_ratio[i] = res[2]
        else:
            # Plots against day of year
            res_noon = simulate(day, np.array([t[i]]), beta, gamma)

            theta_i_noon[i] = res_noon[3][0]

        p_day_array = simulate(day, t_5min, beta, gamma)[0]
        total_system_energy[i] = np.trapezoid(p_day_array, t_5min)
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
        plot_solar_data(t, total_system_power[0], cleaned_2019_data[N[0]], irradiance[0], day_name)
        plot_bd_ratio(t, bd_ratio[0], day_name) # *

    if day_name == 'Feb 5':
        plot_power_delivery(t, total_system_power[0], cleaned_feb_data, day_name)

    # Plots vs day of the year
    if not day_name:
        plot_theta_i(N, theta_i_noon) # *
        plot_energy(N, total_system_energy/1000, annual_actual_energy/1000) # Converts kWh to MWh for plotting
    
    # -----  Annual MWh Calculation -------
    energy_case1_annual = 0
    energy_case3_annual = 0
    daily_mwh_case3 = []

    for day in days_in_year:
        # Case 1 Daily Energy
        p1 = simulate(day,t_5min,beta,gamma)[0]
        e1 = np.trapezoid(p1 * 960, t_5min) / 1e6 # MWh for 960 panels
        energy_case1_annual += e1

        # Case 3 Daily Energy
        p3, irr3, beta3 = simulate_case_3(day, t_5min, gamma)
        e3 = np.trapezoid(p3 * 960, t_5min) / 1e6 # MWh for 960 panels
        energy_case3_annual += e3
        daily_mwh_case3.append(e3)
    
    # Output Table of Annual Energy Production for Case 1 and Case 3 compared to 2019

    energy_actual_annual = np.sum(annual_actual_energy) / 1000 # MWh of 2019 actual energy
    print("\n" + "="*45)
    print(f"{'Case Study':<25} | {'Annual Energy (MWh)':<15}")
    print("-" * 45)
    print(f"{'Case 1 (Base; Fixed Tilt)':<25} | {energy_case1_annual:>15.4f}")
    print(f"{'Case 3 (Tracking)':<25} | {energy_case3_annual:>15.4f}")
    print(f"{'2019 Actual Data':<25} | {energy_actual_annual:>15.4f}")
    print("="*45)

    # ---- For Case 2: Effect of Panel Temperature ---- #
    
    # Plot irradiance and total system power - December 21 and June 21
    temp_sets = {
        "Dec 21 - Case 2":{"day": 355, "temps": [0,25,45]},
        "Jun 21 - Case 2":{"day": 172, "temps": [25,45,85]}
    }

    for label, data in temp_sets.items():
        fig, axis1 = plt.subplots(figsize=(10, 6))
        axis2 = axis1.twinx()

        day = data["day"]
        irr_plot = None # To ensure irradiance is only plotted once per day since it doesn't change with temperature

        for temp in data["temps"]:
            power_temp, irr_temp, _, _ = simulate(day, t_5min, beta, gamma, T_cell=temp)
            axis1.plot(t_5min, power_temp * 960 / 1000, label=f'Power at {temp}°C')

            if irr_plot is None:
                irr_plot = irr_temp
        axis1.plot(t_15min, cleaned_2019_data[day], linestyle=':', label='2019 Power')
        axis2.plot(t_5min, irr_plot / 1000, color='orange', linestyle='--', label='Irradiance')
        axis1.set_xlabel('Time of Day (hours)', fontweight='bold')
        axis1.set_ylabel('Total System Power Delivery (kW)', color='blue', fontweight='bold')
        axis2.set_ylabel('Irradiance (kW/m^2)', color='orange', fontweight='bold')
        plt.title(f"Irradiance and System Power: {label}", fontweight='bold')
        axis1.grid(True, alpha=0.6)

        lines1, labels1 = axis1.get_legend_handles_labels()
        lines2, labels2 = axis2.get_legend_handles_labels()
        axis1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        plt.show()
    
    # Plot total daily energy production vs day of the year
    plt.figure(figsize=(10, 6))
    temps = [0, 25, 45, 85]
    for temp in temps:
        daily_mwh_case2 = []
        for day in days_in_year:
            power_temp, _, _, _ = simulate(day, t_5min, beta, gamma, T_cell=temp)
            e2 = np.trapezoid(power_temp * 960, t_5min) / 1e6 # MWh for 960 panels
            daily_mwh_case2.append(e2)
        plt.plot(days_in_year, daily_mwh_case2, label=f'Temp = {temp}°C')
    plt.plot(days_in_year, annual_actual_energy/1000, label='2019 Energy')
    plt.xlabel('Day of the Year', fontweight='bold')
    plt.ylabel('Daily Energy Production (MWh)', fontweight='bold')
    plt.title('Daily Energy Production vs. Day of the Year for Case 2', fontweight='bold')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # ---- For Case 3: Optimized Vertical Tracking Angle ----
    
    # Irradiance and total system power delivery - December 21
    power_dec, irr_dec, beta_dec = simulate_case_3(355, t_5min, gamma)
    plot_solar_data(t_5min, power_dec, cleaned_2019_data[355], irr_dec, 'December 21 - Case 3 Optimized Vertical Tracking')

    # Irradiance and total system power delivery - June 21
    power_jun, irr_jun, beta_jun = simulate_case_3(172, t_5min, gamma)
    plot_solar_data(t_5min, power_jun, cleaned_2019_data[172], irr_jun, 'June 21 - Case 3 Optimized Vertical Tracking')

    # Plot Tilt Angle vs time of day - June 21
    plt.figure(figsize=(10, 6))
    plt.plot(t_5min, beta_jun, color='purple', linewidth=2, label='Optimized Beta Angle')
    plt.title('Case 3 Optimized Panel Angle vs Time of Day (June 21)')
    plt.ylabel("Tilt Angle (degrees)")
    plt.xlabel("Time (hours)")
    plt.grid(True, alpha=0.6)
    plt.show()

    # Plot total daily energy production vs day of the year
    plot_energy(days_in_year, daily_mwh_case3, annual_actual_energy/1000, title='Daily Energy Production vs. Day of the Year for Case 3')


def simulate(N, t, beta, gamma, T_cell = 25):

    # constants
    L = 30.26 # deg Latitude of Austin
    altitude = .149 # km Altitude of Austin
    panel_eff = .157 # efficiency at 25C cell temperature
    inverter_eff = .965
    power_temp_coeff = -0.0045 # Power temperature coefficient from 25C 
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

                P_25C = I[i] * panel_eff * A *inverter_eff # Power at 25C cell temperature
                Wdot_elec[i] = P_25C * (1 + power_temp_coeff * (T_cell - 25)) # Adjusted for cell temperature
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

def local_time(N, solar_time):
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

    return solar_time - (4 * (long_std - long_loc) + ET) / 60  # In hours

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

def plot_solar_data(t, power_array, real_power_array, irradiance_array, day_name):
    # Plots Total System Power and Irradiance on a dual y-axis vs time of day
    t_15min = np.linspace(0, 24, 96)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left Axis: Total System Power (kW)
    color_power = 'tab:blue'
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Total System Power Delivery (kW)', color=color_power, fontweight='bold')
    ax1.plot(t, power_array * 960 / 1000, color=color_power, label='Total Power Output', linewidth=2)
    if real_power_array.any():
        ax1.plot(t_15min, real_power_array, color=color_power, linestyle = ':', label='2019 Power Output', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color_power)
    ax1.set_xticks(np.arange(0, 25, 1))
    ax1.grid(True, alpha=0.6)

    # Right Axis: Irradiance (kW/m2)
    ax2 = ax1.twinx()
    color_irr = 'tab:orange'
    ax2.set_ylabel('Irradiance (kW/m^2)', color=color_irr, fontweight='bold')
    ax2.plot(t, irradiance_array / 1000, color=color_irr, label='Irradiance', linestyle='--', linewidth=2)
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

def plot_energy(N, energy, actual_energy, title = 'Daily Energy Production vs. Day of the Year'):

    plt.figure(figsize=(10, 6))
    plt.plot(N, energy, color='c', linewidth=2, label='Daily Energy Production')
    plt.plot(N, actual_energy, color='b', linewidth=2, label='2019 Energy Production')
    plt.xlabel('Day', fontweight='bold')
    plt.ylabel('Energy Production (MWh)', fontweight='bold')
    plt.title(title)
    plt.xticks(np.arange(0, 366, 30))

    plt.grid(True, alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plt.show()

def plot_power_delivery(t, power_array, real_power_array, day_name):

    t_15min = np.linspace(0, 24, 96)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left Axis: Total System Power (kW
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Total System Power Delivery (kW)', fontweight='bold')
    ax1.plot(t, power_array * 960 / 1000, color='m', label='Total Power Output', linewidth=2)
    if real_power_array.any():
        ax1.plot(t_15min, real_power_array, linestyle = ':', label='Actual PEC Power Output', linewidth=2)
    ax1.tick_params(axis='y')
    ax1.set_xticks(np.arange(0, 25, 1))
    ax1.grid(True, alpha=0.6)
    plt.title(f'Total System Power Delivery vs. Time ({day_name})')
    plt.grid(True, alpha=0.6)
    plt.legend()
    plt.tight_layout()

    plt.show()


    ## also need to compare to actual 2019 data!

## Case 3: Optimized Vertical Tracking Angle ##

def optimized_beta (N, hour,gamma, L):

    delta = solar_declination_angle(N)
    sol_time = solar_time(N, hour)
    omega = solar_hour_angle(sol_time)
    theta_z = zenith_angle(L, delta, omega)

    if theta_z >= math.pi/2:
        return 0.0 # skips optimizaton math is sun is below horizon (night)

    alpha = 90 - math.degrees(theta_z)
    gamma_s = solar_azimuth_angle(delta, omega, alpha)

    # Define the objecttive function to find beta that minimizes angle_of_incidence 
    def objective(beta):
        return angle_of_incidence(alpha, beta, gamma, gamma_s)
    
    res = minimize_scalar(objective, bounds=(0, 90), method='bounded')

    return res.x

def simulate_case_3(N, t_array, gamma):
    
    L = 30.26 # deg Latitude of Austin

    power_day = np.zeros(np.size(t_array))
    irradiance_day = np.zeros(np.size(t_array))
    beta_day = np.zeros(np.size(t_array))

    for i, hour in enumerate(t_array):
        opt_beta = optimized_beta(N, hour, gamma, L)
        beta_day[i] = opt_beta

        res = simulate(N, np.array([hour]), opt_beta, gamma)

        power_day[i] = res[0][0]
        irradiance_day[i] = res[1][0]
    return power_day, irradiance_day, beta_day


def get_power_outputs_2019(file_path, n_array):
    """
    Extracts and cleans solar power data for an array of day numbers (N).

    Parameters:
    file_path (str): Path to 'PEC 15 minute data for 2019.csv'
    n_array (list or np.array): Array of day numbers (e.g., [55, 172, 355])

    Returns:
    dict: { N: np.array([96 values in kW]) }
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    df['Date & Time'] = pd.to_datetime(df['Date & Time'])
    df = df.sort_values(by='Date & Time')

    # Ensure n_array is iterable if a single integer is passed
    if isinstance(n_array, (int, np.integer)):
        n_array = [n_array]

    results = {}
    start_date = datetime(2019, 1, 1)  # 2019 was a non-leap year

    for n in n_array:
        # Convert the day number N to the actual calendar date
        target_date = (start_date + timedelta(days=int(n) - 1)).date()

        # Filter for the specific day
        day_df = df[df['Date & Time'].dt.date == target_date]

        if not day_df.empty:
            # Extract 'Solar [kW]' and remove nighttime parasitic noise (values < 0)
            raw_power = day_df['Solar [kW]'].values
            cleaned_power = np.maximum(raw_power, 0)

            # Store in the dictionary indexed by N
            results[n] = cleaned_power
        else:
            print(f"Warning: No data found for N={n} (Date: {target_date})")

    return results


def get_power_outputs_2026(file_path):
    """
    Extracts and cleans solar power data from the 2026 PEC CSV.

    Returns:
    np.array: Cleaned solar power values in kW.
    """
    # 1. Load the CSV
    df = pd.read_csv(file_path)

    # 2. Convert to datetime and sort (the 2026 file is currently in reverse order)
    df['Date & Time'] = pd.to_datetime(df['Date & Time'])
    df = df.sort_values(by='Date & Time')

    # 3. Extract the 'Solar [kW]' column
    raw_power = df['Solar [kW]'].values

    # 4. Clean nighttime parasitic noise (clamping values < 0 to 0)
    cleaned_power = np.maximum(raw_power, 0)

    return cleaned_power


def get_annual_daily_energy_array(file_path):
    """
    Calculates total energy (kWh) for every day of the year from the CSV.

    Returns:
    np.array: A 365-element array where index 0 is Jan 1st (N=1).
    """
    # 1. Load the dataset
    df = pd.read_csv(file_path)
    df['Date & Time'] = pd.to_datetime(df['Date & Time'])

    # 2. Convert Power (kW) to Energy (kWh) per interval
    # Since intervals are 15 minutes, Energy = Power * (15/60)
    # We also clamp values at 0 to remove nighttime parasitic draw
    df['kWh_interval'] = np.maximum(df['Solar [kW]'], 0) * 0.25

    # 3. Sum energy by calendar date
    df['date_only'] = df['Date & Time'].dt.date
    daily_sums = df.groupby('date_only')['kWh_interval'].sum()

    # 4. Map sums to a 365-day array (to handle any potential missing days)
    # 2019 was a non-leap year.
    start_date = datetime(2019, 1, 1).date()
    annual_energy_array = np.zeros(365)

    for date, energy in daily_sums.items():
        # Calculate N (Day of Year)
        n = (date - start_date).days + 1
        if 1 <= n <= 365:
            # Store in 0-indexed array (N=1 at index 0)
            annual_energy_array[n - 1] = energy

    return annual_energy_array

if __name__ == '__main__':
    main()