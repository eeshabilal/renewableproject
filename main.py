import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize_scalar
from datetime import datetime, timedelta
import csv
from collections import defaultdict


def main():
    # <editor-fold desc="Control Panel">
    """"""""""""""""""""""""""" Control Panel """""""""""""""""""""""""""""
    # For when you're testing stuff and everyone else's work is slowing you down
    show_case_1 = 0
    show_case_2 = 0
    show_case_3 = 0
    show_case_4 = 0
    show_case_5 = 0
    show_case_6 = 1
    show_annual_calc = 1

    # Relevant days and times
    # Feb 5, N = 36
    # Feb 24, N = 55
    # Jun 21, N = 172 [49248:49536]
    # Dec 21, N = 355 [101952:102240]
    t_15min = np.linspace(0, 24, 96)
    t_5min = np.linspace(0, 24, 288)  # 24 hours*12 increments per hour = 288 increments
    days_in_year = np.arange(1, 366)

    ## Comment or Uncomment Depending on what we're plotting ##

    # For plots vs time of day on individual days
    # N = np.array([355]) # Edit for plots of certain days
    # day_name = 'Dec 21'

    # # For plots vs day of the year at individual times
    N = np.arange(1, 366)  # Day number where Jan 1st is 1
    day_name = ''

    beta = 22  # Panel angle for case 1 and 2
    gamma = 46  # Panel azimuthal angle

    # Case 6 constants
    original_Panels = 960
    pack_capacity = 210  # kWh per tesla power pack
    battery_efficiency = 1.0  # assume ideal unless told otherwise
    battery_cost = 115  # $/kWh

    DT = 5 / 60  # 5-minute timestep in hours

    battery_packs_list = [0, 6, 12]
    austin_energy_prices = np.linspace(0.06, 0.18, 7)
    feb_5_idx = 36

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    # </editor-fold>

    # <editor-fold desc="Case 1">
    days_needed = np.unique(np.append(N, [172, 355])).tolist()  # Adding june 21 and dec 21 for case 2 and 3 plots

    cleaned_2019_data = get_power_outputs_2019('PEC 15 minute data for 2019.csv', days_needed)
    annual_actual_energy = get_annual_daily_energy_array('PEC 15 minute data for 2019.csv')
    cleaned_feb_data, feb_load_data  = get_power_outputs_2026('pec 15 minute data for 2.5.2026.csv')
    monthly_max, monthly_total = max_monthly_energy_2019(annual_actual_energy)
    total_system_energy = np.zeros(len(N))  # kWh

    if show_case_1:
        if not day_name:
            t = np.zeros(len(N))
            for i, day in enumerate(N):
                t[i] = local_time(day, 12)
            theta_i_noon = np.zeros(len(N))  # deg
        else:
            t = t_5min
            total_system_power = np.zeros((np.size(N), np.size(t)))  # W
            irradiance = np.zeros((np.size(N), np.size(t)))  # W/m^2
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

        # Plots vs time of day

        if day_name:
            if day_name == 'Dec 21':
                flat_panel_irradiance = simulate(day, t, beta=0, gamma=gamma)[1]
                plot_solar_data(t, total_system_power[0], cleaned_2019_data[N[0]], flat_panel_irradiance, day_name)
            else:
                plot_solar_data(t, total_system_power[0], cleaned_2019_data[N[0]], irradiance[0], day_name)
            plot_bd_ratio(t, bd_ratio[0], day_name)

        if day_name == 'Feb 5':
            plot_power_delivery(t, total_system_power[0], cleaned_feb_data, day_name)

        # Plots vs day of the year
        if not day_name:
            plot_theta_i(N, theta_i_noon)
            plot_energy(N, total_system_energy / 1000, annual_actual_energy / 1000)  # Converts kWh to MWh for plotting
    # </editor-fold>

    # <editor-fold desc="Annual MWh Calculation">
    # -----  Annual MWh Calculation -------
    if show_annual_calc:
        energy_case1_annual = 0
        energy_case3_annual = 0
        energy_cloudy_annual = 0
        daily_mwh_case1 = []
        daily_mwh_case3 = []
        daily_mwh_cloudy = []
        for day in days_in_year:
            # Case 1 Daily Energy
            p1 = simulate(day, t_5min, beta, gamma)[0]
            e1 = np.trapezoid(p1 * 960, t_5min) / 1e6  # MWh for 960 panels
            energy_case1_annual += e1
            daily_mwh_case1.append(e1)

            # Case 3 Daily Energy
            p3, irr3, beta3 = simulate_case_3(day, t_5min, gamma)
            e3 = np.trapezoid(p3 * 960, t_5min) / 1e6  # MWh for 960 panels
            energy_case3_annual += e3
            daily_mwh_case3.append(e3)

            # Cloudy Daily Energy
            p_cloudy = \
            simulate(day, t_5min, beta, gamma, annual_energy_array=annual_actual_energy, monthly_max=monthly_max)[0]
            e_cloudy = np.trapezoid(p_cloudy * 960, t_5min) / 1e6
            energy_cloudy_annual += e_cloudy
            daily_mwh_cloudy.append(e_cloudy)
        # Output Table of Annual Energy Production for Case 1 and Case 3 compared to 2019

        energy_actual_annual = np.sum(annual_actual_energy) / 1000  # MWh of 2019 actual energy
        print("\n" + "=" * 45)
        print(f"{'Case Study':<25} | {'Annual Energy (MWh)':<15}")
        print("-" * 45)
        print(f"{'Case 1 (Base; Fixed Tilt)':<25} | {energy_case1_annual:>15.4f}")
        print(f"{'Case 3 (Tracking)':<25} | {energy_case3_annual:>15.4f}")
        print(f"{'2019 Actual Data':<25} | {energy_actual_annual:>15.4f}")
        print("=" * 45)
    # </editor-fold>

    # <editor-fold desc="Case 2">
    # ---- For Case 2: Effect of Panel Temperature ---- #

    # Plot irradiance and total system power - December 21 and June 21
    if show_case_2:
        temp_sets = {
            "Dec 21 - Case 2": {"day": 355, "temps": [0, 25, 45]},
            "Jun 21 - Case 2": {"day": 172, "temps": [25, 45, 85]}
        }

        for label, data in temp_sets.items():
            fig, axis1 = plt.subplots(figsize=(10, 6))
            axis2 = axis1.twinx()

            day = data["day"]
            irr_plot = None  # To ensure irradiance is only plotted once per day since it doesn't change with temperature

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
                e2 = np.trapezoid(power_temp * 960, t_5min) / 1e6  # MWh for 960 panels
                daily_mwh_case2.append(e2)
            plt.plot(days_in_year, daily_mwh_case2, label=f'Temp = {temp}°C')
        plt.plot(days_in_year, annual_actual_energy / 1000, label='2019 Energy')
        plt.xlabel('Day of the Year', fontweight='bold')
        plt.ylabel('Daily Energy Production (MWh)', fontweight='bold')
        plt.title('Daily Energy Production vs. Day of the Year for Case 2', fontweight='bold')
        plt.legend()
        plt.grid(True)
        plt.show()
    # </editor-fold>

    # <editor-fold desc="Case 3">
    # ---- For Case 3: Optimized Vertical Tracking Angle ----

    if show_case_3:
        # Irradiance and total system power delivery - December 21
        power_dec, irr_dec, beta_dec = simulate_case_3(355, t_5min, gamma)
        power_dec_no_tracking, irr_dec_no_tracking, _, _ = simulate(355, t_5min, beta, gamma)
        plot_solar_data(t_5min, power_dec, cleaned_2019_data[355], irr_dec, power_dec_no_tracking, irr_dec_no_tracking,
                        'December 21 - Case 3 Optimized Vertical Tracking')

        # Irradiance and total system power delivery - June 21
        power_jun, irr_jun, beta_jun = simulate_case_3(172, t_5min, gamma)
        power_jun_no_tracking, irr_jun_no_tracking, _, _ = simulate(172, t_5min, beta, gamma)
        plot_solar_data(t_5min, power_jun, cleaned_2019_data[172], irr_jun, power_jun_no_tracking, irr_jun_no_tracking,
                        'June 21 - Case 3 Optimized Vertical Tracking')

        # Plot Tilt Angle vs time of day - June 21
        plt.figure(figsize=(10, 6))
        plt.plot(t_5min, beta_jun, color='purple', linewidth=2, label='Optimized Beta Angle')
        plt.title('Case 3 Optimized Panel Angle vs Time of Day (June 21)')
        plt.ylabel("Tilt Angle (degrees)")
        plt.xlabel("Time (hours)")
        plt.grid(True, alpha=0.6)
        plt.show()

        # Plot total daily energy production vs day of the year
        plot_energy(days_in_year, daily_mwh_case3, annual_actual_energy / 1000,daily_mwh_case1,
                    title='Daily Energy Production vs. Day of the Year for Case 3')
    # </editor-fold>

    # <editor-fold desc="Case 4">
    # ---- For Case 4: cloudy data ----
    if show_case_4:
        # Plot cloudy daily power output over the year 2019
        plt.figure()
        plt.plot(days_in_year, daily_mwh_cloudy, label='Cloudy')
        plt.plot(days_in_year,
                 [np.trapezoid(simulate(d, t_5min, beta, gamma)[0] * 960, t_5min) / 1e6 for d in days_in_year],
                 label='Clear')
        plt.plot(days_in_year, annual_actual_energy / 1000, label='2019 Actual')
        plt.xlabel('Day of Year')
        plt.ylabel('Daily Energy (MWh)')
        plt.title('Case 4 Cloudy vs Clear Sky Power Output (2019)')
        plt.legend()
        plt.show()

        # Plot cloudy power output for 1 day in June and one day in December compared to clear sky and 2019 actual data for those days
        cloudy_sets = {
            "Dec 21 - Cloudy (OCI=10)": {"day": 355},  # this day OCI must be 10
            "Jun 21 - Cloudy": {"day": 172}  # this day OCI we must calculate
        }

        for label, data in cloudy_sets.items():
            fig, axis1 = plt.subplots(figsize=(10, 6))
            axis2 = axis1.twinx()

            day = data["day"]

            # power: tilted panel (beta=22, gamma=46) with cloudy/clear sky
            power_cloudy, _, _, _ = simulate(day, t_5min, beta, gamma,
                                            annual_energy_array=annual_actual_energy,
                                            monthly_max=monthly_max)
            power_clear, _, _, _ = simulate(day, t_5min, beta, gamma)

            # irradiance: horizontal panel (beta=0) — this is what a pyranometer measures
            _, irr_cloudy, _, _ = simulate(day, t_5min, beta=0, gamma=gamma,
                                            annual_energy_array=annual_actual_energy,
                                            monthly_max=monthly_max)
            _, irr_clear, _, _ = simulate(day, t_5min, beta=0, gamma=gamma)

            # print OCI for reference
            oci_val = oci(monthly_max, day, annual_actual_energy)
            print(f"{label}: OCI = {oci_val:.2f}")

            axis1.plot(t_5min, power_cloudy * 960 / 1000, label='Cloudy Power', color='steelblue')
            axis1.plot(t_5min, power_clear * 960 / 1000, label='Clear Sky Power', color='blue', linestyle='--')
            axis1.plot(t_15min, cleaned_2019_data[day], linestyle=':', label='2019 Actual Power')
            axis2.plot(t_5min, irr_cloudy / 1000, color='orange', linestyle='--', label='Cloudy Irradiance')
            axis2.plot(t_5min, irr_clear / 1000, color='gold', linestyle=':', label='Clear Irradiance')

            axis1.set_xlabel('Time of Day (hours)', fontweight='bold')
            axis1.set_ylabel('Total System Power Delivery (kW)', color='blue', fontweight='bold')
            axis2.set_ylabel('Irradiance (kW/m^2)', color='orange', fontweight='bold')
            plt.title(f"Case 4 Irradiance and System Power: {label}", fontweight='bold')
            axis1.grid(True, alpha=0.6)

            lines1, labels1 = axis1.get_legend_handles_labels()
            lines2, labels2 = axis2.get_legend_handles_labels()
            axis1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            plt.show()
    # </editor-fold>

    # <editor-fold desc="Case 5">
    # ---- Case 5: Heat Transfer Model ---- #
    if show_case_5:
        temps_array, max_array, min_array = generate_yearly_5min_ambient_temps('austin_weather.csv')
        jun21_clear_simulation = simulate(172, t_5min, 22, 46)
        jun21_cloudy_simulation = simulate(172, t_5min, 22, 46, OCI_manual=10)
        jun21_real_simulation = simulate(172, t_5min, 22, 46, annual_energy_array=annual_actual_energy,
                                         monthly_max=monthly_max)
        dec21_cloudy_simulation = simulate(355, t_5min, 22, 46, OCI_manual=10)

        # Those big indices are the 5 minute increments that represent Jun 21st if a whole year was split into 5 minute segments
        jun21_clear_temps, jun21_clear_power, _ = simulate_case_5(jun21_clear_simulation[1], temps_array[49248:49536],
                                                                  temps_array[49248])
        jun21_cloudy_temps, jun21_cloudy_power, _ = simulate_case_5(jun21_cloudy_simulation[1],
                                                                    temps_array[49248:49536], temps_array[49248])
        jun21_real_temps, jun21_real_power, _ = simulate_case_5(jun21_real_simulation[1], temps_array[49248:49536],
                                                                temps_array[49248])
        dec21_cloudy_temps, dec21_cloudy_power, _ = simulate_case_5(dec21_cloudy_simulation[1],
                                                                    temps_array[101952:102240], temps_array[101952])

        # Initialize your tracking array and the starting panel temperature
        case_5_daily_mwh = []

        # Start the panel at the ambient temperature at midnight on Jan 1
        T_cell_initial = temps_array[0]

        # Loop through all 365 days of the year (Day 1 through Day 365)
        i = 0
        case1_panel_temp_total_system_energy = np.zeros(365)
        case1_total_system_energy = np.zeros(365)
        case_4_daily_mwh = []
        case5_total_system_energy = np.zeros(365)
        pec_actual_kwh = get_annual_daily_energy_array('PEC 15 minute data for 2019.csv')
        for N in range(1, 366):


            daily_oci = oci(monthly_max, N, annual_actual_energy)
            T_a_day = temps_array[(N - 1) * 288: N * 288]
            I_array = simulate(N, t_5min, beta, gamma, OCI_manual=daily_oci)[1]
            T_cell_array,_,_ = simulate_case_5(I_array, T_a_day, T_cell_initial)

            case5_energy = simulate(N, t_5min, beta, gamma, T_cell=T_cell_array, OCI_manual=daily_oci)[0]
            case5_total_system_energy[i] = np.trapezoid(case5_energy, t_5min)
            case1_panel_temp_energy = simulate(N, t_5min, beta, gamma, T_cell=T_cell_array)[0]

            case1_panel_temp_total_system_energy[i] = np.trapezoid(case1_panel_temp_energy, t_5min)
            case1_energy =  simulate(N, t_5min, beta, gamma)[0]
            case1_total_system_energy[i] = np.trapezoid(case1_energy, t_5min)

            # ripped from case 4 so that I don't have to turn show_case_4 on
            power_cloudy = \
            simulate(N, t_5min, beta, gamma, annual_energy_array=annual_actual_energy, monthly_max=monthly_max)[0]
            energy_cloudy = np.trapezoid(power_cloudy * 960, t_5min) / 1e6

            # p_cloudy = \
            #     simulate(day, t_5min, beta, gamma, annual_energy_array=annual_actual_energy, monthly_max=monthly_max)[0]
            # e_cloudy = np.trapezoid(p_cloudy * 960, t_5min) / 1e6
            # daily_mwh_cloudy.append(e_cloudy)

            case_4_daily_mwh.append(energy_cloudy)

            i += 1

        # Hourly Temperatures throughout the year with mins and maxes
        plot_yearly_ambient_temps_with_extremes(temps_array, max_array, min_array)

        # Jun 21 Panel Temp vs Time of Day sunny and cloudy plots
        plt.plot(t_5min, jun21_clear_temps, label='Clear')
        plt.plot(t_5min, jun21_cloudy_temps, label='Cloudy')
        plt.plot(t_5min, temps_array[49248:49536], label='Ambient')
        plt.xlabel('Time of Day (hours)')
        plt.xticks(np.arange(0, 25, 4))
        plt.ylabel('Temperature (C)')
        plt.title('Panel Temperature vs. Time of Day (June 21) - Case 5')
        plt.grid(True, alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Jun 21 Irradiance and total system power delivery vs Time of Day sunny, cloudy, and cloud model plots
        fig, ax1 = plt.subplots()

        ax1.plot(t_5min, jun21_clear_power, label='Clear Power Output', color='b')
        ax1.plot(t_5min, jun21_real_power, label='Cloud Model Power Output', color='g')
        ax1.set_xlabel('Time of Day (hours)')
        ax1.set_xticks(np.arange(0, 25, 4))
        ax1.set_ylabel('Power Output (kW)')

        ax2 = ax1.twinx()
        ax2.plot(t_5min, jun21_clear_simulation[1] / 1e3, label='Clear Irradiance', linestyle='--', color='orange')
        ax2.plot(t_5min, jun21_real_simulation[1] / 1e3, label='Cloud Model Irradiance', linestyle='--', color='r')
        ax2.set_ylabel('Irradiance (kW/m^2)')

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2)

        plt.title('Irradiance and Total System Power vs. Time of Day (June 21) - Case 5')
        plt.tight_layout()
        plt.show()

        # Dec 21 Irradiance and total system power delivery vs Time of day cloudy
        fig, ax1 = plt.subplots()

        ax1.plot(t_5min, dec21_cloudy_power, label='Power Output')
        ax1.set_xlabel('Time of Day (hours)')
        ax1.set_xticks(np.arange(0, 25, 4))
        ax1.set_ylabel('Power Output (kW)')

        ax2 = ax1.twinx()
        ax2.plot(t_5min, dec21_cloudy_simulation[1] / 1e3, label='Irradiance', linestyle='--', color='orange')
        ax2.set_ylabel('Irradiance (kW/m^2)')

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2)

        plt.title('Cloudy Irradiance and Total System Power vs. Time of Day (Dec 21) - Case 5')
        plt.tight_layout()
        plt.show()

        # Total daily energy production (MWh) vs. Day of year starting from Jan 1
        case_1_daily_mwh = case1_total_system_energy / 1e3
        case1_panel_temp_total_system_energy = case1_panel_temp_total_system_energy / 1e3
        case_4_daily_mwh = np.array(case_4_daily_mwh)
        case_5_daily_mwh = case5_total_system_energy / 1e3
        pec_actual_mwh = np.array(pec_actual_kwh) / 1e3

        plt.plot(days_in_year, case_1_daily_mwh, label='Case 1')
        plt.plot(days_in_year, case1_panel_temp_total_system_energy, label='Case 1 Plus Panel Temp Model')
        plt.plot(days_in_year, case_4_daily_mwh, label='Case 4')
        plt.plot(days_in_year, case_5_daily_mwh, label='Case 5')
        plt.plot(days_in_year, pec_actual_mwh, label='PEC 2019')
        plt.xlabel('Day of the Year')
        plt.xticks(np.arange(0, 366, 30))
        plt.ylabel('Energy Production (MWh)')
        plt.title('Total Daily Energy Production vs. Day of the Year and Comparisons - Case 5')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # </editor-fold>

    # <editor-fold desc="Case 6">
    # -----Case 6: Battery Storage and Economic Analysis-----#

    if show_case_6:
        # Performance Plots (Feb 5)
        # Clear Day (OCI=0) for 1x, 4x, 6x
        for scale in [1, 4, 6]:
            plot_case_6_performance(36, 0, scale, 6, cleaned_feb_data, feb_load_data)

        # Cloudy Day (OCI=5) for 4x, 6x
        for scale in [4, 6]:
            plot_case_6_performance(36, 5, scale, 6, cleaned_feb_data, feb_load_data)

        # Economic Sensitivity Plots
        plot_case_6_economics(4, annual_actual_energy, monthly_max)
        plot_case_6_economics(6, annual_actual_energy, monthly_max)
    # </editor-fold>


def simulate(N, t, beta, gamma, T_cell=25, annual_energy_array=None, monthly_max=None, OCI_manual=None):
    # simulate function indexing guide
    # simulate[0] = Power output for 1 panel in W/m^2
    # simulate[1] = Irradiance in W/m^2
    # simulate[2] = Ratio of beam irradiance to diffuse irradiance
    # simulate[3] = Angle of incidence in rads
    # simulate[4] = Total daily energy production for 1 panel in kWh

    # constants
    L = 30.26  # deg Latitude of Austin
    altitude = .149  # km Altitude of Austin
    panel_eff = .157  # efficiency at 25C cell temperature
    inverter_eff = .965
    power_temp_coeff = -0.0045  # Power temperature coefficient from 25C
    derating_factor = .93 * .9 * .94 * .89
    A = 1.64 * .99  # m^2
    I_0 = extraterrestrial_radiation(N)  # W/m^2
    delta = solar_declination_angle(N)  # deg
    array_size = np.size(t)

    solar_times = np.zeros(array_size)  # h
    omega = np.zeros(array_size)  # deg
    theta_z = np.zeros(array_size)  # rad
    alpha = np.zeros(array_size)  # deg
    gamma_s = np.zeros(array_size)  # deg
    theta_i = np.zeros(array_size)  # rad
    tau_b = np.zeros(array_size)
    tau_d = np.zeros(array_size)
    I_cb = np.zeros(array_size)  # W/m^2
    I_cd = np.zeros(array_size)  # W/m^2
    bd_ratio = np.zeros(array_size)
    I = np.zeros(array_size)  # W/m^2
    Wdot_elec = np.zeros(array_size)  # W/m^2

    i = 0
    for hour in t:
        solar_times[i] = solar_time(N, hour)
        omega[i] = solar_hour_angle(solar_times[i])
        theta_z[i] = zenith_angle(L, delta, omega[i])

        if theta_z[i] < math.pi / 2:
            alpha[i] = 90 - math.degrees(theta_z[i])
            gamma_s[i] = solar_azimuth_angle(delta, omega[i], alpha[i])
            theta_i[i] = angle_of_incidence(alpha[i], beta, gamma, gamma_s[i])

            if theta_i[i] < math.pi / 2:
                if annual_energy_array is None and monthly_max is None and OCI_manual is None:
                    tau_b[i] = beam_transmissivity(N, theta_z[i], altitude)
                    tau_d[i] = diffuse_transmittivity(tau_b[i])
                    I_cd[i] = diffuse_radiation(I_0, theta_z[i], tau_d[i], beta)
                    I_cb[i] = beam_radiation(I_0, tau_b[i], theta_i[i])
                    I[i] = I_cd[i] + I_cb[i]
                else:
                    if OCI_manual is None:
                        OCI = oci(monthly_max, int(N), annual_energy_array)
                    else:
                        OCI = OCI_manual
                    tau_b[i] = beam_transmissivity_cloudy(N, theta_z[i], altitude, OCI)
                    tau_d[i] = diffuse_transmittivity_cloudy(N, theta_z[i], altitude, OCI)
                    I_cd[i] = diffuse_radiation_cloudy(I_0, theta_z[i], tau_d[i], beta)
                    I_cb[i] = beam_radiation_cloudy(I_0, tau_b[i], theta_i[i])
                    I[i] = I_cd[i] + I_cb[i]

                P_25C = I[i] * panel_eff * A * inverter_eff  # Power at 25C cell temperature
                if type(T_cell) == int:
                    Wdot_elec[i] = P_25C * (1 + power_temp_coeff * (T_cell - 25))  # Adjusted for cell temperature
                else:
                    Wdot_elec[i] = P_25C * (1 + power_temp_coeff * (T_cell[i] - 25))  # Adjusted for cell temperature
                bd_ratio[i] = I_cb[i] / I_cd[i]
                if bd_ratio[i] > 10:  # Correct for huge spike
                    bd_ratio[i] = bd_ratio[i - 1]

            else:
                Wdot_elec[i] = 0
                bd_ratio[i] = 0

        else:
            Wdot_elec[i] = 0
            bd_ratio[i] = 0

        i += 1

    return Wdot_elec * derating_factor, I, bd_ratio, theta_i


def solar_time(N, standard_time):
    # N is day number, standard_time is in hours
    # Returns the solar time in hours

    tau = math.radians(360 * N / 365)
    long_std = 90  # deg
    long_loc = 97.74  # deg
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

    tau = math.radians(360 * N / 365)
    long_std = 90  # deg
    long_loc = 97.74  # deg
    ET = (
            - 7.3412 * math.sin(tau) + .4944 * math.cos(tau)
            - 9.3795 * math.sin(2 * tau) - 3.2568 * math.cos(2 * tau)
            - .3179 * math.sin(3 * tau) - .0774 * math.cos(3 * tau)
            - .1739 * math.sin(4 * tau) - .1283 * math.cos(4 * tau)
    )

    return solar_time - (4 * (long_std - long_loc) + ET) / 60  # In hours


def solar_hour_angle(solar_time):  # omega
    return solar_time * 15 - 180  # deg


def solar_declination_angle(N):  # delta
    theta = math.radians(360 * ((284 + N) / 365))
    return 23.45 * math.sin(theta)  # deg


def zenith_angle(L, delta, omega):  # theta_z
    L = math.radians(L)
    delta = math.radians(delta)
    omega = math.radians(omega)
    return math.acos(math.sin(L) * math.sin(delta) + math.cos(L) * math.cos(delta) * math.cos(omega))  # radians


def solar_azimuth_angle(delta, omega, alpha):  # gamma_s
    delta = math.radians(delta)
    omega = math.radians(omega)
    alpha = math.radians(alpha)
    return math.degrees(math.asin(math.cos(delta) * math.sin(omega) / math.cos(alpha)))  # degrees


def angle_of_incidence(alpha, beta, gamma, gamma_s):  # theta_i
    alpha = math.radians(alpha)
    beta = math.radians(beta)
    gamma = math.radians(gamma)
    gamma_s = math.radians(gamma_s)
    return math.acos(
        math.sin(alpha) * math.cos(beta) + math.cos(alpha) * math.sin(beta) * math.cos(gamma - gamma_s))  # rad


def beam_transmissivity(N, theta_z, A):  # tau_b
    # theta_z = math.radians(theta_z) # Uncomment this when testing hand calcs with degrees.
    if 81 <= N <= 264:  # Summer Range
        r0, r1, rk = 0.97, 0.99, 1.02
    else:  # Winter range
        r0, r1, rk = 1.03, 1.01, 1.00

    a0_star = 0.4237 - 0.008216 * (6 - A) ** 2
    a1_star = 0.5055 + 0.00595 * (6.5 - A) ** 2
    k_star = 0.2711 + 0.01858 * (2.5 - A) ** 2

    a0 = r0 * a0_star
    a1 = r1 * a1_star
    k = rk * k_star

    return a0 + a1 * math.exp(-k / math.cos(theta_z))


def diffuse_transmittivity(tau_b):  # tau_d
    return .271 - .294 * tau_b


def extraterrestrial_radiation(N):  # I_0
    solar_constant = 1368  # W/m^2
    return solar_constant * (1 + .034 * math.cos(2 * math.pi * (N - 3) / 365))


def diffuse_radiation(I_0, theta_z, tau_d, beta):
    # theta_z = math.radians(theta_z) # uncomment for testing hand calcs w degrees
    beta = math.radians(beta)
    return I_0 * math.cos(theta_z) * tau_d * ((1 + math.cos(beta)) / 2)


def beam_radiation(I_0, tau_b, theta_i):
    # theta_z = math.radians(theta_z) # uncomment for testing hand calcs w degrees
    return I_0 * tau_b * math.cos(theta_i)


def plot_solar_data(t, power_array, real_power_array, irradiance_array, comparison_power=None, comparison_irr=None, day_name=None):

    t_15min = np.linspace(0, 24, 96)
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Left Axis: Total System Power (kW)
    color_power = 'tab:blue'
    ax1.set_xlabel('Time (hours)',fontsize=14, fontweight='bold')
    ax1.set_ylabel('Total System Power Delivery (kW)', color=color_power, fontsize=14, fontweight='bold')

    # Case 3 (tracking)
    ax1.plot(t, power_array * 960 / 1000,
             color=color_power, label='Tracking Power Output', linewidth=2)

    if comparison_power is not None:
        ax1.plot(t, comparison_power * 960 / 1000,
                 color='tab:green', linestyle='--',
                 label='No Tracking Power Output', linewidth=2)

    # Real data
    if real_power_array is not None and real_power_array.any():
        ax1.plot(t_15min, real_power_array,
                 color=color_power, linestyle=':',
                 label='2019 Power Output', linewidth=2)

    ax1.tick_params(axis='y', labelcolor=color_power)
    ax1.set_xticks(np.arange(0, 25, 1))
    ax1.grid(True, alpha=0.6)

    # Right Axis: Irradiance (kW/m2)
    ax2 = ax1.twinx()
    color_irr = 'tab:orange'
    ax2.set_ylabel('Irradiance (kW/m^2)', color=color_irr, fontsize=14, fontweight='bold')

    # Case 3 irradiance
    ax2.plot(t, irradiance_array / 1000,
             color=color_irr, linestyle='--',
             label='Tracking Irradiance', linewidth=2)

    if comparison_irr is not None:
        ax2.plot(t, comparison_irr / 1000,
                 color='tab:red', linestyle=':',
                 label='No Tracking Irradiance', linewidth=2)

    ax2.tick_params(axis='y', labelcolor=color_irr)

    # Title and legend
    plt.title(f'Power & Irradiance vs. Time ({day_name})', fontsize=16, fontweight='bold')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=12, loc='upper left')

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


def plot_energy(N, energy, actual_energy, no_tracking_energy = None, title='Daily Energy Production vs. Day of the Year'):
    plt.figure(figsize=(10, 6))
    plt.plot(N, energy, color='c', linewidth=2, label='Daily Energy Production')
    plt.plot(N, actual_energy, color='b', linewidth=2, label='2019 Energy Production')
    if no_tracking_energy is not None:
        plt.plot(N, no_tracking_energy, color='orange', linewidth=2, label='No Tracking Energy Production')
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
        ax1.plot(t_15min, real_power_array, linestyle=':', label='Actual PEC Power Output', linewidth=2)
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

def optimized_beta(N, hour, gamma, L):
    delta = solar_declination_angle(N)
    sol_time = solar_time(N, hour)
    omega = solar_hour_angle(sol_time)
    theta_z = zenith_angle(L, delta, omega)

    if theta_z >= math.pi / 2:
        return 0.0  # skips optimizaton math is sun is below horizon (night)

    alpha = 90 - math.degrees(theta_z)
    gamma_s = solar_azimuth_angle(delta, omega, alpha)

    # Define the objecttive function to find beta that minimizes angle_of_incidence
    def objective(beta):
        return angle_of_incidence(alpha, beta, gamma, gamma_s)

    res = minimize_scalar(objective, bounds=(-90, 90), method='bounded')

    return res.x


def simulate_case_3(N, t_array, gamma):
    L = 30.26  # deg Latitude of Austin

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
    Extracts and cleans solar power data AND load data from the 2026 PEC CSV.

    Returns:
    tuple: (Cleaned solar power values in kW, Load values in kW)
    """
    df = pd.read_csv(file_path)
    df['Date & Time'] = pd.to_datetime(df['Date & Time'])
    df = df.sort_values(by='Date & Time')

    # Extract Solar (last column)
    raw_solar = df['Solar [kW]'].values
    cleaned_power = np.maximum(raw_solar, 0)

    # Extract Load (second column, index 1)
    # Using .iloc[:, 1] is safer if the header is just "[kW]"
    actual_load = df.iloc[:, 1].values

    return cleaned_power, actual_load


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


# Case 4: Effect of clouds on power output

def max_monthly_energy_2019(annual_energy_array):
    months_2019 = {
        "Jan": range(1, 32),
        "Feb": range(32, 60),
        "Mar": range(60, 91),
        "Apr": range(91, 121),
        "May": range(121, 152),
        "Jun": range(152, 182),
        "Jul": range(182, 213),
        "Aug": range(213, 244),
        "Sep": range(244, 274),
        "Oct": range(274, 305),
        "Nov": range(305, 335),
        "Dec": range(335, 366),
    }

    monthly_max = {}
    monthly_total = {}

    for month, day_range in months_2019.items():
        # slice the array for that month (converting N to 0-indexed)
        month_energies = annual_energy_array[day_range.start - 1: day_range.stop - 1]

        monthly_max[month] = np.max(month_energies)
        monthly_total[month] = np.sum(month_energies)

    return monthly_max, monthly_total


def oci(monthly_max, N, annual_energy_array):
    months_2019 = {
        "Jan": range(1, 32),
        "Feb": range(32, 60),
        "Mar": range(60, 91),
        "Apr": range(91, 121),
        "May": range(121, 152),
        "Jun": range(152, 182),
        "Jul": range(182, 213),
        "Aug": range(213, 244),
        "Sep": range(244, 274),
        "Oct": range(274, 305),
        "Nov": range(305, 335),
        "Dec": range(335, 366),
    }

    # find energy for day n
    E = annual_energy_array[N - 1]
    # find which month n belongs to
    month_of_n = next(month for month, day_range in months_2019.items() if N in day_range)
    # calc OCI
    max_E = monthly_max[month_of_n]
    if E < 0.05 * max_E:
        return 10.0
    OCI = 10 - 10 * ((E - 0.05 * max_E) / (0.95 * max_E))

    return OCI


# defining cloudy variables with OCI
def beam_transmissivity_cloudy(N, theta_z, A, OCI):
    tau_b = beam_transmissivity(N, theta_z, A)
    tau_b_cloudy = tau_b * (1 - OCI / 10)
    return tau_b_cloudy


def diffuse_transmittivity_cloudy(N, theta_z, A, OCI):
    tau_b_cloudy = beam_transmissivity_cloudy(N, theta_z, A, OCI)
    tau_d_cloudy = (1 - 0.75 * OCI / 10) * (0.271 - 0.294 * tau_b_cloudy)
    return tau_d_cloudy


def diffuse_radiation_cloudy(I_0, theta_z, tau_d_cloudy, beta):
    # theta_z = math.radians(theta_z) # uncomment for testing hand calcs w degrees
    beta = math.radians(beta)
    return I_0 * math.cos(theta_z) * tau_d_cloudy * ((1 + math.cos(beta)) / 2)


def beam_radiation_cloudy(I_0, tau_b_cloudy, theta_i):
    # theta_z = math.radians(theta_z) # uncomment for testing hand calcs w degrees
    return I_0 * tau_b_cloudy * math.cos(theta_i)


def simulate_cloudy_day(N, t, beta, gamma, annual_actual_energy, monthly_max,
                        OCI_manual=None, T_cell=25):
    # determine OCI
    if OCI_manual is not None:
        oci_val = OCI_manual
        cloudy_power, cloudy_irradiance, cloudy_bd, theta = simulate(N, t, beta, gamma, T_cell=T_cell,
                                                                     OCI_manual=oci_val)
    else:
        oci_val = oci(monthly_max, N, annual_actual_energy)
        cloudy_power, cloudy_irradiance, cloudy_bd, theta = simulate(N, t, beta, gamma, T_cell=T_cell,
                                                                     annual_energy_array=annual_actual_energy,
                                                                     monthly_max=monthly_max)

    # clear sky for comparison
    clear_power, clear_irradiance, clear_bd, theta = simulate(N, t, beta, gamma, T_cell=T_cell)

    return cloudy_power, cloudy_irradiance, clear_power, clear_irradiance, oci_val


# Case 5 - Solar panel temps
# ------Ambient Temperature Model---------
def generate_yearly_5min_ambient_temps(csv_filepath='austin_weather.csv'):
    """
    Reads historical Austin weather data, calculates the average daily min/max,
    and generates a full year of ambient temperatures in 5-minute increments
    using a sinusoidal approximation.

    Returns:
    yearly_temps_c : list of 105,120 ambient temperatures (in Celsius)
    """
    # Dictionary to group all historical highs and lows by day of the year (MM-DD)
    daily_temps = defaultdict(lambda: {'highs': [], 'lows': []})

    # Parse the CSV and aggregate historical temps for each day
    with open(csv_filepath, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            date_str = row['Date']  # Format: YYYY-MM-DD
            month_day = date_str[5:]  # Extract just the MM-DD

            # Skip leap year extra day to ensure a standard 365-day year
            if month_day == '02-29':
                continue

            try:
                # Extract the Fahrenheit high and low from the CSV
                high_f = float(row['TempHighF'])
                low_f = float(row['TempLowF'])

                daily_temps[month_day]['highs'].append(high_f)
                daily_temps[month_day]['lows'].append(low_f)
            except ValueError:
                pass  # Skip any rows with missing or malformed data

    # Calculate the historical average min and max for each day
    avg_daily_celsius = {}
    for md, temps in daily_temps.items():
        avg_high_f = sum(temps['highs']) / len(temps['highs'])
        avg_low_f = sum(temps['lows']) / len(temps['lows'])

        # Convert Fahrenheit to Celsius for the Case 5 thermodynamics model
        avg_high_c = (avg_high_f - 32) * 5.0 / 9.0
        avg_low_c = (avg_low_f - 32) * 5.0 / 9.0

        avg_daily_celsius[md] = {'t_max': avg_high_c, 't_min': avg_low_c}

    # Sort the dictionary keys to guarantee chronological order (01-01 to 12-31)
    sorted_days = sorted(avg_daily_celsius.keys())

    # Generate the temperatures
    yearly_temps_c = []
    daily_maxes = []
    daily_mins = []

    for md in sorted_days:
        t_max = avg_daily_celsius[md]['t_max']
        t_min = avg_daily_celsius[md]['t_min']

        # Save the daily extremes for plotting
        daily_maxes.append(t_max)
        daily_mins.append(t_min)

        t_mean = (t_max + t_min) / 2.0
        t_amp = (t_max - t_min) / 2.0

        for step in range(288):
            hour_of_day = step / 12.0
            t_amb = t_mean + t_amp * math.cos(math.pi * (hour_of_day - 16.0) / 12.0)
            yearly_temps_c.append(t_amb)

    # Return all three arrays
    return yearly_temps_c, daily_maxes, daily_mins


def plot_yearly_ambient_temps_with_extremes(yearly_temps_c, daily_maxes, daily_mins):
    """
    Plots the 5-minute interval ambient temperatures alongside
    the daily average maximum and minimum temperatures.
    """
    # X-axis for 5-minute intervals (105,120 points -> 365 days)
    days_5min = np.arange(len(yearly_temps_c)) / 288.0

    # X-axis for daily intervals (365 points)
    days_daily = np.arange(len(daily_maxes))

    plt.figure(figsize=(12, 6))

    # Plot the continuous 5-minute temperature band
    plt.plot(days_5min, yearly_temps_c, color='orange', linewidth=0.5,
             alpha=0.7, label='5-Min Ambient Temp')

    # Plot the daily maximums and minimums
    plt.plot(days_daily, daily_maxes, color='red', linewidth=1.5, label='Daily Avg High')
    plt.plot(days_daily, daily_mins, color='blue', linewidth=1.5, label='Daily Avg Low')

    # Set up the x-axis ticks to align with the start of each month
    month_start_days = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    plt.xticks(month_start_days, month_labels)

    plt.xlabel('Month of the Year', fontsize=12)
    plt.ylabel('Ambient Temperature (°C)', fontsize=12)
    plt.title('Modeled Ambient Temperature with Daily Extremes (Austin, TX)', fontsize=14)

    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def simulate_case_5(I_array, T_a_array, T_cell_initial=25.0):
    """
    Calculates the transient panel temperature and temperature-dependent power output
    for Case 5 using the irradiance array (I_array) and an array of ambient temperatures (T_a_array).

    Parameters:
    I_array        : List or array of solar irradiance values in W/m^2 (from your simulate function)
    T_a_array      : List or array of ambient temperatures in C (from your sinusoidal model)
    T_cell_initial : Starting temperature of the panel in C (default 25.0)

    Returns:
    T_cell_array   : Array of panel temperatures (C) at each 5-min step
    P_elec_array   : Array of electrical power output (W) at each 5-min step
    T_cell         : The final panel temperature to carry over to the next day
    """

    # Constants
    A = 1.640 * 0.99  # Panel area in m^2 [1]
    m = 20.0  # Panel weight in kg [1]
    cp = 677.0  # Cell specific heat in J/kg-K [1]
    tau = 0.96  # Cover glass transmissivity [1]
    alpha = 0.94  # Panel absorptivity [1]
    eta_ref = 0.157  # Rated module efficiency (15.7%) [1]
    temp_coeff = 0.0045  # Power temp coefficient (-0.45%/C -> 0.0045) [1]
    derating_factor = .93 * .9 * .94 * .89

    # --- NOCT Conditions to calculate U_L ---
    NOCT = 45.0  # Nominal Operating Cell Temp in C [1]
    I_NOCT = 800.0  # Standard irradiance for NOCT in W/m^2 [1, 2]
    T_air_NOCT = 20.0  # Standard ambient air temp for NOCT in C [1, 2]

    # Calculate Overall Heat Loss Coefficient (U_L)
    U_L = (I_NOCT * tau * alpha) / (NOCT - T_air_NOCT)

    dt = 300.0  # 5-minute time step in seconds

    T_cell = T_cell_initial
    T_cell_array = []
    P_elec_array = []

    # Loop through the 5-minute increments for the given day
    for i in range(len(I_array)):
        I = I_array[i]
        T_a = T_a_array[i]

        # Temperature-Dependent Efficiency
        # Efficiency decreases by 0.45% for every degree above 25C
        eta = eta_ref * (1 - temp_coeff * (T_cell - 25.0))

        # Calculate Electrical Power (Watts)
        P_elec = I * A * eta * derating_factor

        # Calculate heat transfer rates (Watts)
        Q_in = I * A * tau * alpha
        Q_loss = U_L * A * (T_cell - T_a)

        # Save the current step's data for plotting
        T_cell_array.append(T_cell)
        P_elec_array.append(P_elec)

        # 1st Law of Thermodynamics (Explicit Euler Integration)
        # Calculate the new temperature for the *next* time step
        dT_dt = (Q_in - Q_loss - P_elec) / (m * cp)
        T_cell = T_cell + (dT_dt * dt)

    return np.array(T_cell_array), np.array(P_elec_array), T_cell


# Case 6 - Battery Storage & Economic Analysis
original_Panels = 960
pack_capacity = 210  # kWh per tesla power pack
battery_efficiency = 1.0  # assume ideal unless told otherwise
battery_cost = 115  # $/kWh

# Time
DT = 5 / 60  # 5-minute timestep in hours


# Adjust for Daylight Savings Time (if needed)
def apply_dst(t, date):
    year = date.year

    # DST starts 2nd Sunday in March
    march = datetime(year, 3, 1)
    first_sunday_march = march + timedelta(days=(6 - march.weekday()) % 7)
    second_sunday_march = first_sunday_march + timedelta(days=7)

    # DST ends 1st Sunday in November
    november = datetime(year, 11, 1)
    first_sunday_november = november + timedelta(days=(6 - november.weekday()) % 7)

    if second_sunday_march <= date < first_sunday_november:
        return t - 1  # shift solar time back 1 hour
    else:
        return t


#  Idealized PEC power use for a typical summer and winter day
# summer days go from June 1 to Nov. 20 and the winter days are from Nov. 21 to May 31
def load_model(t, date):
    month = date.month
    day = date.day

    is_summer = (
            (month > 6 and month < 11) or
            (month == 6 and day >= 1) or
            (month == 11 and day <= 20)
    )

    if is_summer:
        # Summer
        if 0 <= t < 6:
            return 220
        elif 6 <= t < 19:
            return 580
        else:
            return 220
    else:
        # Winter
        if 0 <= t < 6:
            return 200
        elif 6 <= t < 18:
            return 300
        else:
            return 200


def battery_step(pv, load, current_capacity, max_capacity):
    net = pv - load  # positive = surplus, negative = deficit

    grid_import = 0
    grid_export = 0

    if net > 0:
        # Charge battery
        charge = net * DT

        available_space = max_capacity - current_capacity
        actual_charge = min(charge, available_space)

        current_capacity += actual_charge
        # leftover goes to grid
        grid_export = (charge - actual_charge) / DT

    else:
        # Discharge battery
        needed = -net * DT

        actual_discharge = min(needed, current_capacity)

        current_capacity -= actual_discharge

        remaining_deficit = needed - actual_discharge
        grid_import = remaining_deficit / DT

    return current_capacity, grid_import, grid_export


def get_scaled_pv_power(N, t_array, panel_scale, oci_val):
    # Different panel azimuthal angles based on size
    # 1x = original (960 panels at 46 deg)
    # 4x = 1x (original at 46) + 2x (at 0 deg) + 1x (at 23 deg)
    # 5x = 1x (original at 46) + 2x (at 0 deg) + 2x (at 23 deg)
    # 6x = 1x (original at 46) + 2x (at 0 deg) + 2x (at 23 deg) + 1x (at 46 deg)

    # Configuration Map: {azimuth: multiplier_of_960}
    if panel_scale == 1:
        config = {46: 1}
    elif panel_scale == 4:
        config = {46: 1, 0: 2, 23: 1}
    elif panel_scale == 5:
        config = {46: 1, 0: 2, 23: 2}
    elif panel_scale == 6:
        config = {46: 2, 0: 2, 23: 2}
    else:
        config = {46: 1}

    total_kw_array = np.zeros(len(t_array))
    all_amb_temps, _, _ = generate_yearly_5min_ambient_temps('austin_weather.csv')
    T_a_day = np.array(all_amb_temps[(N - 1) * 288: N * 288])

    for gamma_val, multiplier in config.items():
        # Get irradiance (res[1]) from simulate function
        res = simulate(N, t_array, beta=22, gamma=gamma_val, OCI_manual=oci_val)
        irr_array = res[1]

        # Calculate electrical power (W) for one panel based on case 5 irradiance and  dynamic temperature
        _, p_single_panel_watts, _ = simulate_case_5(irr_array, T_a_day, T_cell_initial=T_a_day[0])

        # Summation: (Watts * number_of_panels) / 1000 = kW
        # For scale 4x, this should be (p_single * 960 * 4) total
        num_panels_in_block = 960 * multiplier
        total_kw_array += (p_single_panel_watts * num_panels_in_block) / 1000.0

    return total_kw_array


def run_case_6_simulation(N, oci_val, panel_scale, battery_packs, pv_precalc=None, start_soc=0):
    t_5min = np.linspace(0, 24, 288)
    capacity_kwh = battery_packs * 210  # pack_capacity

    if pv_precalc is not None:
        pv_total_kw = pv_precalc
    else:
        pv_total_kw = get_scaled_pv_power(N, t_5min, panel_scale, oci_val)
    # ---------------------

    soc_history, grid_buy, grid_sell, load_history = [], [], [], []
    current_soc = start_soc# start with empty battery
    current_date = datetime(2026, 1, 1) + timedelta(days=int(N - 1))

    for i, t_local in enumerate(t_5min):
        t_corrected = apply_dst(t_local, current_date)
        load_kw = load_model(t_corrected, current_date)

        new_soc, g_in, g_out = battery_step(pv_total_kw[i], load_kw, current_soc, capacity_kwh)

        soc_history.append(current_soc)
        grid_buy.append(g_in)
        grid_sell.append(g_out)
        load_history.append(load_kw)
        current_soc = new_soc

    return {
        "time": t_5min, "pv_kw": pv_total_kw, "load_kw": np.array(load_history),
        "soc_kwh": np.array(soc_history), "grid_buy_kw": np.array(grid_buy),
        "final_soc": current_soc,
        "grid_sell_kw": np.array(grid_sell)
    }


# Economic Model - annual cost from 10 year cost
def compute_annual_total_cost(results, buy_price, sell_price, battery_packs):
    # Sum the 5-min steps and convert to kWh
    energy_imported_kwh = np.sum(results["grid_buy_kw"]) * (5 / 60)
    energy_exported_kwh = np.sum(results["grid_sell_kw"]) * (5 / 60)

    # Net electricity cost for one day
    daily_elec_cost = (energy_imported_kwh * buy_price) - (energy_exported_kwh * sell_price)

    # Battery
    capacity = battery_packs * pack_capacity
    battery_investment = capacity * 115

    # Maintenance over 10 years (5% of battery cost annually)
    maint_10yr = 0.05 * battery_investment * 10

    total_10yr = (daily_elec_cost * 365 * 10) + battery_investment + maint_10yr
    return total_10yr / 10


# Plotting functions for Case 6
def plot_case_6_performance(N, oci_val, panel_scale, battery_packs, actual_pv_kw, actual_load_kw):
    res = run_case_6_simulation(N, oci_val, panel_scale, battery_packs)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()  # Second axis for Battery Energy

    # --- Left Axis: Power (kW) ---
    ax1.plot(res["time"], res["pv_kw"], 'g-', linewidth=2, label='Model PV Power (kW)')
    ax1.plot(res["time"], res["load_kw"], 'r--', linewidth=2, label='Model Load (kW)')
    ax1.plot(res["time"], res["grid_buy_kw"], 'b:', linewidth=1.5, label='Grid Purchase (kW)')
    #ax1.plot(res["time"], -res["grid_sell_kw"], 'm-.', linewidth=1.5, label='Grid Export (kW)')

    # Scatter actual data
    t_actual = np.linspace(0, 24, len(actual_pv_kw))
    ax1.scatter(t_actual, actual_pv_kw, color='darkgreen', s=15, alpha=0.6, label='Actual PV (eGauge)')
    ax1.scatter(t_actual, actual_load_kw, color='darkred', s=15, alpha=0.6, label='Actual Load (eGauge)')

    # --- Right Axis: Battery Energy (kWh) ---
    #ax2.fill_between(res["time"], 0, res["soc_kwh"], color='orange', alpha=0.15, label='Battery Energy (kWh)')
    ax2.plot(res["time"], res["soc_kwh"], color='orange', linewidth=2, label='Battery SOC (kWh)')
    ax2.set_ylabel('Battery Energy (kWh)', color='orange', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.set_ylim(0, battery_packs * pack_capacity * 1.1)

    # Formatting
    ax1.set_xlabel('Time of Day (Hours)', fontsize=12)
    ax1.set_ylabel('Power (kW)', fontsize=12, fontweight='bold')
    title_str = f"Case 6: {panel_scale}x Panels, OCI={oci_val}, {battery_packs} Packs (Feb 5)"
    plt.title(title_str, fontsize=14)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize='small', ncol=2)

    ax1.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_case_6_economics(panel_scale, annual_actual_energy, monthly_max):
    buy_prices = np.linspace(0.06, 0.18, 7)
    pack_options = [0, 6, 12]
    sell_back_ratio = [0.5,1.0]
    t_5min = np.linspace(0, 24, 288)

    # Pre-calculate year of solar to reduce processing time
    print(f"Pre-calculating solar for {panel_scale}x expansion...")
    yearly_pv = []
    for N in range(1, 366):
        daily_oci = oci(monthly_max, N, annual_actual_energy)
        yearly_pv.append(get_scaled_pv_power(N, t_5min, panel_scale, daily_oci))

    for packs in pack_options:
        for ratio in sell_back_ratio:
            annual_costs = []
            for buy_p in buy_prices:
                sell_p=buy_p * ratio
                total_annual_spend = 0
                running_soc = 0  # start with empty battery each year
                for N in range(1, 366):
                    res = run_case_6_simulation(N, None, panel_scale, packs, pv_precalc=yearly_pv[N - 1], start_soc=running_soc)
                    import_kwh = np.sum(res["grid_buy_kw"]) * DT
                    export_kwh = np.sum(res["grid_sell_kw"]) * DT
                    total_annual_spend += (import_kwh * buy_p) - (export_kwh * sell_p)
                    running_soc = res["final_soc"]

                # 10-year math
                inv = (packs * 210) * 115
                maint = 0.05 * inv * 10
                annual_costs.append(((total_annual_spend * 10) + inv + maint) / 10)

            plt.plot(buy_prices, annual_costs, marker='o', label=f"{packs} Packs, ${ratio}x Sell")

    plt.xlabel('Austin Energy Purchase Price ($/kWh)')
    plt.ylabel('Average Total Yearly Cost ($)')
    plt.title(f'Economic Sensitivity Analysis: {panel_scale}x Panel Expansion')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()