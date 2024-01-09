# ------------------------------ #
        # Python's packages
        
from astropy.table import Table
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from typing import Tuple, Dict, List
from termcolor import colored
from scipy.optimize import curve_fit

import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------ #

"""
This module contains functions for calculating the Optimal Pointing Point (OPP) for the NICER (Neutron star Interior Composition Explorer) telescope. It includes methods for scaling count rates, calculating angular separations, signal-to-noise ratios, and determining optimal telescope pointing positions to maximize observational efficiency. The module integrates astropy for astronomical calculations, scipy for optimization, and matplotlib for visualization.

Functions:
- scaled_ct_rate: Scales a given count rate based on angular distance and effective area.
- ang_separation: Calculates the angular separation between two celestial objects.
- signal_to_noise: Calculates the signal-to-noise ratio given source and background count rates, instrumental background, and exposure time.
- nominal_pointing_info: Calculates and prints various information related to nominal pointing.
- calculate_opti_point: Calculates the optimal pointing position for NICER to maximize the signal-to-noise ratio.
- optimal_point_infos: Prints information for the optimal NICER pointing position.
- data_map: Plots a map of the Signal-to-Noise ratio as a function of NICER pointing.
- count_rates: Calculates X-ray count rates for nearby sources using PIMMS modeling.
- vignetting_factor: Calculates vignetting factors for nearby sources and the target object.
- write_fits_file: Writes the nearby sources table to a FITS file and opens it with TOPCAT.
- modeling: Performs modeling of nearby sources using a power-law model and creates a plot.

The module is specifically tailored for use with the NICER telescope, providing essential tools for astronomical data analysis in X-ray astronomy. It facilitates the optimization of observational strategies by calculating the most efficient pointing positions, considering various celestial and instrumental factors.

Note:
- This module is intended for astronomers and astrophysicists, especially those working with NICER data.
- It requires a comprehensive understanding of astrophysical concepts, telescope operation, and data analysis techniques in X-ray astronomy.
"""


def scaled_ct_rate(D, OptCtRate, effareaX, effareaY) -> float:
    """
    Scale a given count rate based on an angular distance and effective area.

    Parameters:
    D (float): The angular distance.
    OptCtRate (float): The original count rate.
    effareaX (array-like): Effective area data points (X values).
    effareaY (array-like): Effective area data points (Y values).

    Returns:
    float: The scaled count rate.
    """
    return OptCtRate * np.interp(D,effareaX,effareaY)


def ang_separation(reference, obj) -> Angle:
    """
    Calculate the angular separation between two celestial objects.

    Parameters:
    reference (SkyCoord): The reference object's coordinates.
    obj (SkyCoord): The coordinates of the object to which the separation is calculated.

    Returns:
    Quantity: The angular separation between the two objects.
    """
    return reference.separation(obj)


def signal_to_noise(SrcCtsRate, BkgSrcRates, InstBkgd, ExpTime) -> float:
    """
    Calculate the signal-to-noise ratio (S/N) given various parameters.

    Parameters:
    SrcCtsRate (float): Source count rate.
    BkgSrcRates (array-like): Count rates of background sources.
    InstBkgd (float): Instrumental and particle background count rate.
    ExpTime (float): Exposure time.

    Returns:
    float: The signal-to-noise ratio (S/N).
    """
    SNR = (SrcCtsRate*ExpTime) / np.sqrt(ExpTime*(SrcCtsRate+np.sum(BkgSrcRates)+InstBkgd))
    return SNR


def nominal_pointing_info(simulation_data, NearbySRCposition) -> None:
    """
    Calculate and print various information related to nominal pointing.

    Parameters:
    SIM_parameters (dict): Dictionary containing simulation parameters.
    NearbySRCposition (SkyCoord): Coordinates of nearby sources.

    Returns:
    None
    """

    object_data = simulation_data['object_data']
    telescop_data = simulation_data['telescop_data']
    
    SRCnominalDIST = ang_separation(NearbySRCposition, SkyCoord(ra=object_data['object_position'].ra, dec=object_data['object_position'].dec)).arcmin
    SRCscaleRates = scaled_ct_rate(SRCnominalDIST, simulation_data["nearby_sources_table"]["count_rate"], telescop_data["EffArea"], telescop_data["OffAxisAngle"])
    PSRcountrates = object_data['count_rate']

    print('PSR S/N at Nominal Pointing ' + str(signal_to_noise(PSRcountrates, SRCscaleRates, simulation_data["INSTbkgd"], simulation_data["EXPtime"])))
    print("PSR count rate at Nominal pointing = " + str(PSRcountrates) + "cts/sec")
    print("BKG sources count rate at Nominal Pointing = " + str(np.sum(SRCscaleRates)) + "cts/sec")
    print("             Individual BKG Sources rates:")
    print(str(SRCscaleRates))
    print("             BKG sources distance from PSR (\')")
    print(SRCnominalDIST)
    print("--------------------------------------------------")


def calculate_opti_point(simulation_data, nearby_src_position) -> Tuple[int, float, float, dict]:
    """
    Calculate the optimal pointing position for a telescope to maximize the signal-to-noise ratio (SNR).

    Args:
        simulation_data (dict): A dictionary containing simulation data including telescope data and object data.
        nearby_src_position (numpy.ndarray): An array containing the positions of nearby sources.

    Returns:
        Tuple[int, float, float, dict]: A tuple containing:
            - OptimalPointingIdx (int): Index of the optimal pointing position.
            - SRCoptimalSEPAR (float): Angular separation of nearby sources from the optimal pointing position (in arcminutes).
            - SRCoptimalRATES (float): Scaled count rates of nearby sources at the optimal pointing position.
            - vector_dictionary (dict): A dictionary containing various vectors and results, including:
                - 'SampleRA': Array of right ascensions for sample pointing positions (in degrees).
                - 'SampleDEC': Array of declinations for sample pointing positions (in degrees).
                - 'PSRrates': Array of scaled count rates for the target object at sample pointing positions.
                - 'SRCrates': Array of total scaled count rates for nearby sources at sample pointing positions.
                - 'SNR': Array of signal-to-noise ratios at sample pointing positions.
    """
    
    min_value, max_value, step = -7.0, 7.1, 0.05
    DeltaRA = Angle(np.arange(min_value, max_value, step), unit=u.deg)/60
    DeltaDEC = Angle(np.arange(min_value, max_value, step), unit=u.deg)/60
    
    telescop_data = simulation_data["telescop_data"]
    object_data = simulation_data["object_data"]
    nearby_sources_table = simulation_data['nearby_sources_table']
    
    RA_grid, DEC_grid = np.meshgrid(DeltaRA, DeltaDEC)

    SampleRA = object_data["object_position"].ra.deg + RA_grid.flatten().deg
    SampleDEC = object_data["object_position"].dec.deg + DEC_grid.flatten().deg

    NICERpointing = SkyCoord(ra=SampleRA*u.deg, dec=SampleDEC*u.deg)

    PSRseparation = ang_separation(object_data["object_position"], NICERpointing).arcmin
    nearby_src_position = nearby_src_position.reshape(1, -1)
    NICERpointing = NICERpointing.reshape(-1, 1)
    SRCseparation = ang_separation(nearby_src_position, NICERpointing).arcmin

    PSRcountrateScaled = scaled_ct_rate(PSRseparation, object_data['count_rate'], telescop_data["EffArea"], telescop_data["OffAxisAngle"])

    count_rate = nearby_sources_table['count_rate']
    SRCcountrateScaled = scaled_ct_rate(SRCseparation, count_rate, telescop_data["EffArea"], telescop_data["OffAxisAngle"])

    SNR, PSRrates, SRCrates  = np.zeros((3, len(DeltaRA) * len(DeltaDEC)))
    for item in range(len(PSRcountrateScaled)):
        PSRrates[item] = PSRcountrateScaled[item]
        SRCrates[item] = np.sum(SRCcountrateScaled[item])
        SNR[item] = signal_to_noise(PSRrates[item], SRCrates[item], simulation_data["INSTbkgd"], simulation_data["EXPtime"])

    OptimalPointingIdx = np.where(SNR==max(SNR))[0][0]

    SRCoptimalSEPAR = ang_separation(nearby_src_position, SkyCoord(ra=SampleRA[OptimalPointingIdx]*u.degree, dec=SampleDEC[OptimalPointingIdx]*u.degree)).arcmin
    SRCoptimalRATES = scaled_ct_rate(SRCoptimalSEPAR, nearby_sources_table["count_rate"], telescop_data["EffArea"], telescop_data["OffAxisAngle"])

    vector_dictionary = {
        'SampleRA': SampleRA,
        'SampleDEC': SampleDEC,
        'PSRrates': PSRrates,
        'SRCrates': SRCrates,
        'SNR': SNR
    }

    return OptimalPointingIdx, SRCoptimalSEPAR, SRCoptimalRATES, vector_dictionary


def optimal_point_infos(vector_dictionary, OptimalPointingIdx, SRCoptimalRATES) -> None:
    """
    Print information for the optimal NICER pointing that maximizes the signal-to-noise ratio (S/N).

    This function prints information about the S/N ratio, pulsar count rate, background sources count rate,
    individual background source rates, and the optimal pointing coordinates for the NICER telescope.

    Args:
        vector_dictionary (dict): A dictionary contai
        # if isinstance(model_value, tuple):
        #     model_value = model_value[0]ning result vectors, including sampled RA and DEC positions,
                                  pulsar count rates, SRC count rates, and the S/N ratio for each pointing.
        OptimalPointingIdx (int): The index of the optimal pointing in the result vectors.
        SRCoptimalRATES (float): The SRC count rate at the optimal pointing.
    """

    # Print info for the optimal NICER pointing that maximizes the S/N ratio
    print ("PSR S/N at Optimal Pointing " + str(vector_dictionary['SNR'][OptimalPointingIdx]))
    print ("PSR count rate at Optimal pointing = " + str(vector_dictionary["PSRrates"][OptimalPointingIdx]) + " cts/sec")
    print ("BKG sources count rate at Optimal pointing = " + str(vector_dictionary["SRCrates"][OptimalPointingIdx]) + " cts/sec")
    print ("     Individual BKG Sources: " )
    print (str(SRCoptimalRATES))
    #print "     Distance from Optimal Pointing (\")"
    #print str(SRCoptimalSEPAR)
    print ("Optimal Pointing:  " + str(vector_dictionary["SampleRA"][OptimalPointingIdx]) + "  " + str(vector_dictionary["SampleDEC"][OptimalPointingIdx]))
    print ("----------------------------------------------------------------------")


def data_map(simulation_data, vector_dictionary, OptimalPointingIdx, NearbySRCposition) -> None:
    """
    Plot the map of the Signal-to-Noise (S/N) ratio as a function of NICER pointing.

    Parameters:
    - SIM_parameters (dict): A dictionary containing simulation parameters, including the pulsar position.
    - Vector_Dictionary (dict): A dictionary containing vector data, including SampleRA, SampleDEC, and SNR.
    - OptimalPointingIdx (int): The index of the optimal pointing in Vector_Dictionary.
    - NearbySRCposition (SkyCoord): SkyCoord object representing the positions of nearby sources.

    Returns:
    None
    
    """
    os_dictionary = simulation_data["os_dictionary"]
    object_data = simulation_data['object_data']

    # Plot the map of S/N ratio as function of NICER pointing
    ra_opti = vector_dictionary['SampleRA'][OptimalPointingIdx]
    dec_opti = vector_dictionary['SampleDEC'][OptimalPointingIdx]
    
    nearby_ra = [NearbySRCposition[item].ra.value for item in range(len(NearbySRCposition))]
    nearby_dec = [NearbySRCposition[item].dec.value for item in range(len(NearbySRCposition))]
    
    figure, axes = plt.subplots(1, 1, figsize=(15, 8))
    figure.suptitle(f"S/N map for {object_data['object_name']}\nOptimal pointing point : {ra_opti} deg, {dec_opti} deg")
    
    axes.invert_xaxis()
    sc = axes.scatter(vector_dictionary['SampleRA'], vector_dictionary['SampleDEC'], c=vector_dictionary["SNR"], s=10, edgecolor='face')
    axes.scatter(nearby_ra, nearby_dec, marker='.', color='black', label=f"Nearby sources : {len(nearby_ra)}")
    axes.scatter(object_data["object_position"].ra, object_data["object_position"].dec, marker='*', color='green', label=f"{object_data['object_name']}")
    axes.scatter(ra_opti, dec_opti, s=50, marker='+', color='red', label="Optimal Pointing Point")
    
    axes.set_xlabel('Right Ascension [deg]', fontsize='large')
    axes.set_ylabel('Declination [deg]', fontsize='large')
    axes.legend(loc="upper right", ncol=2)
    cbar = figure.colorbar(sc, ax=axes)
    cbar.set_label('S/N')
    
    key = simulation_data["os_dictionary"]["catalog_key"]
    name = object_data["object_name"]
    plt.savefig(os.path.join(os_dictionary['img'], f"{key}_SNR_{name}.png".replace(" ", "_")))
    plt.show()


def count_rates(nearby_src_table, model_dictionary, telescop_data) -> Tuple[List[float], Table]:
    """
    Calculate X-ray count rates for nearby sources using PIMMS modeling.

    This function calculates X-ray count rates for a set of nearby sources based on their model information and associated parameters.
    It uses PIMMS (Portable, Interactive Multi-Mission Simulator) to perform the modeling and computes the count rates.

    Parameters:
        nearby_src_table (Table): A table containing data on nearby sources, including model information, X-ray flux, and column density.
        model_dictionary (dict): A dictionary mapping sources (e.g., "src_0") to model and parameter information.

    Returns:
        tuple: A tuple containing two elements:
            - A NumPy array of X-ray count rates for each nearby source.
            - An updated 'nearby_src_table' with the added 'Count Rates' column.

    Note:
    - PIMMS modeling commands are generated for each source based on the model, model value, X-ray flux, and column density.
    - The 'Count Rates' column is added to the 'nearby_src_table' with the calculated count rates.
    """
    number_source = len(model_dictionary)
    count_rates = np.array([], dtype=float)
    
    telescop_name = telescop_data['telescop_name']
    min_value = telescop_data['min_value']
    max_value = telescop_data['max_value']
    energy_band = telescop_data['energy_band']
    
    for item in range(number_source):
        model = model_dictionary[f"src_{item}"]["model"]
        model_value = model_dictionary[f"src_{item}"]["model_value"]
        xmm_flux =  model_dictionary[f"src_{item}"]["flux"]
        nh_value = model_dictionary[f"src_{item}"]["column_dentsity"]
                
        pimms_cmds = f"instrument {telescop_name} {min_value}-{max_value}\nfrom flux ERGS {energy_band}\nmodel galactic nh {nh_value}\nmodel {model} {model_value} 0.0\ngo {xmm_flux}\nexit\n"
        
        with open('pimms_script.xco', 'w') as file:
            file.write(pimms_cmds)
            file.close()

        result = subprocess.run(['pimms', '@pimms_script.xco'], stdout=subprocess.PIPE).stdout.decode(sys.stdout.encoding)
        count_rate = float(result.split("predicts")[1].split('cps')[0])
        count_rates = np.append(count_rates, count_rate)
        
    nearby_src_table["count_rate"] = count_rates
        
    return count_rates, nearby_src_table


def vignetting_factor(OptimalPointingIdx, vector_dictionary, simulation_data, data, nearby_sources_table) -> Tuple[List[float], Table]:
    """
    Calculate the vignetting factors for nearby sources and the target object based on their distances.

    Args:
        OptimalPointingIdx (int): Index of the optimal pointing position.
        vector_dictionary (dict): A dictionary containing vectors for RA and DEC.
        simulation_data (dict): A dictionary containing simulation data including object data, telescope data, etc.
        data (tuple): A tuple containing RA, DEC, and name data.
        nearby_sources_table (Table): A table containing information about nearby sources.

    Returns:
        Tuple[List[float], Table]: A tuple containing the calculated vignetting factors and an updated nearby sources table.
    """
    
    ra, dec, name = data
    
    object_data = simulation_data["object_data"]
    EffArea, OffAxisAngle = simulation_data["telescop_data"]["EffArea"], simulation_data["telescop_data"]["OffAxisAngle"]
    
    optipoint_ra, optipoint_dec = vector_dictionary['SampleRA'][OptimalPointingIdx], vector_dictionary['SampleDEC'][OptimalPointingIdx]
    
    def calculate_vignetting_factor(D, effareaX, effareaY):
        return np.interp(D,effareaX,effareaY)
    
    vignetting_factor, distance = np.array([], dtype=float), np.array([], dtype=float)
    
    for index in range(len(nearby_sources_table)):
        SRCposition  = SkyCoord(ra=nearby_sources_table[ra][index]*u.degree, dec=nearby_sources_table[dec][index]*u.degree)
        SRCnominalDIST = ang_separation(SRCposition, SkyCoord(ra=optipoint_ra, dec=optipoint_dec, unit=u.deg)).arcmin
        distance = np.append(distance, SRCnominalDIST)
        vignetting = calculate_vignetting_factor(SRCnominalDIST, EffArea, OffAxisAngle)
        vignetting_factor = np.append(vignetting_factor, vignetting)
    
    optimal_pointing_point = SkyCoord(ra=optipoint_ra, dec=optipoint_dec, unit=u.deg)
    psr_position = SkyCoord(ra=object_data['object_position'].ra, dec=object_data['object_position'].dec, unit=u.deg)
    distance_psr_to_optipoint = ang_separation(psr_position, optimal_pointing_point).arcmin
    vignetting_factor_psr2optipoint = calculate_vignetting_factor(distance_psr_to_optipoint, EffArea, OffAxisAngle)

    max_vignet, min_distance  = np.max(vignetting_factor), np.min(distance)
    max_vignet_index, min_distance_index = np.argmax(vignetting_factor), np.argmin(distance)
    
    print(f"\nThe closest source of the optimal pointing point is : {colored(nearby_sources_table[name][min_distance_index], 'magenta')}.")
    print(f"The distance between {colored(nearby_sources_table[name][min_distance_index], 'yellow')} and optimal pointing point is {colored(min_distance, 'blue')} arcmin.\n"
          f"With a vignetting factor of : {colored(max_vignet, 'light_green')} ")
    print(f"The distance between {colored(object_data['object_name'], 'yellow')} and optimal pointing point is {colored(distance_psr_to_optipoint, 'blue')} arcmin,\n"
          f"with a vagnetting factor of : {colored(vignetting_factor_psr2optipoint, 'light_green')}")
    
    nearby_sources_table["vignetting_factor"] = vignetting_factor
    
    return vignetting_factor, nearby_sources_table


def write_fits_file(nearby_sources_table, simulation_data) -> None:
    """
    Write the nearby sources table to a FITS file and open it with TOPCAT.

    Args:
        nearby_sources_table (Table): A table containing information about nearby sources.
        simulation_data (dict): A dictionary containing simulation data and file path information.

    Returns:
        None
    """
    
    try:
        os_dictionary = simulation_data["os_dictionary"]
        key = os_dictionary["catalog_key"]
        cloesest_dataset_path = os_dictionary["cloesest_dataset_path"]
        nearby_sources_table_path = os.path.join(cloesest_dataset_path, f"{key}_nearby_sources_table.fits").replace("\\", "/")
        nearby_sources_table.write(nearby_sources_table_path, format='fits', overwrite=True)
        print(f"Nearby sources table was created in : {colored(nearby_sources_table_path, 'magenta')}")
        
        topcat_path = os_dictionary["topcat_software_path"]
        command = f"java -jar {topcat_path} {nearby_sources_table_path}"
        subprocess.run(command)
        
    except Exception as error:
        print(f"{colored('An error occured : ', 'red')} {error}")
    
    
def modeling(vignetting_factor: List, simulation_data: Dict, column_dictionary: Dict, catalog_name: str) -> None:
    """
    Perform modeling of nearby sources using a power-law model and create a plot.

    Args:
        vignetting_factor (List): List of calculated vignetting factors for nearby sources.
        simulation_data (Dict): A dictionary containing simulation data and object information.
        column_dictionary (Dict): A dictionary containing column names for flux and energy band.
        catalog_name (str): Name of the catalog used for modeling.

    Returns:
        None
    """
    
    object_data = simulation_data["object_data"]
    os_dictionary = simulation_data["os_dictionary"]
    nearby_sources_table = simulation_data["nearby_sources_table"]
    number_source = len(nearby_sources_table)
    
    flux_name, err_flux_name, energy_band = column_dictionary["band_flux_obs"], column_dictionary["band_flux_obs_err"], column_dictionary["energy_band"]
    photon_index_list, constant_list = np.array([], dtype=float), np.array([], dtype=float)
    main_flux_obs, main_err_flux_obs = [], []
    
    

    def power_law(vignetting_factor, energy_band, constant, gamma):
        sigma = column_dictionary["sigma"]
        return (constant * energy_band ** (-gamma) * np.exp(-sigma * 3e20)) * vignetting_factor
    
    
    def sum_power_law(energy_band, constant, photon_index):
        summed_power_law = 0.0
        for cst, pho in zip(constant, photon_index):
            summed_power_law += (cst * energy_band ** (-pho))
        return summed_power_law

    
    for item in range(number_source):
        flux_obs = [nearby_sources_table[name][item] for name in flux_name]
        main_flux_obs.append(flux_obs)
        
        err_flux_obs = [nearby_sources_table[err_name][item] for err_name in err_flux_name]
        main_err_flux_obs.append(err_flux_obs)
        try:
            popt, pcov = curve_fit(lambda energy_band, constant, gamma: power_law(vignetting_factor[item], energy_band, constant, gamma), energy_band, flux_obs, sigma=err_flux_obs)
            constant, photon_index = popt
        except Exception as error:
            constant = 1e-14
            photon_index = 1.7
        photon_index_list = np.append(photon_index_list, photon_index)
        constant_list = np.append(constant_list, constant)
        
        
    flux_obs_array = []
    for index in range(number_source):
        flux_obs_array.append(power_law(vignetting_factor[index], energy_band, constant_list[index], photon_index_list[index]))
        
    percentiles = np.percentile(flux_obs_array, (16, 50, 84), axis=0)
    
    nrows, ncols = 1, 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 6), sharex=True)
    fig.suptitle(f"Modeling nearby sources with {catalog_name}\n{object_data['object_name']}", fontsize=16)
    fig.text(0.5, 0.04, 'Energy [keV]', ha='center', va='center', fontsize=12)
    fig.text(0.07, 0.5, 'Flux [erg/cm2/s]', ha='center', va='center', rotation='vertical', fontsize=12)

    ax00, ax01, ax02 = axes[0], axes[1], axes[2]
    ax00.set_title("All nearby sources power law")
    ax00.set_xscale("log")

    for index in range(number_source):
        ax00.plot(energy_band, power_law(vignetting_factor[index], energy_band, constant_list[index], photon_index_list[index]))
    
    ax01.plot(energy_band, percentiles[1], color="navy", label="mean value")
    ax01.plot(energy_band, percentiles[0], color="royalblue", ls="--", linewidth=1.5, label="$16^{th}$ percentile")
    ax01.plot(energy_band, percentiles[2], color="midnightblue", ls="--", linewidth=1.5, label="$84^{th}$ percentile")
    ax01.fill_between(energy_band, percentiles[0], percentiles[2], alpha=0.3, color="navy", hatch='\\'*3, label="envelop")
    ax01.legend(loc="upper right", fontsize=8, ncols=2)

    ax02.plot(energy_band, sum_power_law(energy_band, constant_list, photon_index_list), color="darkmagenta", ls='-.', label="summed power law")
    ax02.legend(loc="upper right", fontsize=10)
    
    key = simulation_data["os_dictionary"]["catalog_key"]
    name = object_data['object_name']
    plt.savefig(os.path.join(os_dictionary["img"], f"{key}_modeling_{name}.png".replace(" ", "_")))
    plt.show()
    