# ------------------------------ #
        # Python's packages
        
from astropy.io import fits
from astropy.table import Table
from termcolor import colored
from typing import Dict
from tqdm import tqdm
from jaxspec.data.util import fakeit_for_multiple_parameters
from typing import List

import os
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------ #

"""
This module provides functions for astronomical data analysis and spectral modeling. It leverages the astropy library for handling astronomical data tables and FITS files, and uses jaxspec for spectral modeling.

Functions:
- cross_catalog_index: Determines the indices in a nearby sources table that correspond to sources found in a master source cone file.
- modeling_source_spectra: Generates model spectra for sources in a nearby sources table using specified instrument and model parameters.
- total_plot_spectra: Creates and saves a plot of spectral modeling data from a specified catalog and returns spectral data.
- write_txt_file: Writes the spectral modeling data into a formatted text file.

The module integrates with astropy.io, astropy.table, termcolor, tqdm, and jaxspec, among others, for its functionality. It is designed to facilitate the cross-catalog matching of astronomical sources, the generation and analysis of spectral data, and the efficient presentation and storage of the results.

Note:
- The module is intended for use in the field of astrophysics, particularly in the analysis of spectral data from various astronomical sources.
- It requires a proper understanding of astronomical data formats, spectral analysis techniques, and the specific instruments and models involved in the spectral data generation.
"""


def cross_catalog_index(output_name: str, key: str, iauname: str, nearby_sources_table: Table) -> List:
    """
    Determines the indices in a nearby sources table that correspond to sources found in a master source cone file.

    Args:
    output_name (str): Directory path where the master source cone file is located.
    key (str): Key representing the catalog (e.g., "CS_Chandra" or "Chandra").
    iauname (str): Column name in the nearby sources table representing the IAU name of sources.
    nearby_sources_table (Table): A table containing data about nearby sources.

    The function reads the master source cone FITS file to extract source names. It then matches these 
    names with those in the nearby sources table to find corresponding indices.

    Returns:
    List: A list of indices from the nearby sources table that match with the master source cone.

    Note:
    - This function is specific to astronomical data analysis where cross-catalog matching is required.
    """
    
    master_source_cone_path = os.path.join(output_name, "Master_source_cone.fits").replace("\\", "/")
    with fits.open(master_source_cone_path) as data:
        master_source_cone = Table(data[1].data)
    
    if key == "CS_Chandra":
        key = "Chandra"
    
    msc_name = [name for name in master_source_cone[key] if name != ""]
    var_index_in_nearby_sources_table = []
    for name in msc_name:
        if name in nearby_sources_table[iauname]:
            index_in_table = list(nearby_sources_table[iauname]).index(name)
            var_index_in_nearby_sources_table.append(index_in_table)
            
    return var_index_in_nearby_sources_table


def modeling_source_spectra(nearby_sources_table: Table, instrument, model, var_index) -> List:
    """
    Generates model spectra for sources in a nearby sources table using specified instrument and model parameters.

    Args:
    nearby_sources_table (Table): A table containing data of nearby sources.
    instrument: Instrument object with parameters for generating spectra.
    model: Spectral model to be applied for the spectra generation.
    var_index (List): List of indices indicating variable sources in the nearby sources table.

    The function iterates through the nearby sources table, applying the spectral model to each source. 
    It accounts for vignetting factors and adjusts parameters like 'N_H' and 'alpha' based on source data.

    Returns:
    List: A tuple of two lists - the total spectra and total variable spectra for all sources.

    Note:
    - The function assumes 'vignetting_factor', 'Nh', and 'Photon Index' columns in the nearby sources table.
    - 'fakeit_for_multiple_parameters' function is used for generating fake spectra based on the given model and instrument.
    """
    
    print(f"\n{colored('Modeling spectra...', 'yellow', attrs=['underline'])}")
    total_spectra = []
    total_var_spectra = []
    size = 10_000
    
    for index, vignet_factor in tqdm(enumerate(nearby_sources_table["vignetting_factor"])):
        parameters = {}
        parameters = {
            "tbabs_1": {"N_H": np.full(size, nearby_sources_table["Nh"][index]/1e22)},
            "powerlaw_1": {
                "alpha": np.full(size, nearby_sources_table["Photon Index"][index] if nearby_sources_table["Photon Index"][index] > 0.0 else 1.7),
                "norm": np.full(size, 1e-5),
            }
        }
        
        spectra = fakeit_for_multiple_parameters(instrument=instrument, model=model, parameters=parameters) * vignet_factor

        if index in var_index:
            total_var_spectra.append(spectra)
        
        total_spectra.append(spectra)
        
    return total_spectra, total_var_spectra


def total_plot_spectra(total_spectra: List, total_var_spectra: List, instrument, simulation_data: Dict, catalog_name: str) -> Dict:
    """
    Generates and saves a plot of spectral modeling data from a specified catalog and returns spectral data.

    Args:
    total_spectra (List): A list containing spectra from nearby sources.
    total_var_spectra (List): A list containing variability spectra data.
    instrument: An object containing instrument-specific data such as output energies.
    simulation_data (Dict): A dictionary containing simulation parameters and paths.
    catalog_name (str): Name of the catalog used for spectral modeling.

    The function creates three subplots:
    1. Spectra from Nearby Sources: Plots the median spectra for each source.
    2. Sum of Spectra: Shows the sum of all spectra.
    3. Spectrum Summed with Variability Sources Error: Includes error bars representing variability.

    The plot includes logarithmic scaling on both axes, and the x-axis represents energy in keV. 
    The y-axis represents counts. The plot is saved as a PNG file in a directory specified in `simulation_data`.

    Returns:
    Dict: A dictionary containing energy, counts, and their upper and lower limits.

    Note:
    - The saved plot is named based on the catalog name and the object of interest from the simulation data.
    - The function assumes `instrument` has an attribute `out_energies` for energy values.
    """
    
    object_data = simulation_data["object_data"]
    os_dictionary = simulation_data["os_dictionary"]
    graph_data = {"min_lim_x": 0.2,
                  "max_lim_x": 10.0,
                  "percentile_0": 10,
                  "percentile_2": 90}

    figure_1, axes = plt.subplots(1, 3, figsize=(17, 9), sharey=True)
    figure_1.suptitle(f"Spectral modeling close to {object_data['object_name']}\ncatalog : {catalog_name}", fontsize=20)
    figure_1.text(0.5, 0.04, 'Energy [keV]', ha='center', va='center', fontsize=16)
    figure_1.text(0.085, 0.5, 'Counts', ha='center', va='center', rotation='vertical', fontsize=16)

    for ax in axes:
        ax.set_xlim([graph_data["min_lim_x"], graph_data["max_lim_x"]])
        ax.loglog()

    ax0 = axes[0]
    for spectra in total_spectra:
        ax0.step(instrument.out_energies[0],
                np.median(spectra, axis=0),
                where="post")
    ax0.set_title("Spectra from Nearby Sources")

    spectrum_summed = 0.0
    for item in range(len(total_spectra)):
        spectrum_summed += total_spectra[item]

    spectrum_var_summed = 0.0
    for item in range(len(total_var_spectra)):
        spectrum_var_summed += total_var_spectra[item]  

    y_upper = np.median(spectrum_summed, axis=0) + np.median(spectrum_var_summed, axis=0)
    y_lower = np.median(spectrum_summed, axis=0) - np.median(spectrum_var_summed, axis=0)

    ax1 = axes[1]
    ax1.step(instrument.out_energies[0],
            np.median(spectrum_summed, axis=0),
            where='post', color='black'
            )

    ax1.set_title("Sum of spectra")

    ax2 = axes[2]
    ax2.errorbar(instrument.out_energies[0], y=np.median(spectrum_summed, axis=0), yerr=np.median(spectrum_var_summed, axis=0), 
                fmt="none", ecolor='red', capsize=2, capthick=3,
                label='error')
    ax2.step(instrument.out_energies[0], np.median(spectrum_summed, axis=0), color='black', label="sum powerlaw")
    ax2.set_title("Spectrum Summed with var sources error")
    ax2.legend(loc='upper right')
    ax2.loglog()

    key = simulation_data["os_dictionary"]["catalog_key"]
    img_path = os.path.join(os_dictionary['img'], f"{key}_spectral_modeling_close_to_{object_data['object_name']}.png".replace(" ", "_")).replace("\\", "/")
    plt.savefig(img_path)
    plt.show()
    
    data = {
        "Energy": instrument.out_energies[0],
        "Counts": np.median(spectrum_summed, axis=0),
        "Upper limit": y_upper,
        "Lower limit": y_lower
    }
    
    return data


def write_txt_file(simulation_data: Dict, data: Dict) -> None:
    """
    Writes the spectral modeling data into a formatted text file.

    Args:
    simulation_data (Dict): A dictionary containing simulation parameters and paths.
    data (Dict): A dictionary with spectral data including energy, counts, and their upper and lower limits.

    The function creates a text file in the specified directory within the simulation data dictionary. 
    The file includes a header and rows of data, each containing energy values, count rates, and their upper 
    and lower limits. The data is formatted for readability and analysis purposes.

    Note:
    - The output text file is named based on the 'catalog_key' value in the simulation data dictionary.
    - The function expects 'data' to have keys corresponding to 'Energy', 'Counts', 'Upper limit', 
      and 'Lower limit' and their associated values.
    """
    
    catalog_directory = simulation_data['os_dictionary']["catalog_directory"]
    key = simulation_data["os_dictionary"]["catalog_key"]
    txt_path = os.path.join(catalog_directory, f'{key}_output_modeling_plot.txt').replace("\\", "/")
    
    data_to_txt = [
        list(data.keys())
    ]
    
    energy, counts, y_upper, y_lower = list(data.values())
    data_to_txt.extend([energy[index], counts[index], y_upper[index], y_lower[index]] for index in range(len(energy)))
    
    with open(txt_path, 'w') as file:
        header = "{:<15} {:<15} {:<15} {:<15}".format(*data_to_txt[0])
        file.write(header + "\n")

        for row in data_to_txt[1:]:
            new_row = "{:<10.5f}     {:<10.5f}       {:<10.5f}       {:<10.5f}".format(*row)
            file.write(new_row + "\n")
            
    print(f"\n{colored(f'{key}_output_modeling_plot.txt', 'yellow')} has been created in {colored(txt_path, 'blue')}")

