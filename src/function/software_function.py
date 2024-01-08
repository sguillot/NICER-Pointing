# ------------------------------ #
        # Python's packages
        
from astropy.io import fits
from astropy.table import Table
from termcolor import colored
from astropy.coordinates import SkyCoord
from typing import Dict
from astropy import units as u
from astropy.time import Time
from tqdm import tqdm

# ---------- import class ---------- #

import catalog_class.MasterSourceClass

# ---------------------------------- #

import os
import numpy as np
import catalog_information as dict_cat
import matplotlib.pyplot as plt

# ------------------------------ #

"""
This module is designed for handling and processing astronomical data from various catalogs, specifically focusing on multi-instrument source analysis. It includes functionalities for loading, processing, and plotting data from catalogs like Swift, XMM, and others. The module integrates Astropy for data handling and Matplotlib for plotting, among other packages.

Functions:

- load_relevant_sources(cat, file_to_load): Loads and processes source data from a specified catalog, returning it in a structured format. This includes sorting and extracting unique sources, computing additional parameters like time steps, observation IDs, and flux data.

- load_master_sources(file_to_load): Loads multi-instrument source data from a master source file, integrating data from various catalogs. It creates a comprehensive view of each master source by combining data from multiple catalogs.

- master_source_plot(master_sources, simulation_data, number_graph): Generates and saves plots for multi-instrument sources based on catalog data. It combines data from different catalogs and plots energy bands and fluxes for a specified number of master sources.

The module is essential for astronomers and data analysts working with multi-instrument astronomical data. It simplifies the process of loading and integrating data from various catalogs, and provides tools for visualizing this data in a meaningful way. The module's reliance on well-established Python libraries like Astropy and Matplotlib ensures robust and accurate data processing and visualization.
"""

def load_relevant_sources(cat: str, file_to_load: str) -> Dict:
    """
    Loads and processes source data from a specified catalog and returns it in a structured format.

    Args:
    cat (str): Name of the catalog to load (e.g., "Swift", "XMM").
    file_to_load (str): Path to the FITS file containing the raw catalog data.

    The function loads data from a FITS file, processes it by sorting and extracting unique sources, 
    and computes additional parameters like time steps, observation IDs, and flux data.

    Returns:
    Dict: A dictionary of processed source data, with keys being the names of the sources and values being 
          the corresponding data structured in a pre-defined format.

    Note:
    - The function handles different catalogs with specific processing needs, particularly for time-based data.
    - An error in data loading is handled and reported.
    """
    
    print(f"Loading {cat}...")
    try:
        with fits.open(file_to_load, memmap=True) as raw_data:
            sources_raw = Table(raw_data[1].data)
            sources_raw = sources_raw[np.argsort(sources_raw[dict_cat.src_names[cat]])]
    except Exception as error:
        print(f"An error occured : {colored(error, 'magenta')}")
        
    indices_for_source = [i for i in range(1, len(sources_raw)) if (sources_raw[dict_cat.src_names[cat]][i] != sources_raw[dict_cat.src_names[cat]][i - 1])]

    if cat == "Swift":
        time_start_obs = Time(sources_raw["StartTime_UTC"], format="iso").mjd
        time_end_obs = Time(sources_raw["StopTime_UTC"], format="iso").mjd
        time_start_obs = np.split(time_start_obs, indices_for_source)
        time_end_obs = np.split(time_end_obs, indices_for_source)
        
    time_steps = np.split(np.array(sources_raw[dict_cat.dictionary_catalog[cat]["time_name"]]), indices_for_source)
    if cat in ("XMM", "Swift", "Stacked"):
        obs_ids = np.split(np.array(sources_raw[dict_cat.dictionary_catalog[cat]["obsid_name"]]), indices_for_source)
    else:
        obs_ids = [[] for elt in indices_for_source]
    names = np.split(np.array(sources_raw[dict_cat.src_names[cat]]), indices_for_source)

    band_flux, band_flux_errors_pos, band_flux_errors_neg = [], [], []
    
    flux = np.split(dict_cat.dictionary_catalog[cat]["conv_factor"]*np.array(sources_raw[dict_cat.dictionary_catalog[cat]["flux_obs"]]), indices_for_source)
    flux_errors_neg = np.split(dict_cat.dictionary_catalog[cat]["conv_factor"]*np.array(sources_raw[dict_cat.dictionary_catalog[cat]["flux_obs_err"][0]]), indices_for_source)
    flux_errors_pos = np.split(dict_cat.dictionary_catalog[cat]["conv_factor"]*np.array(sources_raw[dict_cat.dictionary_catalog[cat]["flux_obs_err"][1]]), indices_for_source)
    flux_errors = [[flux_neg, flux_pos] for (flux_neg, flux_pos) in zip(flux_errors_neg, flux_errors_pos)]

    band_flux_obs = dict_cat.dictionary_catalog[cat]["band_flux_obs"] #band_flux_obs_err
    band_flux_obs_err_neg, band_flux_obs_err_pos = dict_cat.dictionary_catalog[cat]["band_flux_obs_err"][0], dict_cat.dictionary_catalog[cat]["band_flux_obs_err"][1]
    for band_flux_name, band_flux_err_neg_name, band_flux_err_pos_name in zip(band_flux_obs, band_flux_obs_err_neg, band_flux_obs_err_pos):
        band_flux.append(np.array(sources_raw[band_flux_name]))
        band_flux_errors_pos.append(np.array(sources_raw[band_flux_err_pos_name]))
        band_flux_errors_neg.append(np.array(sources_raw[band_flux_err_neg_name]))
        
    band_flux = np.transpose(np.array(band_flux))
    band_flux_errors_pos = np.transpose(np.array(band_flux_errors_pos))
    band_flux_errors_neg = np.transpose(np.array(band_flux_errors_neg))
    band_flux = np.split(band_flux, indices_for_source)
    band_flux_errors_pos = np.split(band_flux_errors_pos, indices_for_source)
    band_flux_errors_neg = np.split(band_flux_errors_neg, indices_for_source)

    band_flux_err = [[flux_neg, flux_pos] for (flux_neg, flux_pos) in zip(band_flux_errors_neg, band_flux_errors_pos)]
    dict_sources = {}
    
    #This loops on all sources, to build the Source objects
    for (index, flux, flux_error, time, name, band_flux, band_flux_err, obsid) in zip(range(len(flux)),flux, flux_errors, time_steps, names, band_flux, band_flux_err, obs_ids):
            swift_stacked_flux=[]
            swift_stacked_flux_err=[[],[]]
            swift_stacked_times=[[],[]]
            if cat == "Swift":
                tab_src_timestartobs = time_start_obs[index]
                tab_src_timeendobs = time_end_obs[index]

                #We select the stacked Swift detections first
                swift_stacked_flux=flux[obsid>1e10]
                swift_stacked_flux_err=[flux_error[0][obsid>1e10],flux_error[1][obsid>1e10]]
                swift_stacked_times=[tab_src_timestartobs[obsid>1e10], tab_src_timeendobs[obsid>1e10]]

                # We then treat the classical, non-stacked Swift detections
                flux = flux[obsid < 1e10]
                flux_error = [flux_error[0][obsid < 1e10], flux_error[1][obsid < 1e10]]
                time = time[np.where(obsid < 1e10)]
                band_flux = band_flux[obsid < 1e10]
                band_flux_err = [band_flux_err[0][obsid < 1e10], band_flux_err[1][obsid < 1e10]]
                obsid = obsid[obsid < 1e10]

            band_data = catalog_class.MasterSourceClass.BandFlux(flux=band_flux, flux_err=band_flux_err)
            swift_data = catalog_class.MasterSourceClass.SwiftData(stacked_flux=swift_stacked_flux, stacked_flux_err=swift_stacked_flux_err, stacked_times=swift_stacked_times)
            source = catalog_class.MasterSourceClass.Source(catalog=cat, iau_name=name[0].strip(), flux=flux, flux_err=flux_error, time_steps=time, 
                            band_flux_data=band_data, obsids=[...], swift_data=swift_data, xmm_offaxis=[], short_term_var=[])
            # **kwargs : obsids, swift_data, xmm_offaxis, short_term_var
            
            dict_sources[name[0].strip()] = source
            
    return dict_sources


def load_master_sources(file_to_load: str) -> Dict:
    """
    Loads multi-instrument source data from a master source file, integrating data from various catalogs.

    Args:
    file_to_load (str): Directory path where the master source file and related catalog files are located.

    The function reads a master source cone FITS file and integrates it with relevant catalog data. 
    It combines data from multiple catalogs to create a comprehensive view of each master source.

    Returns:
    Dict: A dictionary of master sources, where each key is a master source ID and its value is an object 
          representing the combined data from various catalogs.

    Note:
    - The function assumes the existence of catalog-specific FITS files in the same directory as the master source file.
    - Any errors during data loading are handled and reported.
    """
    
    """Loads the multi-instruments sources in a dictionary"""
    print(f"Loading Master Sources...")
    path_file_to_load = os.path.join(file_to_load, 'Master_source_cone.fits').replace("\\", "/")
    try:
        with fits.open(path_file_to_load, memmap=True) as raw_data:
            sources_raw = Table(raw_data[1].data)
    except Exception as error:
        print(f"An error occured : {colored(error, 'magenta')}")
        
    tab_catalog_sources = {}
    for cat in dict_cat.catalogs:
        catalog_path = os.path.join(file_to_load, cat+'.fits').replace("\\", "/")
        try:
            tab_catalog_sources[cat] = load_relevant_sources(cat, catalog_path)
        except Exception as error:
            print(f"No sources detected in {cat} catalog !")
        
    dict_master_sources = {}
    for line in tqdm(sources_raw):
        tab_sources_for_this_ms = []
        for cat in dict_cat.catalogs:
            try:
                if line[cat] != '':
                    name = line[cat].strip()
                    if name in tab_catalog_sources[cat].keys():
                        tab_sources_for_this_ms.append(tab_catalog_sources[cat][name])
            except Exception as error:
                pass
        ms_id = line["MS_ID"]
        ms = catalog_class.MasterSourceClass.MasterSource(ms_id, tab_sources_for_this_ms, line["MS_RA"], line["MS_DEC"], line["MS_POSERR"])
        dict_master_sources[ms_id] = ms
        
    print("Master sources loaded!")
    return dict_master_sources


def master_source_plot(master_sources: Dict, simulation_data: Dict, number_graph: int) -> None:
    """
    Generates and saves plots for multi-instrument sources based on catalog data.

    Args:
    master_sources (Dict): A dictionary of multi-instrument sources to be plotted.
    simulation_data (Dict): A dictionary containing simulation data including object positions.
    number_graph (int): The number of graphs to generate from the master sources.

    The function iterates over a specified number of master sources and generates a plot for each. 
    It combines data from different catalogs and plots energy bands and fluxes.

    Note:
    - The function is designed to handle various catalogs and adjusts the plot according to the specific data available for each source.
    - The plots are saved in a specified directory, and their names are indexed.
    """

    object_data = simulation_data["object_data"]
    plot_var_sources_path = simulation_data["os_dictionary"]["plot_var_sources_path"]
    
    count = 0
    
    for multi_instrument_source in list(master_sources.values())[:number_graph]:
        #Each multi_instrument_source is an object with the underlying catalog sources associated with it

        #Here we compute the NICER off-axis angle between the source and the pointing
        source_coords = SkyCoord(multi_instrument_source.ra*u.degree, multi_instrument_source.dec*u.degree, frame="icrs")
        off_axis = object_data["object_position"].separation(source_coords)

        # plt.figure(figsize=(15, 8))
        figure, axes = plt.subplots(1, 1, figsize=(15, 8))
        for catalog in multi_instrument_source.sources.keys():
            #If a given catalog is contained in this source, it will be in the "sources" dictionary, catalog as key,
            #source object as value
            catalog_source = multi_instrument_source.sources[catalog]
            tab_width = 2 * np.array(dict_cat.dictionary_catalog[catalog]["energy_band_half_width"])
            for band_det in range(len(catalog_source.band_flux)):
                #The band fluxes are stored in catalog_source.band_flux. They're in erg/s/cm2, so divide by tab_width to
                #be in erg/s/cm2/keV. Here I plot them, but you can do whatever you want with those
                axes.step(dict_cat.band_edges[catalog], 
                        [catalog_source.band_flux[band_det][0] / tab_width[0]] 
                        + list(catalog_source.band_flux[band_det] / tab_width),
                        c=dict_cat.colors[catalog], where='pre')
                axes.errorbar(dict_cat.dictionary_catalog[catalog]["energy_band_center"], catalog_source.band_flux[band_det] / tab_width,
                              yerr=[catalog_source.band_flux_err[0][band_det] / tab_width,
                                    catalog_source.band_flux_err[1][band_det] / tab_width],
                              fmt="o", markeredgecolor='gray', c=dict_cat.colors[catalog], alpha=0.4)
            axes.step([], [], c=dict_cat.colors[catalog], label=f"{catalog_source.iau_name}, {catalog}")
        axes.set_xlabel("Energy [keV]")
        axes.set_ylabel(r"$F_{\nu}$ [$\mathrm{erg.s}^{-1}.\mathrm{cm}^{-2}.\mathrm{keV}^{-1}$]")
        axes.legend()
        axes.loglog()
        
        img_path = os.path.join(plot_var_sources_path, f'sources_plot_{count}.png')
        plt.savefig(img_path)
        plt.close()
        
        count += 1
