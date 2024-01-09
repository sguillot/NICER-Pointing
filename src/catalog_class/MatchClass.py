# --------------- Packages --------------- #

from astropy.table import Table, Column
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.optimize import curve_fit
from typing import Dict, Tuple, List
from termcolor import colored
from tqdm import tqdm

# ---------- import function ---------- #

import function.init_function as i_f
import function.calculation_function as c_f
import function.software_function as s_f
import function.unique_function as u_f

# ------------------------------------- #

import catalog_information as dict_cat
import sys
import numpy as np
import matplotlib.pyplot as plt
import pyvo as vo
import subprocess
import os
import platform
import shlex

# ---------------------------------------- #

class MatchCatalog:
    """
    A class to match and analyze astronomical sources from different catalogs.

    This class matches sources from two astronomical catalogs based on their 
    proximity and performs various analyses including calculating photon indices, 
    modeling source spectra, and creating source tables. It is designed to handle 
    complex astronomical data sets and prepare them for further scientific analysis.

    Attributes:
        nearby_sources_table_1 (Table): Nearby sources table from the first catalog.
        nearby_sources_table_2 (Table): Nearby sources table from the second catalog.
        nearby_sources_position_1 (SkyCoord): Positions of nearby sources from the first catalog.
        nearby_sources_position_2 (SkyCoord): Positions of nearby sources from the second catalog.
        mixed_index (List): List of mixed indexes where sources from both catalogs are matched.
        coordinates (List): List of coordinates of matched sources.
        photon_index_list (List): List of photon indices for sources.
        flag (List): Flags to identify the source of data.
        nh_list (List): List of hydrogen column densities.
        model_dictionary (Dict): Dictionary of models for sources.
        nearby_sources_table (Table): Combined nearby sources table.
        nearby_sources_position (SkyCoord): Positions of sources in the combined table.
        count_rate (List): List of count rates for sources.
        vignetting_factor (List): Vignetting factors for sources.
        master_source_cone (Table): Table of master sources within a cone search.
        var_index (List): List of variable indexes in the nearby sources table.

    Methods:
        __init__(self, catalog_name, radius, simulation_data): Initializes the CatalogMatch object.
        load_catalog(...): Loads the catalogs and returns their tables.
        load_cs_catalog(...): Loads the cone search catalog.
        unique_sources_table(...): Creates a table of unique sources.
        find_nearby_sources(...): Finds nearby sources within a specified radius.
        get_mixed_coordinate(...): Gets coordinates of mixed sources from both catalogs.
        neighbourhood_of_object(...): Plots the neighborhood of an object.
        get_photon_index(...): Calculates the photon index for a source.
        get_mixed_photon_index(...): Calculates the photon index for a mixed source.
        get_total_photon_nh_list(...): Returns total photon indices and hydrogen column densities.
        model_dictionary(...): Creates a dictionary of models for sources.
        create_nearby_sources_table(...): Creates a table of nearby sources.
        get_sources_position(...): Gets positions of sources in the nearby sources table.
        count_rate_SNR_map(...): Calculates count rates and generates a signal-to-noise ratio map.
        vignetting_factor(...): Calculates vignetting factors for sources.
        write_fits_table(...): Writes the nearby sources table to a FITS file.
        load_master_source_cone(...): Loads the master source cone.
        cross_table_index(...): Finds cross table indexes in the nearby sources table.

    The class is essential for analyzing astronomical data from multiple catalogs and preparing
    it for scientific research.
    """

    def __init__(self, catalog_name: Tuple[str, str], radius, simulation_data: Dict) -> None:
        """
        Initializes the CatalogMatch class with specific catalog names, a search radius, 
        and simulation data. The class is designed to match and analyze data from two different 
        astronomical catalogs.

        Args:
        catalog_name (Tuple[str, str]): A tuple containing the names of the two catalogs to be matched.
        radius (float): The search radius for finding nearby sources, typically specified in arcminutes.
        simulation_data (Dict): A dictionary containing various simulation parameters and data paths.

        This method loads the specified catalogs, finds nearby sources within the given radius, 
        calculates photon indices and hydrogen column densities, models the sources, and prepares 
        the data for further analysis.

        """
        
        table_1, table_2 = self.load_catalog(catalog_name=catalog_name, os_dictionary=simulation_data["os_dictionary"])

        self.nearby_sources_table_1, self.nearby_sources_table_2, self.nearby_sources_position_1, self.nearby_sources_position_2 = self.find_nearby_sources(radius=radius, simulation_data=simulation_data, table=(table_1, table_2))
        self.mixed_index, self.coordinates = self.neighbourhood_of_object(simulation_data=simulation_data, radius=radius)
        self.photon_index_list, self.flag, self.nh_list = self.get_total_photon_nh_list(os_dictionary=simulation_data["os_dictionary"])
        self.model_dictionary = self.model_dictionary()
        self.nearby_sources_table = self.create_nearby_sources_table()
        self.nearby_sources_position = self.get_sources_position()
        self.count_rate = self.count_rate_SNR_map(simulation_data=simulation_data, radius=radius)
        self.vignetting_factor = self.vignetting_factor(OptimalPointingIdx=self.OptimalPointingIdx, vector_dictionary=self.vector_dictionary, simulation_data=simulation_data)
        self.write_fits_table(os_dictionary=simulation_data["os_dictionary"])
        
        self.master_source_cone = self.load_master_source_cone(radius=radius.value, simulation_data=simulation_data)
        self.var_index = self.cross_table_index()
        

    def load_catalog(self, catalog_name: Tuple[str, str], os_dictionary: Dict) -> Tuple[Table, Table]:
        """
        Loads two specified astronomical catalogs for further analysis and matching.

        Args:
        catalog_name (Tuple[str, str]): A tuple containing the names of the two catalogs to be loaded.
        os_dictionary (Dict): A dictionary containing the operating system information, 
                            including the data path for the catalogs.

        Returns:
        Tuple[Table, Table]: A tuple of astropy Table objects corresponding to the loaded catalogs.

        This method attempts to open FITS files for the specified catalogs. If successful, 
        it sets various class attributes for future data processing. 

        Raises:
        SystemExit: If the file paths are invalid or an error occurs during file loading.
        """
        
        name_1, name_2 = catalog_name
        catalog_datapath = os_dictionary["catalog_datapath"]
        
        if name_1 == "Xmm_DR13" and name_2 == "Chandra":
            try:
                path_1 = os.path.join(catalog_datapath, "4XMM_slim_DR13cat_v1.0.fits").replace("\\", "/")
                path_2 = os.path.join(catalog_datapath, "Chandra.fits").replace("\\", "/")                    
                if os.path.exists(path=path_1) and os.path.exists(path=path_2):
                    self.catalog_key = ["XMM", "Chandra"]
                    self.data_to_vignetting = ["RA", "DEC", "Xmm_IAUNAME", "Chandra_IAUNAME"]
                    self.column_name = {"XMM": {"right_ascension": "SC_RA",
                                           "declination": "SC_DEC"},
                                        "Chandra": {"right_ascension": "RA",
                                               "declination": "DEC"}
                                        }
                    with fits.open(path_1, memmap=True) as data_1, fits.open(path_2, memmap=True) as data_2:
                        print(f"{colored('Xmm_DR13 and Chandra catalog are loaded !', 'green')}")
                        table_1, table_2 = Table(data_1[1].data), Table(data_2[1].data)
                        return table_1, table_2
                else:
                    print(f"{colored('An error occured : ', 'red')} invalid path !")
                    sys.exit()
                    
            except Exception as error:
                print(f"{colored('An error occured : ', 'red')} {error}")
                sys.exit()
                
                
    def load_cs_catalog(self, radius: float, object_data: Dict) -> Table:
        """
        Loads a cone search catalog based on the provided object name and search radius.

        Args:
        radius (float): The radius for the cone search, typically specified in arcminutes.
        object_data (Dict): A dictionary containing the object's data, specifically its name.

        Returns:
        Table: An astropy Table object containing the results of the cone search.

        This method uses the VO cone search service to find astronomical objects within the specified 
        radius of the named object. The results are converted to an astropy Table for easy handling.
        """
        
        cone = vo.dal.SCSService('http://cda.cfa.harvard.edu/csc2scs/coneSearch') 
        name = SkyCoord.from_name(object_data['object_name'])
        cone_search_catalog = cone.search(pos=name, radius=radius, verbosity=3)
        print(f"{colored('Cone search catalog are loaded !', 'green')}")
        return cone_search_catalog.to_table()

    
    def unique_sources_table(self, nearby_sources_table, column_name) -> Table:
        """
        Generates a table of unique sources from a given nearby sources table.

        Args:
        nearby_sources_table (Table): The table containing sources from a particular catalog.
        column_name (dict): A dictionary specifying column names to be used, 
                            including catalog name and other relevant fields.

        Returns:
        Table: A table of unique sources, with averaged values for sources 
            that are listed multiple times in the input table.

        This method processes the nearby sources table to identify and consolidate 
        duplicate entries. It averages the flux values for duplicated sources 
        and creates a new table with unique sources.
        """
        
        key = column_name["catalog_name"]

        dict_flux_name = {"flux_obs": dict_cat.dictionary_catalog[key]["flux_obs"],
                          "flux_obs_err": dict_cat.dictionary_catalog[key]["flux_obs_err"],
                          "band_flux_obs": dict_cat.dictionary_catalog[key]["band_flux_obs"],
                          "band_flux_obs_err": dict_cat.dictionary_catalog[key]["band_flux_obs_err"]}
        
        list_flux_name = [dict_flux_name["flux_obs"], dict_flux_name["flux_obs_err"], dict_flux_name["band_flux_obs"], dict_flux_name["band_flux_obs_err"][0], dict_flux_name["band_flux_obs_err"][1]]
        
        flux_name = []
        for value in list_flux_name:
            if isinstance(value, str):
                flux_name.append(value)
            else:
                for item in value:
                    flux_name.append(item)
                    
        for flux in flux_name:
            min_value = np.nanmean(nearby_sources_table[flux])
            nan_mask = np.isnan(nearby_sources_table[flux])
            nearby_sources_table[flux][nan_mask] = min_value

        unique_sources_dict = u_f.unique_dict(nearby_sources_table[column_name["source_name"]])
        
        new_row = []
        for index, name in enumerate(nearby_sources_table[column_name["source_name"]]):
            if name not in unique_sources_dict.keys():
                new_row.append((name, index))
            
        sources_dict = u_f.insert_row(unique_sources_dict=unique_sources_dict, new_row=new_row)
        
        
        if key == "Chandra":
            
            iauname_col, ra_col, dec_col = [], [], []
            for key, value in list(sources_dict.items()):
                iauname_col.append(key)
                ra_col.append(np.mean([nearby_sources_table[column_name["right_ascension"]][index] for index in value]))
                dec_col.append(np.mean([nearby_sources_table[column_name["declination"]][index] for index in value]))
            
            unique_table = Table()
            unique_table["Chandra_IAUNAME"] = iauname_col
            unique_table["RA"] = ra_col
            unique_table["DEC"] = dec_col
            
            for flux in flux_name:
                data = []
                for value in list(sources_dict.values()):
                    if len(value) != 1:
                        new_value = np.mean([nearby_sources_table[flux][index] for index in value])
                    else:
                        new_value = nearby_sources_table[flux][value[0]]
                    data.append(new_value)
                unique_table[flux] = data
                    
        return unique_table
    
    
    def find_nearby_sources(self, radius: float, simulation_data: Dict, table: Tuple[Table, Table]) -> Tuple[Table, Table]:
        """
        Identifies nearby sources from astronomical catalogs based on a given radius around an object.

        Args:
        radius (float): The radius within which to search for nearby sources.
        simulation_data (Dict): A dictionary containing the simulation data, including the object's information.
        table (Tuple[Table, Table]): A tuple containing two tables from different catalogs to search within.

        Returns:
        Tuple[Table, Table]: A tuple of tables containing the nearby sources from each catalog.

        This method filters the sources in the provided catalog tables to find those 
        that are within a specified radius from the object's position. It creates new 
        tables containing these nearby sources.
        """
        
        object_data = simulation_data["object_data"]
        pointing_area = radius + 5*u.arcmin
        name = object_data['object_name']
        object_position = object_data['object_position']
        min_ra, max_ra = object_position.ra - pointing_area, object_position.ra + pointing_area
        min_dec, max_dec = object_position.dec - pointing_area, object_position.dec + pointing_area
        
        if self.catalog_key == ["XMM", "Chandra"]:
            table_1, table_2 = table
            small_table_1 = Table(names=table_1.colnames,
                                  dtype=table_1.dtype)
            nearby_sources_table_1 = Table(names=table_1.colnames,
                                           dtype=table_1.dtype)
            
            print(fr"{colored(f'Reducing {self.catalog_key[0]} catalog...', 'yellow')} to fov of {radius.value + 5 } arcmin")
            for number in tqdm(range(len(table_1))):
                if min_ra/u.deg < table_1['SC_RA'][number] < max_ra/u.deg and min_dec/u.deg < table_1['SC_DEC'][number] < max_dec/u.deg:
                    small_table_1.add_row(table_1[number])

            sources_position_1 = SkyCoord(ra=small_table_1["SC_RA"], dec=small_table_1["SC_DEC"], unit=u.deg)
            sources_number_1 = len(small_table_1)
            print(f"{colored(f'Find sources close to {name} with {self.catalog_key[0]} catalog', 'blue')}")
            for number in tqdm(range(sources_number_1)):
                if c_f.ang_separation(object_position, sources_position_1[number]) < radius:
                    nearby_sources_table_1.add_row(small_table_1[number])         
                    
            small_table_2 = Table(names=table_2.colnames,
                                  dtype=table_2.dtype)
            nearby_sources_table_2 = Table(names=table_2.colnames,
                                           dtype=table_2.dtype)
            
            print(f"\n{colored(f'Reducing {self.catalog_key[1]} catalog...', 'yellow')} to fov of {radius.value + 5 } arcmin")
            for number in tqdm(range(len(table_2))):
                if min_ra/u.deg < table_2['RA'][number] < max_ra/u.deg and min_dec/u.deg < table_2['DEC'][number] < max_dec/u.deg:
                    small_table_2.add_row(table_2[number])
                
            sources_position_2 = SkyCoord(ra=small_table_2["RA"], dec=small_table_2["DEC"], unit=u.deg)
            sources_number_2 = len(small_table_2)
            print(f"{colored(f'Find sources close to {name} with {self.catalog_key[1]} catalog', 'blue')}")
            for number in tqdm(range(sources_number_2)):
                if c_f.ang_separation(object_position, sources_position_2[number]) < radius:
                    nearby_sources_table_2.add_row(small_table_2[number])
                    
            column_name = {"source_name": "Chandra_IAUNAME",
                           "right_ascension": "RA",
                           "declination": "DEC",
                           "catalog_name": "Chandra"}
                        
            unique_table = u_f.create_unique_sources_catalog(nearby_sources_table=nearby_sources_table_2, column_name=column_name)
            
            os_dictionary = simulation_data["os_dictionary"]
            catalog = os_dictionary["cloesest_dataset_path"]
            chandra_catalog = os.path.join(catalog, "unique_table.fits").replace("\\", "/")
            
            unique_table.write(chandra_catalog, format='fits', overwrite=True)
            topcat_path = os.path.join(os_dictionary["active_workflow"], 'softwares/topcat-extra.jar').replace("\\", "/")
            command = f"java -jar {topcat_path} {chandra_catalog}"
            subprocess.run(command)
        
            unique_table = u_f.create_unique_sources_catalog(nearby_sources_table=nearby_sources_table_2, column_name=column_name)
            sources_position_2 = SkyCoord(ra=unique_table['RA'], dec=unique_table['DEC'], unit=u.deg)
            print("\n")
            print(f"We have detected {colored(str(len(nearby_sources_table_1)) + ' sources', 'yellow')} with {colored(self.catalog_key[0] + ' catalog', 'yellow')} and",
                f"{colored(str(len(unique_table)) + ' sources', 'magenta')} with {colored(self.catalog_key[1] + ' catalog', 'magenta')} close to {colored(name, 'blue')}.")

            
            return nearby_sources_table_1, unique_table, sources_position_1, sources_position_2    
    
    
    def get_mixed_coordinate(self, catalog_key: Tuple[str, str], table: Tuple[Table, Table]) -> Tuple[List, List]:
        """
        Identifies and averages coordinates of overlapping sources between two astronomical catalogs.

        Args:
        catalog_key (Tuple[str, str]): A tuple containing the names of the two catalogs.
        table (Tuple[Table, Table]): A tuple of tables from the two catalogs.

        Returns:
        Tuple[List, List]: A tuple containing the mixed coordinates and indices of sources found in both catalogs.

        This method compares the coordinates of sources in both catalogs. If sources are within a certain 
        threshold distance, it averages their coordinates to create a 'mixed' coordinate. It also tracks 
        the indices of these sources in both catalogs for further analysis.

        """
        
        key_1, key_2 = catalog_key
        table_1, table_2 = table
        
        ra_1_list, dec_1_list = list(table_1[self.column_name[key_1]["right_ascension"]]), list(table_1[self.column_name[key_1]["declination"]])
        ra_2_list, dec_2_list = list(table_2[self.column_name[key_2]["right_ascension"]]), list(table_2[self.column_name[key_2]["declination"]])
        sources_1 = SkyCoord(ra=ra_1_list, dec=dec_1_list, unit=u.deg)
        sources_2 = SkyCoord(ra=ra_2_list, dec=dec_2_list, unit=u.deg)

        mixed_ra, mixed_dec = [], []
        ra_1, dec_1 = [], []
        
        index_2 = []
        mixed_index = []

        for index_1, src_1 in enumerate(sources_1):
            distance = []
            for src_2 in sources_2:
                distance.append(src_1.separation(src_2).arcmin)
            min_distance = np.min(distance)
            
            if min_distance < 5e-2:
                min_arg = np.argmin(distance)
                mixed_ra.append(np.mean([table_1[self.column_name[key_1]["right_ascension"]][index_1], table_2[self.column_name[key_2]["right_ascension"]][min_arg]]))
                mixed_dec.append(np.mean([table_1[self.column_name[key_1]["declination"]][index_1], table_2[self.column_name[key_2]["declination"]][min_arg]]))
                index_2.append(min_arg)
                mixed_index.append((index_1, min_arg))
            else:
                ra_1.append(table_1[self.column_name[key_1]["right_ascension"]][index_1])
                dec_1.append(table_1[self.column_name[key_1]["declination"]][index_1])
                
        ra_2, dec_2 = [], []
        for index, ra in enumerate(ra_2_list):
            if index not in index_2:
                ra_2.append(ra)
        for index, dec in enumerate(dec_2_list):
            if index not in index_2:
                dec_2.append(dec)

        coordinates = [(ra_1, dec_1), (ra_2, dec_2), (mixed_ra, mixed_dec)]
        return coordinates, mixed_index
    
    
    def neighbourhood_of_object(self, simulation_data: Dict, radius: float) -> Tuple[List, List]:
        """
        Visualizes the neighborhood of an astronomical object within a given radius, using data from multiple catalogs.

        Args:
        simulation_data (Dict): Dictionary containing simulation data including the object's information.
        radius (float): Radius within which to identify neighboring sources.

        Returns:
        Tuple[List, List]: A tuple containing indices of mixed sources and their coordinates.

        This method plots the neighborhood of the specified object and shows the distribution of sources 
        from different catalogs. It also identifies and marks sources that are common between catalogs.

        """
        
        object_data = simulation_data["object_data"]
        os_dictionary = simulation_data["os_dictionary"]
        table = (self.nearby_sources_table_1, self.nearby_sources_table_2)
        coordinates, mixed_index = self.get_mixed_coordinate(catalog_key=self.catalog_key, table=table)
        
        psr_ra, psr_dec = object_data["object_position"].ra, object_data["object_position"].dec
        
        figure, axes = plt.subplots(1, 1, figsize=(15, 8))
        figure.suptitle(f"Neighbourhood of {object_data['object_name']}, radius = {radius}", fontsize=20)
        figure.text(0.5, 0.04, 'Right Ascension [deg]', ha='center', va='center', fontsize=16)
        figure.text(0.04, 0.5, 'Declination [deg]', ha='center', va='center', rotation='vertical', fontsize=16)
        axes.scatter(coordinates[0][0], coordinates[0][1], edgecolors='black', facecolors='none', label=f'xmm : {len(coordinates[0][0])}')
        axes.scatter(coordinates[1][0], coordinates[1][1], edgecolors='green', facecolors='none', label=f"csc : {len(coordinates[1][0])}")
        axes.scatter(coordinates[2][0], coordinates[2][1], edgecolors='magenta', facecolors='none', label=f'xmm and csc : {len(coordinates[2][0])}')
        axes.scatter(psr_ra, psr_dec, marker="*", s=100, color="red", label=f"{object_data['object_name']}")
        axes.legend(loc="upper right", ncol=2)
        
        img = os_dictionary["img"]
        figure_path = os.path.join(img, f"neighbourhood_of_{object_data['object_name']}.png").replace("\\", "/")
        plt.savefig(figure_path)
        plt.show()
        
        return mixed_index, coordinates
    
    
    def get_photon_index(self, catalog_key: Tuple[str, str], table: Table, index: int, os_dictionary: Dict) -> float:
        """
        Calculates the photon index for a source from a specific catalog based on its spectral data.

        Args:
        catalog_key (Tuple[str, str]): A tuple representing the name of the catalog.
        table (Table): The table containing the source data from the catalog.
        index (int): The index of the source in the table.
        os_dictionary (Dict): Dictionary containing various paths and settings used in the simulation.

        Returns:
        float: The calculated photon index for the source.

        This method processes the spectral data of a source and fits it to a model to determine the photon index. 
        It supports both non-absorbed and absorbed models for fitting. For visualization purposes, it plots the 
        interpolation and fitting results for the first source in the dataset.

        """
        
        band_flux_name = dict_cat.dictionary_catalog[catalog_key]["band_flux_obs"]
        band_flux = [table[name][index] for name in band_flux_name]
        
        band_flux_obs_err_name = dict_cat.dictionary_catalog[catalog_key]["band_flux_obs_err"]
        band_flux_obs_err = [np.mean([table[err_0][index], table[err_1][index]]) for err_0, err_1 in zip(band_flux_obs_err_name[0], band_flux_obs_err_name[1])]
        
        energy_band = dict_cat.dictionary_catalog[catalog_key]["energy_band_center"]
        energy_band_half_width = dict_cat.dictionary_catalog[catalog_key]["energy_band_half_width"]
        tab_width = 2 * energy_band_half_width
        
        y_array = [num/det for num, det in zip(band_flux, tab_width)]
        yerr_array = [num/det for num, det in zip(band_flux_obs_err, tab_width)]
        
        def model(energy, constant, photon_index):
            return constant * energy ** (-photon_index)
        
        def absorb_model(energy, constant, photon_index):
            sigma = np.linspace(1e-20, 1e-24, len(energy))
            return (constant * energy ** (-photon_index))*np.exp(-sigma*3e20)
        
        model_popt, model_pcov = curve_fit(model, energy_band, y_array, sigma=yerr_array)
        m_constant, m_photon_index = model_popt
        absorb_model_popt, absorb_model_pcov = curve_fit(absorb_model, energy_band, y_array, sigma=yerr_array)
        am_constant, am_photon_index = absorb_model_popt
        
        if index == 0:
            figure, axes = plt.subplots(1, 1, figsize=(8, 5))
            figure.suptitle("Interpolation vizualisation")
            axes.scatter(energy_band, y_array, marker="+")
            axes.errorbar(energy_band, absorb_model(energy_band, *absorb_model_popt), yerr=yerr_array, fmt='*', color='red', ecolor='black')
            axes.plot(energy_band, model(energy_band, *model_popt), label=f"Non absorb $\Gamma : {m_photon_index:.4f}$")
            axes.plot(energy_band, absorb_model(energy_band, *absorb_model_popt), label=f"Absorb $\Gamma : {am_photon_index:.4f}$")
            axes.loglog()
            axes.set_xlabel("Energy [$keV$]")
            axes.set_ylabel("flux $[erg.cm^{-2}.s^{-1}.keV^{-1}]$")
            axes.legend(loc="upper right")
            
            img = os_dictionary["img"]
            figure_path = os.path.join(img, f"Interpolation_vizualisation_{catalog_key}_.png").replace("\\", "/")
            plt.savefig(figure_path)
            plt.show()
            
        return am_photon_index
           
            
    def get_mixed_photon_index(self, catalog_key: Tuple[str, str], table: Tuple[Table, Table], mixed_index: List[Tuple], row: int, os_dictionary: Dict) -> float:
        """
        Determines the photon index for a source that appears in both catalogs, using combined spectral data.

        Args:
        catalog_key (Tuple[str, str]): A tuple containing the names of the two catalogs.
        table (Tuple[Table, Table]): A tuple of tables from the two catalogs.
        mixed_index (List[Tuple]): List of tuples with indices of sources appearing in both catalogs.
        row (int): The row number in the mixed_index list for which the photon index is to be calculated.
        os_dictionary (Dict): Dictionary containing various paths and settings used in the simulation.

        Returns:
        float: The calculated photon index for the mixed source.

        This method merges the spectral data from two different catalogs for the same source and calculates the photon index. 
        It supports fitting to both non-absorbed and absorbed models. A visualization of the interpolation and fitting 
        is provided for a specific case (row = 10).

        """
        
        key_1, key_2 = catalog_key
        table_1, table_2 = table[0], table[1]
        
        energy_band_1 = dict_cat.dictionary_catalog[key_1]["energy_band_center"]
        energy_band_2 = dict_cat.dictionary_catalog[key_2]["energy_band_center"]
        energy_band = sorted(energy_band_1 + energy_band_2)
        
        energy_band_half_width_1 = dict_cat.dictionary_catalog[key_1]["energy_band_half_width"]
        energy_band_half_width_2 = dict_cat.dictionary_catalog[key_2]["energy_band_half_width"]
        energy_band_half_width = sorted(energy_band_half_width_1 + energy_band_half_width_2)
        tab_width = 2 * energy_band_half_width
        
        band_flux_name_1 = dict_cat.dictionary_catalog[key_1]["band_flux_obs"]
        band_flux_name_2 = dict_cat.dictionary_catalog[key_2]["band_flux_obs"]
        
        band_flux_1 = [table_1[name][mixed_index[row][0]] for name in band_flux_name_1]
        band_flux_2 = [table_2[name][mixed_index[row][1]] for name in band_flux_name_2]
        
        value = 0
        for index, item in enumerate(energy_band):
            if item in energy_band_2:
                band_flux_1.insert(index, band_flux_2[value])
                value += 1

        y_array = [num/det for num, det in zip(band_flux_1, tab_width)]
        
        band_flux_obs_err_name_1 = dict_cat.dictionary_catalog[key_1]["band_flux_obs_err"]
        band_flux_obs_err_name_2 = dict_cat.dictionary_catalog[key_2]["band_flux_obs_err"]

        band_flux_obs_err_1 = [np.mean([table_1[err_0][row], table_1[err_1][row]]) for err_0, err_1 in zip(band_flux_obs_err_name_1[0], band_flux_obs_err_name_1[1])]
        band_flux_obs_err_2 = [np.mean([table_2[err_0][row], table_2[err_1][row]]) for err_0, err_1 in zip(band_flux_obs_err_name_2[0], band_flux_obs_err_name_2[1])]

        value = 0
        for index, item in enumerate(energy_band):
            if item in energy_band_2:
                band_flux_obs_err_1.insert(index, band_flux_obs_err_2[value])
                value += 1

        yerr_array = [num/det for num, det in zip(band_flux_obs_err_1, tab_width)]

        def model(energy, constant, photon_index):
            return constant * energy ** (-photon_index)

        def absorb_model(energy, constant, photon_index):
            sigma = np.linspace(1e-20, 1e-24, len(energy))
            return (constant * energy ** (-photon_index))*np.exp(-sigma*3e20)

        model_popt, model_pcov = curve_fit(model, energy_band, y_array, sigma=yerr_array)
        m_constant, m_photon_index = model_popt
        absorb_model_popt, absorb_model_pcov = curve_fit(absorb_model, energy_band, y_array, sigma=yerr_array)
        am_constant, am_photon_index = absorb_model_popt

        if row == 10:
            figure, axes = plt.subplots(1, 1, figsize=(8, 5))
            figure.suptitle("Interpolation vizualisation")
            axes.scatter(energy_band, y_array, marker="+")
            axes.errorbar(energy_band, absorb_model(energy_band, *absorb_model_popt), yerr=yerr_array, fmt='*', color='red', ecolor='black')
            axes.plot(energy_band, model(energy_band, *model_popt), label=f"Non absorb $\Gamma : {m_photon_index:.4f}$")
            axes.plot(energy_band, absorb_model(energy_band, *absorb_model_popt), label=f"Absorb $\Gamma : {am_photon_index:.4f}$")
            axes.loglog()
            axes.set_xlabel("Energy [$keV$]")
            axes.set_ylabel("flux $[erg.cm^{-2}.s^{-1}.keV^{-1}]$")
            axes.legend(loc="upper right")
            
            img = os_dictionary["img"]
            figure_path = os.path.join(img, f"Interpolation_vizualisation_{key_1}_{key_2}_.png").replace("\\", "/")
            plt.savefig(figure_path)
            plt.show()
            
        return am_photon_index
    
    
    def get_total_photon_nh_list(self, os_dictionary: Dict) -> Tuple[List[float], List[Tuple], List[float]]:
        """
        Compiles a list of photon indices, source flags, and column densities for sources in the catalog.

        Args:
        os_dictionary (Dict): Dictionary containing various paths and settings used in the simulation.

        Returns:
        Tuple[List[float], List[Tuple], List[float]]: A tuple containing three lists - photon indices, flags, and column densities.

        This method iterates over the sources in the provided catalogs (XMM and Chandra), calculates the photon index 
        for each source, and assigns a flag indicating the catalog(s) to which the source belongs. It also appends 
        a default column density value for each source. The method handles sources unique to one catalog and those 
        appearing in both.

        """
        
        key_1, key_2 = self.catalog_key
        photon_index_list, nh_list = [], []
        flag = [] 
        if key_1 == "XMM" and key_2 == "Chandra":
            
            row = 0  
            index_1 = [self.mixed_index[index][0] for index in range(len(self.mixed_index))]
            index_2 = [self.mixed_index[index][1] for index in range(len(self.mixed_index))]

            number_source_1 = len(self.nearby_sources_table_1)
            for index in range(number_source_1):
                if index not in index_1:
                    photon_index_list.append(self.get_photon_index(catalog_key=key_1, table=self.nearby_sources_table_1, index=index,os_dictionary=os_dictionary))
                    flag.append(('xmm', index))
                    nh_list.append(3e20)
                else:
                    keys = (key_1, key_2)
                    tables = (self.nearby_sources_table_1, self.nearby_sources_table_2)
                    photon_index_list.append(self.get_mixed_photon_index(catalog_key=keys, table=tables, mixed_index=self.mixed_index, row=row, os_dictionary=os_dictionary))
                    flag.append(('both', self.mixed_index[row]))
                    nh_list.append(3e20)
                    row += 1
                    
            number_source_2 = len(self.nearby_sources_table_2)
            for index in range(number_source_2):
                if index not in index_2:
                    photon_index_list.append(self.get_photon_index(catalog_key=key_2, table=self.nearby_sources_table_2, index=index, os_dictionary=os_dictionary))
                    flag.append(("chandra", index))
                    nh_list.append(3e20)
            
        return photon_index_list, flag, nh_list
    
    
    def model_dictionary(self) -> Dict:
        """
        Creates a dictionary mapping each source to its spectral model parameters.

        Returns:
        Dict: A dictionary where each key represents a source and its value is a dictionary of model parameters.

        The method iterates over all sources, categorizing them based on their catalog (XMM, Chandra, or both). 
        It then constructs a model dictionary for each source, specifying the model type ('power'), the photon index, 
        observed flux, and a default column density. The model parameters are derived from the catalog data and 
        previously calculated photon indices.

        """
     
        model_dictionary = {}
        
        for item, flag in enumerate(self.flag):
                        
            if flag[0] == "xmm":
                key = self.catalog_key[0]
                flux_obs = dict_cat.dictionary_catalog[key]["flux_obs"]
                flux = self.nearby_sources_table_1[flux_obs][flag[1]]
            elif flag[0] == "both":
                key_1, key_2 = self.catalog_key
                flux_obs_1, flux_obs_2 = dict_cat.dictionary_catalog[key_1]["flux_obs"], dict_cat.dictionary_catalog[key_2]["flux_obs"]
                flux_1, flux_2 = self.nearby_sources_table_1[flux_obs_1][flag[1][0]], self.nearby_sources_table_2[flux_obs_2][flag[1][1]]
                flux = np.mean([flux_1, flux_2])
            else:
                key = self.catalog_key[1]
                flux_obs = dict_cat.dictionary_catalog[key]["flux_obs"]
                flux = self.nearby_sources_table_2[flux_obs][flag[1]]
            
            dictionary = {
                "model": 'power',
                "model_value": self.photon_index_list[item],
                "flux": flux,
                "column_dentsity": 3e20
            }
            
            model_dictionary[f"src_{item}"] = dictionary
            
        return model_dictionary


    def create_nearby_sources_table(self) -> Table:
        """
        Constructs a table of nearby sources with their respective details from XMM and Chandra catalogs.

        Returns:
        astropy.table.Table: A table containing the combined data of nearby sources from both catalogs.

        This method processes the sources identified as nearby in either the XMM or Chandra catalogs, or both.
        It creates a table with columns for flags indicating the source's catalog, names from both catalogs, 
        coordinates (RA, DEC), and flux observations. The method also appends photon index and column density (Nh) 
        data to each source. The table facilitates easy reference to the properties of all nearby sources.

        """
                
        key_1, key_2 = self.catalog_key
        row = 0
        flux_obs_1, flux_obs_2 = dict_cat.dictionary_catalog[key_1]["flux_obs"], dict_cat.dictionary_catalog[key_2]["flux_obs"]
        column_names = ["Flag","Xmm_IAUNAME", "Chandra_IAUNAME", "RA", "DEC", flux_obs_1, flux_obs_2]
        dtype = [str, str, str, float, float, float, float]
        nearby_sources_table = Table(names=column_names, dtype=dtype)
        
        for flag in self.flag:
            
            if flag[0] == "xmm":
                flag_value = flag[0]
                x_name = self.nearby_sources_table_1["IAUNAME"][flag[1]]
                c_name = ""
                ra_value = self.nearby_sources_table_1["SC_RA"][flag[1]]
                dec_value = self.nearby_sources_table_1["SC_DEC"][flag[1]]
                flux_1 = self.nearby_sources_table_1[flux_obs_1][flag[1]]
                flux_2 = np.nan
            elif flag[0] == "both":
                flag_value = flag[0]
                x_name = self.nearby_sources_table_1["IAUNAME"][flag[1][0]]
                c_name = self.nearby_sources_table_2["Chandra_IAUNAME"][flag[1][1]]
                ra_value = self.coordinates[2][0][row]
                dec_value = self.coordinates[2][1][row]
                flux_1 = self.nearby_sources_table_1[flux_obs_1][flag[1][0]]
                flux_2 = self.nearby_sources_table_2[flux_obs_2][flag[1][1]]
                row += 1
            else:
                flag_value = flag[0]
                x_name = ""
                c_name = self.nearby_sources_table_2["Chandra_IAUNAME"][flag[1]]
                ra_value = self.nearby_sources_table_2["RA"][flag[1]]
                dec_value = self.nearby_sources_table_2["DEC"][flag[1]]
                flux_1 = np.nan
                flux_2 = self.nearby_sources_table_2[flux_obs_2][flag[1]]
                
            add_row = [flag_value, x_name, c_name, ra_value, dec_value, flux_1, flux_2]
            nearby_sources_table.add_row(add_row)
            
        photon_index_column = Column(name="Photon Index", data=self.photon_index_list)
        nearby_sources_table.add_column(photon_index_column)
        nh_column = Column(name="Nh", data=self.nh_list)
        nearby_sources_table.add_column(nh_column)
        
        return nearby_sources_table
    
    
    def get_sources_position(self) -> SkyCoord:
        """
        Retrieves the positions of all nearby sources as SkyCoord objects.

        Returns:
        astropy.coordinates.SkyCoord: SkyCoord object containing the RA and DEC of each nearby source.

        This method extracts the right ascension (RA) and declination (DEC) of each nearby source from the 
        compiled table and converts them into a SkyCoord object for convenient handling of celestial coordinates.
        It is particularly useful for astronomical calculations and visualizations that require positional data.

        """
        
        ra_value = list(self.nearby_sources_table["RA"])
        dec_value = list(self.nearby_sources_table["DEC"])
        
        return SkyCoord(ra=ra_value, dec=dec_value, unit=u.deg)
    
    
    def count_rate_SNR_map(self, simulation_data: Dict, radius: float) -> List[float]:
        """
        Calculates the count rates for nearby sources and generates a Signal-to-Noise Ratio (SNR) map.

        Args:
        simulation_data (Dict): A dictionary containing simulation data including telescope and object data.
        radius (float): The radius within which to consider nearby sources.

        Returns:
        List[float]: A list of count rates for each nearby source.

        This method computes count rates for sources within a specified radius from the object of interest.
        It generates count rates based on the sources' table, the model dictionary, and telescope data. 
        The method also involves various stages of data processing and visualization, including calculating 
        optimal pointing information, creating a SNR map, and updating the nearby sources table with 
        relevant data. It is specific to astronomy and astrophysics simulations where understanding the 
        surrounding field of sources is crucial.

        """
        
        telescop_data = simulation_data["telescop_data"]
        object_data = simulation_data["object_data"]
        data_path = simulation_data["os_dictionary"]["data_path"]
        excel_data_path = os.path.join(data_path, 'excel_data').replace("\\", "/")
        
        key = "xmmXchandra"
        catalog ="match"
        
        if not os.path.exists(excel_data_path):
            os.mkdir(excel_data_path)
            
        if platform.system() == "Linux" or platform.system() == "Darwin":
            count_rates, self.nearby_sources_table = c_f.count_rates(self.nearby_sources_table, self.model_dictionary, telescop_data)
            # i_f.py_to_xlsx(excel_data_path=excel_data_path, count_rates=count_rates, object_data=object_data, args=(catalog, key), radius=radius)
        elif platform.system() == "Windows":
            count_rates, self.nearby_sources_table = i_f.xlsx_to_py(excel_data_path=excel_data_path, nearby_sources_table=self.nearby_sources_table, object_data=object_data, args=(catalog, key), radius=radius.value)
        else:
            sys.exit()
            
        simulation_data['nearby_sources_table'] = self.nearby_sources_table
        
        c_f.nominal_pointing_info(simulation_data, self.nearby_sources_position)
        self.OptimalPointingIdx, self.SRCoptimalSEPAR, self.SRCoptimalRATES, self.vector_dictionary = c_f.calculate_opti_point(simulation_data, self.nearby_sources_position)
        c_f.optimal_point_infos(self.vector_dictionary, self.OptimalPointingIdx, self.SRCoptimalRATES)
        c_f.data_map(simulation_data, self.vector_dictionary, self.OptimalPointingIdx, self.nearby_sources_position)
        
        return count_rates


    def vignetting_factor(self, OptimalPointingIdx: int, vector_dictionary: Dict, simulation_data: Dict) -> List[float]:
        """
        Calculates the vignetting factor for each nearby source based on the optimal pointing index.

        Args:
        OptimalPointingIdx (int): The index of the optimal pointing position.
        vector_dictionary (Dict): A dictionary containing vectors for calculation.
        simulation_data (Dict): A dictionary containing simulation data.

        Returns:
        List[float]: A list of vignetting factors for each nearby source.

        This method evaluates the vignetting factor, a measure of the decrease in telescope sensitivity 
        with increasing off-axis angles, for each nearby source. It uses the optimal pointing index to 
        determine the minimum angular distance of each source from the optimal pointing position and 
        calculates the corresponding vignetting factor. The method is essential in astrophysical 
        observations to correct for instrumental effects on observed fluxes.
        
        """
        
        ra, dec, name_1, name_2 = self.data_to_vignetting
        
        object_data = simulation_data["object_data"]
        EffArea, OffAxisAngle = simulation_data["telescop_data"]["EffArea"], simulation_data["telescop_data"]["OffAxisAngle"]
        
        optipoint_ra, optipoint_dec = vector_dictionary['SampleRA'][OptimalPointingIdx], vector_dictionary['SampleDEC'][OptimalPointingIdx]
    
        def calculate_vignetting_factor(D, effareaX, effareaY):
            return np.interp(D,effareaX,effareaY)
        
        vignetting_factor, distance = np.array([], dtype=float), np.array([], dtype=float)
    
        total_source = len(self.nearby_sources_table)
        for index in range(total_source):
            SRCposition  = SkyCoord(ra=self.nearby_sources_table[ra][index]*u.degree, dec=self.nearby_sources_table[dec][index]*u.degree)
            SRCnominalDIST = c_f.ang_separation(SRCposition, SkyCoord(ra=optipoint_ra, dec=optipoint_dec, unit=u.deg)).arcmin
            distance = np.append(distance, SRCnominalDIST)
            vignetting = calculate_vignetting_factor(SRCnominalDIST, EffArea, OffAxisAngle)
            vignetting_factor = np.append(vignetting_factor, vignetting)
    
        optimal_pointing_point = SkyCoord(ra=optipoint_ra, dec=optipoint_dec, unit=u.deg)
        psr_position = SkyCoord(ra=object_data['object_position'].ra, dec=object_data['object_position'].dec, unit=u.deg)
        distance_psr_to_optipoint = c_f.ang_separation(psr_position, optimal_pointing_point).arcmin
        vignetting_factor_psr2optipoint = calculate_vignetting_factor(distance_psr_to_optipoint, EffArea, OffAxisAngle)
    
        max_vignet, min_distance  = np.max(vignetting_factor), np.min(distance)
        max_vignet_index, min_distance_index = np.argmax(vignetting_factor), np.argmin(distance)
        
        
        print(f"The distance between {colored(object_data['object_name'], 'yellow')} and optimal pointing point is {colored(distance_psr_to_optipoint, 'blue')} arcmin,\n"
              f"with a vagnetting factor of : {colored(vignetting_factor_psr2optipoint, 'light_green')}")
        
        flag = self.nearby_sources_table["Flag"][max_vignet_index]
        if flag == "both":
            name_1, name_2 = self.nearby_sources_table[name_1][max_vignet_index], self.nearby_sources_table[name_2][max_vignet_index]
            key_1, key_2 = self.catalog_key
            print(f"The closest source of the optimal pointing point is : ")
            print(f"{colored(name_1, 'yellow')} in {key_1} and {colored(name_2, 'yellow')} in {key_2}.")
            print(f"The distance between this source and the optimal pointing point is : {colored(min_distance, 'blue')} arcmin for a vignetting factor of {colored(max_vignet, 'light_green')}")
        elif flag == "xmm":
            name_1 = self.nearby_sources_table[name_1][max_vignet_index]
            key_1 = self.catalog_key[0]
            print(f"The closest source of the optimal pointing point is : ")
            print(f"{colored(name_1, 'yellow')} in {key_1}.")
            print(f"The distance between this source and the optimal pointing point is : {colored(min_distance, 'blue')} arcmin for a vignetting factor of {colored(max_vignet, 'light_green')}")
        else:
            name_2 = self.nearby_sources_table[name_2][max_vignet_index]
            key_2 = self.catalog_key[1]
            print(f"The closest source of the optimal pointing point is : ")
            print(f"{colored(name_2, 'yellow')} in {key_2}.")
            print(f"The distance between this source and the optimal pointing point is : {colored(min_distance, 'blue')} arcmin for a vignetting factor of {colored(max_vignet, 'light_green')}")

        self.nearby_sources_table["vignetting_factor"] = vignetting_factor
        
        return vignetting_factor
    
    
    def write_fits_table(self, os_dictionary: Dict) -> None:
        """
        Writes the nearby sources table to a FITS file and opens it using TOPCAT software.

        Args:
        os_dictionary (Dict): A dictionary containing operating system paths and other related information.

        This method saves the nearby sources table as a FITS file to the path specified in the 
        'cloesest_dataset_path' key of the os_dictionary. It then attempts to open this file using 
        TOPCAT software for further analysis and visualization. This function is essential for astronomers 
        and astrophysicists who need to store and analyze large sets of astronomical data.

        Note:
        - The method assumes the existence of the TOPCAT software in the active workflow directory.
        - Errors during the process are caught and printed, but do not halt the execution of the program.

        """
        
        try:
            cloesest_dataset_path = os_dictionary["cloesest_dataset_path"]
            nearby_sources_table_path = os.path.join(cloesest_dataset_path, "nearby_sources_table.fits").replace("\\", "/")
            self.nearby_sources_table.write(nearby_sources_table_path, format='fits', overwrite=True)
            print(f"Nearby sources table was created in : {colored(nearby_sources_table_path, 'magenta')}")
            
            topcat_path = os.path.join(os_dictionary["active_workflow"], 'softwares/topcat-extra.jar').replace("\\", "/")
            command = f"java -jar {topcat_path} {nearby_sources_table_path}"
            subprocess.run(command)
            
        except Exception as error:
            print(f"{colored('An error occured : ', 'red')} {error}")
            
            
    def load_master_source_cone(self, radius: float, simulation_data: Dict) -> Table:
        """
        Loads and processes a master source cone file within a given radius for a specific object.

        Args:
        radius (float): The search radius in arcminutes.
        simulation_data (Dict): A dictionary containing essential simulation data and paths.

        Returns:
        Table: An astropy table containing the processed master source cone data.

        This method processes a master source FITS file to select and match sources around a 
        specified region based on the object's position in the simulation data. It uses various 
        software tools like STILTS and TOPCAT for data processing and visualization. The method is 
        crucial for isolating and analyzing celestial objects within a specific region in 
        astrophysical studies.

        Note:
        - The method assumes the presence of necessary software tools in the specified paths.
        - It handles exceptions during processing and displays relevant error messages.
        
        """
        
        catalogs = dict_cat.catalogs
        object_data = simulation_data["object_data"]
        os_dictionary = simulation_data["os_dictionary"]
        
        stilts_software_path = os_dictionary["stilts_software_path"]
        topcat_software_path = os_dictionary["topcat_software_path"]
        catalog_datapath = os_dictionary["catalog_datapath"]
        output_name = os_dictionary["output_name"]
        
        master_source_path = os.path.join(catalog_datapath, 'Master_source.fits').replace("\\", "/")

        def select_master_sources_around_region(ra, dec, radius, output_name):
            """Radius is in arcminutes"""
            print(f"Extracting sources around region: RA {ra} and Dec {dec}")
            master_cone_path = os.path.join(output_name, 'Master_source_cone.fits').replace("\\", "/")
            command = (f"java -jar {stilts_software_path} tpipe {master_source_path} cmd='"+
                    f'select skyDistanceDegrees({ra},{dec},MS_RA,MS_DEC)*60<{radius} '+
                    f"' out={master_cone_path}")
            command = shlex.split(command)
            subprocess.run(command)


        def select_catalogsources_around_region(output_name):
            print('Selecting catalog sources')
            master_cone_path = os.path.join(output_name, 'Master_source_cone.fits').replace("\\", "/")
            for cat in catalogs:
                path_to_cat_init = os.path.join(catalog_datapath, cat).replace("\\", "/")
                path_to_cat_final = os.path.join(output_name, cat).replace("\\", "/")
                command = (f"java -jar {stilts_software_path} tmatch2 matcher=exact \
                        in1='{master_cone_path}' in2='{path_to_cat_init}.fits' out='{path_to_cat_final}.fits'\
                            values1='{cat}' values2='{cat}_IAUNAME' find=all progress=none")
                command = shlex.split(command)
                subprocess.run(command)

        right_ascension = object_data["object_position"].ra.value
        declination = object_data["object_position"].dec.value
        try:
            print(f"\n{colored('Load Erwan s code for :', 'yellow')} {object_data['object_name']}")
            select_master_sources_around_region(ra=right_ascension, dec=declination, radius=radius, output_name=output_name)
            select_catalogsources_around_region(output_name=output_name)
            master_sources = s_f.load_master_sources(output_name)
            s_f.master_source_plot(master_sources=master_sources, simulation_data=simulation_data, number_graph=len(master_sources))
        except Exception as error :
            print(f"{colored('An error occured : ', 'red')} {error}")
            
        path = os.path.join(output_name, "Master_source_cone.fits").replace("\\", "/")
        
        command = f"java -jar {topcat_software_path} {path}"
        subprocess.run(command)
        
        with fits.open(path, memmap=True) as data:
            master_source_cone = Table(data[1].data)
            
        return master_source_cone
    

    def cross_table_index(self) -> List:
        """
        Generates a list of indices where sources from the master source cone match those in the nearby sources table.

        Returns:
        List: A list of indices representing the matching sources across the two tables.

        This method compares the master source cone table with the nearby sources table to identify common sources. 
        It specifically looks for matches in the 'Xmm_IAUNAME' and 'Chandra_IAUNAME' columns of the nearby sources table 
        based on the names listed in the master source cone table. The method is essential for astrophysical research 
        where identifying overlapping observations from different catalogs is necessary.

        Note:
        - The method relies on the catalog key to determine the columns for comparison.
        - The result is a sorted list of unique indices, representing the intersection of the two tables.

        """
        
        key_0, key_1 = self.catalog_key
        msc_xmm_name = [name for name in self.master_source_cone[key_0] if name != ""]
        msc_csc_name = [name for name in self.master_source_cone[key_1] if name != ""]
        
        xmmXchandra_xmm_index = []
        for name in msc_xmm_name:
            if name in self.nearby_sources_table["Xmm_IAUNAME"]:
                index_in_table = list(self.nearby_sources_table["Xmm_IAUNAME"]).index(name)
                xmmXchandra_xmm_index.append(index_in_table)
                
        xmmXchandra_csc_index = []
        for name in msc_csc_name:
            if name in self.nearby_sources_table["Chandra_IAUNAME"]:
                index_in_table = list(self.nearby_sources_table["Chandra_IAUNAME"]).index(name)
                xmmXchandra_csc_index.append(index_in_table)
        
        var_index_in_nearby_sources_table = sorted(set(xmmXchandra_xmm_index + xmmXchandra_csc_index))

        return var_index_in_nearby_sources_table
   
