# --------------- Packages --------------- #

from astropy.table import Table, Column
from astropy.io import fits
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from astropy.units import Quantity
from scipy.optimize import curve_fit
from astroquery.esasky import ESASky
from astropy.visualization import PercentileInterval, ImageNormalize, LinearStretch
from astropy.wcs import WCS
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
from typing import Dict, Tuple, Union, List
from termcolor import colored
from tqdm import tqdm
from jaxspec.model.multiplicative import Tbabs
from jaxspec.model.additive import Powerlaw
from jax.config import config
from jaxspec.data.instrument import Instrument
from jaxspec.data.util import fakeit_for_multiple_parameters


import function as f
import openpyxl
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

class XmmCatalog:
    """
    Class for managing XMM source catalogs.

    This class is designed for loading and processing XMM (X-ray Multi-Mirror Mission) source catalogs.
    It includes functionalities for finding nearby sources to a given position, analyzing their variability, 
    and visualizing the corresponding data.

    Attributes:
        key (str): Key identifying the XMM catalog.
        xmm_catalog (Table): Table of the opened XMM catalog.
        nearby_sources_table (Table): Table of nearby sources.
        nearby_sources_position (SkyCoord): Coordinates of the nearby sources.
        xmm_dr11_catalog (Table): Table of the XMM DR11 catalog.
        x2a_catalog (Table): Table of the Xmm2Athena catalog.
        index_table (Table): Table indexing sources in different catalogs.
        variability_table (Table): Table of sources with their variability.
        model_dictionary (dict): Dictionary of models associated with the sources.

    Args:
        catalog_path (str): Path to the XMM catalog file.
        radius (Quantity): Search radius around the object.
        simulation_data (Dict): Simulation data used for analysis.
        user_table (Table): User data table to use in the analysis.

    Methods:
        open_catalog: Opens an XMM catalog from a given path.
        find_nearby_sources: Finds sources close to a given object.
        optimization_phoindex: Optimizes the photon index for a given source.
        visualization_interp: Visualizes the interpolation of the photon index.
        empty_row: Creates an empty row for a catalog table.
        get_phoindex_nh: Retrieves the photon index and column density for nearby sources.
        variability_table: Constructs a table of nearby sources with their variability.
        neighbourhood_of_object: Displays the neighborhood of a given object.
        dictionary_model: Creates a dictionary of models for the sources.

    """

    def __init__(self, catalog_path: str, radius: Quantity, simulation_data:Dict, user_table: Table) -> None:
        """
        Initializes the XmmCatalog class.

        This constructor sets up the catalog by loading data from specified paths, finding nearby sources,
        and preparing various tables and models for analysis.

        Args:
            catalog_path (str): Path to the primary XMM catalog file. This file is used to initialize the xmm_catalog attribute.
            radius (Quantity): The search radius for finding nearby sources. This is used in conjunction with object_data from simulation_data to determine the area of interest.
            simulation_data (Dict): A dictionary containing simulation data. This includes 'object_data' used in find_nearby_sources and 'os_dictionary' for additional catalog paths.
            user_table (Table): An optional user-provided table to include in the analysis. This is used in the find_nearby_sources method to add user-specific data.

        The constructor performs several key operations:
        - It initializes the xmm_catalog attribute by opening the catalog from the given path.
        - It finds nearby sources using the find_nearby_sources method and stores the results in nearby_sources_table and nearby_sources_position.
        - It loads additional XMM catalogs (DR11 and Xmm2Athena) from paths specified in the simulation_data and initializes xmm_dr11_catalog and x2a_catalog.
        - It calculates photon indices and column densities for nearby sources using the get_phoindex_nh method.
        - It builds a variability table for the nearby sources and visualizes the neighborhood of the object of interest.

        Attributes Initialized:
            key (str): A key for identifying the catalog type, set to "XMM".
            ra (str) : right ascension name in XMM catalog.
            dec (str) : declination name in XMM catalog.
            xmm_catalog (Table): The main catalog data loaded from catalog_path.
            nearby_sources_table (Table): A table of sources found near the object of interest.
            nearby_sources_position (SkyCoord): The positions of nearby sources.
            xmm_dr11_catalog (Table): The XMM DR11 catalog data.
            x2a_catalog (Table): The Xmm2Athena catalog data.
            index_table (Table): An indexing table for sources across different catalogs.
            variability_table (Table): A table capturing the variability of sources.
            model_dictionary (Dict): A dictionary of models derived from the analysis.
        """
        self.key = "XMM"
        # ---------- coord ---------- #
        self.ra = dict_cat.dictionary_coord[self.key]["right_ascension"]
        self.dec = dict_cat.dictionary_coord[self.key]["declination"]
        # -------------------------- #
        
        self.xmm_catalog = self.open_catalog(catalog_path=catalog_path)
        self.nearby_sources_table, self.nearby_sources_position = self.find_nearby_sources(radius=radius, object_data=simulation_data["object_data"], user_table=user_table)
        
        test_dr11_path = os.path.join(simulation_data["os_dictionary"]["catalog_datapath"], "4XMM_DR11cat_v1.0.fits").replace("\\", "/")
        test_x2a_path = os.path.join(simulation_data["os_dictionary"]["catalog_datapath"], "xmm2athena_D6.1_V3.fits").replace("\\", "/")
        xmm_dr11_path = f.get_valid_file_path(test_dr11_path)
        x2a_path = f.get_valid_file_path(test_x2a_path)
        
        self.xmm_dr11_catalog = self.open_catalog(catalog_path=xmm_dr11_path)
        self.x2a_catalog = self.open_catalog(catalog_path=x2a_path)
        self.nearby_sources_table, self.index_table = self.get_phoindex_nh()
        self.variability_table = self.variability_table(object_data=simulation_data["object_data"])
        self.neighbourhood_of_object(radius=radius, simulation_data=simulation_data)
        self.model_dictionary = self.dictionary_model()
        
    
    def open_catalog(self, catalog_path: str) -> Table:
        """
        Opens an XMM catalog from a specified file path and returns it as an Astropy Table.

        This method is designed to open FITS files which contain astronomical data, typically used in X-ray astronomy. 
        It utilizes the 'fits' module to read the file and convert the data into a format that is easily manageable 
        and usable for further analysis.

        Args:
            catalog_path (str): The file path to the XMM catalog FITS file. This file is expected to be in a 
                                format compatible with the FITS standard, commonly used in astrophysics and astronomy.

        Returns:
            Table: An Astropy Table object containing the data from the FITS file. The table structure allows for 
                easy manipulation and analysis of the catalog data, leveraging the capabilities of the Astropy package.

        """
        with fits.open(catalog_path, memmap=True) as data:
            return Table(data[1].data)
        
    
    def find_nearby_sources(self, radius: Quantity, object_data: dict, user_table: Table) -> Tuple[Table, SkyCoord]:
        """
        Searches for sources within a specified radius around a given astronomical object and returns a table of these sources along with their coordinates.

        This method filters the XMM catalog to find sources that are within a certain radius of the specified object's position. 
        It can incorporate additional user-provided data for a more tailored search.

        Args:
            radius (Quantity): The search radius around the object of interest. This radius is used to define a circular region in the sky within which the sources will be searched for.
            object_data (dict): A dictionary containing data about the object of interest. It should include keys 'object_position' (SkyCoord) and 'object_name' (str), representing the astronomical coordinates and the name of the object, respectively.
            user_table (Table): An optional Astropy Table containing user-defined data. If provided, this table is used in addition to the XMM catalog for finding nearby sources.

        Returns:
            Tuple[Table, SkyCoord]: A tuple containing two elements:
                - An Astropy Table of sources found within the specified radius. The table includes various details of these sources as defined in the XMM catalog.
                - A SkyCoord object representing the coordinates of the found sources.

        The method involves the following steps:
        - Calculating the minimum and maximum right ascension (RA) and declination (Dec) based on the object's position and the specified radius.
        - Filtering the XMM catalog to create a smaller table of sources within this RA and Dec range.
        - Further filtering these sources based on the angular separation from the object's position to ensure they are within the specified radius.
        - Handling user-provided data if available, to include additional sources in the search.
        - Returning the final list of nearby sources and their positions.
        
        """
        object_position = object_data['object_position']
        object_name = object_data['object_name']
        
        pointing_area = radius + 5*u.arcmin
        min_ra, max_ra = object_position.ra - pointing_area, object_position.ra + pointing_area
        min_dec, max_dec = object_position.dec - pointing_area, object_position.dec + pointing_area
        
        small_table = Table(names=self.xmm_catalog.colnames,
                            dtype=self.xmm_catalog.dtype)
        nearby_src_table = Table(names=self.xmm_catalog.colnames,
                                 dtype=self.xmm_catalog.dtype)
        
        print(fr"{colored('Reducing XMM catalog...', 'yellow')} to fov of {radius.value + 5 } arcmin")
        for number in tqdm(range(len(self.xmm_catalog))):
            if min_ra/u.deg < self.xmm_catalog[self.ra][number] < max_ra/u.deg and min_dec/u.deg < self.xmm_catalog[self.dec][number] < max_dec/u.deg:
                small_table.add_row(self.xmm_catalog[number])
                
        if len(user_table) == 0:
            src_position = SkyCoord(ra=small_table[self.ra], dec=small_table[self.dec], unit=u.deg)
            print(f"{colored(f'Find sources close to {object_name} with XMM catalog', 'blue')}")
            for number in tqdm(range(len(small_table))):
                if f.ang_separation(object_position, src_position[number]) < radius:
                    nearby_src_table.add_row(small_table[number])
            nearby_src_position = SkyCoord(ra=nearby_src_table[self.ra], dec=nearby_src_table[self.dec], unit=u.deg)

        else:
            for number in range(len(user_table)):
                small_table.add_row(user_table[number])
                
            src_position = SkyCoord(ra=small_table[self.ra], dec=small_table[self.dec], unit=u.deg)
            print(f"{colored(f'Find sources close to {object_name} with XMM catalog', 'blue')}")
            for number in tqdm(range(len(small_table))):
                if f.ang_separation(object_position, src_position[number]) < radius:
                    nearby_src_table.add_row(small_table[number])
            nearby_src_position = SkyCoord(ra=nearby_src_table[self.ra], dec=nearby_src_table[self.dec], unit=u.deg)
                
        try:
            if len(nearby_src_table) != 0:
                print((f"We have detected {len(nearby_src_table)} sources close to {object_name}.\n"))
                return nearby_src_table, nearby_src_position
            else:
                print(f"No sources detected close to {object_name}.")
                sys.exit()
        except Exception as error:
            print(f"An error occured : {error}")
            
       
    def optimization_phoindex(self, number: int) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[float, float]]:
        """
        Optimizes the photon index for a specific source in the catalog using an absorbed power law model.

        This method applies curve fitting to observed flux data to determine the best-fit parameters of an absorbed power law model. 
        It is primarily used to analyze the energy distribution of a source and estimate its photon index, a key parameter in astrophysical analysis.

        Args:
            number (int): The index of the source in the nearby_sources_table for which the photon index is to be optimized.

        Returns:
            Tuple[Tuple[np.ndarray, ...], Tuple[float, float]]: A tuple containing two elements:
                - A tuple of numpy arrays representing the energy band, observed flux (y_array), flux error (yerr_array), 
                and the fitted power law values for the given energy band.
                - A tuple containing the optimized constant and photon index values from the power law fit.

        The method involves:
        - Retrieving the observed flux and its error for the given source based on predefined band names.
        - Normalizing the flux values based on the energy band width.
        - Using the `curve_fit` function from `scipy.optimize` to fit the absorbed power law model to the observed data.
        - Returning the optimized parameters along with the energy band and normalized flux values.

        Note:
            The method assumes specific keys in 'dict_cat.dictionary_catalog' for retrieving energy band and flux information.
            The absorbed power law model is defined internally within the method.
        """    
        
        def absorbed_power_law(x, constant, gamma):
            sigma = np.array([1e-20, 5e-21, 1e-22, 1e-23, 1e-24], dtype=float)
            return (constant * x ** (-gamma)) * (np.exp(-sigma * 3e20))
        

        energy_band = dict_cat.dictionary_catalog[self.key]["energy_band_center"]
        energy_band_half_width = dict_cat.dictionary_catalog[self.key]["energy_band_half_width"]
        tab_width = 2 * energy_band_half_width
        
        band_flux_obs_name = dict_cat.dictionary_catalog[self.key]["band_flux_obs"]
        band_flux_obs_err_name = dict_cat.dictionary_catalog[self.key]["band_flux_obs_err"]
        
        flux_obs = [self.nearby_sources_table[name][number] for name in band_flux_obs_name]
        flux_err = [[self.nearby_sources_table[err_0][number] for err_0 in band_flux_obs_err_name[0]],
                    [self.nearby_sources_table[err_1][number] for err_1 in band_flux_obs_err_name[1]]]

        flux_err_obs = [np.mean([flux_err_0, flux_err_1]) for flux_err_0, flux_err_1 in zip(flux_err[0], flux_err[1])]
        
        y_array = [num/det for num, det in zip(flux_obs, tab_width)]
        yerr_array = [num/det for num, det in zip(flux_err_obs, tab_width)]
        
        popt, pcov = curve_fit(absorbed_power_law, energy_band, y_array, sigma=yerr_array)
        constant, absorb_pho_index = popt

        optimization_parameters = (energy_band, y_array, yerr_array, absorbed_power_law(energy_band, *popt))
        
        return optimization_parameters, absorb_pho_index


    def visualization_interp(self, optimization_parameters, photon_index) -> None:
        """
        Visualizes the results of the photon index interpolation for each source.

        This method creates a plot for each source showing the observed flux versus the energy band, along with the fitted power law model. 
        It's useful for visually assessing the quality of the fit and understanding the energy distribution of the sources.

        Args:
            optimization_parameters (tuple): A tuple containing the energy band, observed flux, flux error, and the fitted power law values for each source. These parameters are obtained from the optimization_phoindex method.
            photon_index (array): An array of photon index values for each source. These are the optimized photon index values obtained from the absorbed power law fit.

        The method performs the following steps:
        - For each set of optimization parameters, it plots the observed flux (with errors) against the energy band.
        - It also plots the corresponding absorbed power law model, using the optimized photon index for each source.
        - The plots are annotated with the photon index value for each source.
        - The plot uses a logarithmic scale for both axes for better visualization of the power law behavior.

        Note:
            - The method assumes that the energy band data is provided in keV and the flux in erg.cm^-2.s^-1.keV^-1.
            - This method does not return any value but displays the plots directly using matplotlib.
        """
        
        energy_band = dict_cat.dictionary_catalog["XMM"]["energy_band_center"]
        
        fig, axes = plt.subplots(1, 1, figsize=(15, 8))
        fig.suptitle("Interpolation Photon Index plot", fontsize=20)
        fig.text(0.5, 0.04, 'Energy [keV]', ha='center', va='center')
        fig.text(0.04, 0.5, 'Flux $[erg.cm^{-2}.s^{-1}.keV^{-1}]$', ha='center', va='center', rotation='vertical')
        
        for item in range(len(optimization_parameters)):
            flux_obs = optimization_parameters[item][1]
            flux_obs_err = optimization_parameters[item][2]
            absorbed_power_law = optimization_parameters[item][3]
            absorb_pho_index = photon_index[item]
            
            axes.errorbar(energy_band, flux_obs, flux_obs_err, fmt='*', color='red', ecolor='black')
            axes.plot(energy_band, absorbed_power_law, label=f"$\Gamma$ = {absorb_pho_index:.8f}")
    
        axes.legend(loc="upper left", ncol=4, fontsize=6)
        axes.loglog()
        
        plt.show()
            

    def empty_row(self, catalog):
        new_row = {}
        for nom, type in catalog.dtype:
            if 'i' in type or 'u' in type:  # Types entiers
                new_row[nom] = 0
            elif 'f' in type:  # Types flottants
                new_row[nom] = np.nan
            elif 'U' in type:  # Types chaîne de caractères
                new_row[nom] = ''
            elif '?' in type:  # Type booléen
                new_row[nom] = False
            else:
                new_row[nom] = None
        return new_row


    def get_phoindex_nh(self) -> Tuple[Table, Table]:
        """
        Retrieves and calculates the photon index and hydrogen column density (Nh) for each source in the nearby_sources_table.

        This method performs a series of operations to determine the photon index and Nh values for sources 
        based on data from the XMM DR11 and Xmm2Athena catalogs. It also involves optimizing the photon index 
        for sources not found in the Xmm2Athena catalog using an absorbed power law model.

        Returns:
            Tuple[Table, Table]: A tuple of two Astropy Tables:
                - The first table is the nearby_sources_table, updated with new columns for Photon Index and Nh.
                - The second table is an index table that maps each source to its corresponding index in the xmm_dr11 and x2a catalogs.

        The method includes the following steps:
        - Compiling an index table that correlates the sources in the nearby_sources_table with their respective indices in the xmm_dr11 and x2a catalogs.
        - Calculating the Nh value for each source. If the source is found in the x2a catalog, the Nh value is taken from there; otherwise, a default value is used.
        - Optimizing the photon index for each source using the optimization_phoindex method. For sources in the x2a catalog, the photon index is taken from there.
        - Updating the nearby_sources_table with the calculated Photon Index and Nh values.
        - Visualizing the photon index interpolation results for sources not found in the x2a catalog.

        Note:
            - The method assumes specific keys and structures in the xmm_dr11 and x2a catalogs for data extraction.
            - It prints messages to the console regarding missing data in the xmm_dr11 catalog and visualizes the interpolation results.
        """
    
        number_source = len(self.nearby_sources_table)
        name_list = self.nearby_sources_table["IAUNAME"]
        xmm_dr_11_table = Table(names=self.xmm_dr11_catalog.colnames,
                                dtype=self.xmm_dr11_catalog.dtype)
        index_dr11 = np.array([], dtype=int)
        for name in name_list:
            if name in self.xmm_dr11_catalog['IAUNAME']:
                index = list(self.xmm_dr11_catalog['IAUNAME']).index(name)
                index_dr11 = np.append(index_dr11, index)
                xmm_dr_11_table.add_row(self.xmm_dr11_catalog[index])
            else:
                print(f"{colored('Missing data in Xmm_DR11 : ', 'red')} {name}")
                index_dr11 = np.append(index_dr11, np.nan)
                xmm_dr_11_table.add_row(self.empty_row([]))
                

        index_x2a = np.array([], dtype=int)
        for det_id in xmm_dr_11_table["DETID"]:
            if det_id in self.x2a_catalog["DETID"]: 
                index = list(self.x2a_catalog["DETID"]).index(det_id)
                index_x2a = np.append(index_x2a, index)
            else:
                index_x2a = np.append(index_x2a, "No data found")

        column_names = ["Index in nearby_sources_table", "Index in xmm_dr11", "Index in x2a"]
        column_data = [[item for item in range(number_source)], index_dr11, index_x2a]
        index_table = Table(names=column_names,
                            data=column_data)
        
        column_nh, column_phoindex = np.array([], dtype=float), np.array([], dtype=float)
        optimization_parameters, photon_index = [], []
        
        for number in range(number_source):
            if index_table["Index in x2a"][number] != "No data found":
                nh_value = self.x2a_catalog["logNH_med"][number]
                column_nh = np.append(column_nh, np.exp(nh_value * np.log(10)))
                column_phoindex = np.append(column_phoindex, self.x2a_catalog['PhoIndex_med'][number])
            else:
                column_nh = np.append(column_nh, 3e20)
                parameters, pho_value = self.optimization_phoindex(number)
                optimization_parameters.append(parameters)
                photon_index.append(pho_value)
                column_phoindex = np.append(column_phoindex, pho_value)

        self.visualization_interp(optimization_parameters=optimization_parameters, photon_index=photon_index)
        
        col_names = ["Photon Index", "Nh"]
        col_data = [column_phoindex, column_nh]
        
        for name, data in zip(col_names, col_data):
            self.nearby_sources_table[name] = data
        
        return self.nearby_sources_table, index_table
    
    
    def variability_table(self, object_data: dict) -> Table:
        """
        Creates a table summarizing the variability of sources near a specified astronomical object.

        This method assesses the variability of each source in the nearby_sources_table using data from the xmm_dr11_catalog. 
        It compiles a new table that includes details about the variability of these sources, along with their inclusion in the Xmm2Athena catalog.

        Args:
            object_data (dict): A dictionary containing data about the object of interest, including its name.

        Returns:
            Table: An Astropy Table containing information on the variability of each source. The table includes the source's index, 
                IAU name, coordinates (RA and Dec), variability factor (SC_FVAR), and a boolean indicating if the source is included 
                in the Xmm2Athena catalog.

        The method performs the following operations:
        - Iterates through the nearby_sources_table to check for each source's variability factor from the xmm_dr11_catalog.
        - Constructs a new table with relevant information for sources that have a defined variability factor.
        - The table is populated with indexes, names, coordinates, variability factors, and Xmm2Athena catalog inclusion status.
        - Prints a summary message to the console about the number of variable sources and their distribution in the Xmm2Athena catalog.

        Note:
            - The method assumes the presence of specific keys in 'dict_cat.dictionary_coord' for coordinate comparison and in 'object_data' for the object's name.
            - It prints summary messages to the console for informative purposes.
        """
    
        object_name = object_data["object_name"]

        index_array, iauname_array, sc_ra_array = np.array([], dtype=int), np.array([], dtype=str), np.array([], dtype=float)
        sc_dec_array, sc_fvar_array, in_x2a_array = np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=bool)

        for number in range(len(self.nearby_sources_table)):
            if not np.isnan(self.xmm_dr11_catalog["SC_FVAR"][self.index_table["Index in xmm_dr11"][number]]):

                index_array = np.append(index_array, self.index_table["Index in nearby_sources_table"][number])
                iauname_array = np.append(iauname_array, self.nearby_sources_table["IAUNAME"][number])
                sc_ra_array = np.append(sc_ra_array, self.nearby_sources_table[self.ra][number])
                sc_dec_array = np.append(sc_dec_array, self.nearby_sources_table[self.dec][number])
                sc_fvar_array = np.append(sc_fvar_array, self.nearby_sources_table["SC_FVAR"][number])

                if self.index_table["Index in x2a"][number] != "No data found":
                    in_x2a_array = np.append(in_x2a_array, True)
                else:
                    in_x2a_array = np.append(in_x2a_array, False)

        column_names = ["INDEX", "IAUNAME", self.ra, self.dec, "SC_FVAR", "IN_X2A"]
        data_array = [index_array, iauname_array, sc_ra_array, sc_dec_array, sc_fvar_array, in_x2a_array]
        variability_table = Table()

        for data, name in zip(data_array, column_names):
            variability_table[name] = data

        message_xmm = f"Among {len(self.nearby_sources_table)} sources detected close to {object_name}, {len(index_array)} of them are variable. Using DR13 Catalog."
        print(message_xmm)
        message_xmm2ath = f"Among {len(index_array)} variable sources, {list(variability_table['IN_X2A']).count(True)} are in Xmm2Athena and {list(variability_table['IN_X2A']).count(False)} are not in Xmm2Athena. "    
        print(message_xmm2ath)

        return variability_table
    
    
    def neighbourhood_of_object(self, radius: Quantity, simulation_data: dict) -> None:
        """
        Visualizes the neighborhood of a specified astronomical object, highlighting the variable and invariable sources around it.

        This method creates a visual representation of sources around the given object. It differentiates between sources that are variable 
        and those that are not, based on their presence in the Xmm2Athena catalog, and plots their positions relative to the object.

        Args:
            radius (Quantity): The radius around the object within which to search for neighboring sources.
            simulation_data (dict): A dictionary containing simulation data, including 'object_data' with the target object's information and 'os_dictionary' for saving the plot.

        The method performs the following operations:
        - Queries the ESASky catalog for XMM-EPIC observations of the object and retrieves corresponding images if available.
        - Identifies variable and invariable sources from the nearby_sources_table and the variability_table.
        - Creates two plots: 
            - The first plot shows all sources around the object.
            - The second plot differentiates between variable sources found in Xmm2Athena and those not found.
        - Both plots include the position of the target object and are saved as images.

        Note:
            - This method attempts to query ESASky for relevant data and handles any exceptions that may occur during this process.
            - It assumes the availability of right ascension (RA) and declination (Dec) keys in 'dict_cat.dictionary_coord'.
            - The plots are saved to the directory specified in 'os_dictionary' within 'simulation_data'.
        """

        print("\n")
        object_data = simulation_data["object_data"]
        object_name = object_data['object_name']
        obj_ra, obj_dec = object_data['object_position'].ra, object_data['object_position'].dec
        try:
            result = ESASky.query_object_catalogs(position=object_name, catalogs="XMM-EPIC")
            xmm_epic = Table(result[0])
            xmm_obs_id = list(xmm_epic["observation_id"])
            result_fits_images = ESASky.get_images(observation_ids=xmm_obs_id[0], radius=radius, missions="XMM")
        except Exception as error:
            print(f"{colored('An error occured : ', 'red')} {error}")
            result_fits_images = {}
            
        
        ra_in_x2a = [ra_value for index, ra_value in enumerate(self.variability_table[self.ra]) if self.variability_table['IN_X2A'][index] == True]
        dec_in_x2a = [dec_value for index, dec_value in enumerate(self.variability_table[self.dec]) if self.variability_table['IN_X2A'][index] == True]
        ra_in_dr11 = [ra_value for index, ra_value in enumerate(self.variability_table[self.ra]) if self.variability_table['IN_X2A'][index] == False]
        dec_in_dr11 = [dec_value for index, dec_value in enumerate(self.variability_table[self.dec]) if self.variability_table['IN_X2A'][index] == False]
        invar_ra = [ra_value for ra_value in self.nearby_sources_table[self.ra] if ra_value not in self.variability_table[self.ra]]
        invar_dec = [dec_value for dec_value in self.nearby_sources_table[self.dec] if dec_value not in self.variability_table[self.dec]]
    
        figure = plt.figure(figsize=(17, 8))
        figure.suptitle(f"Neighbourhood of {object_name}", fontsize=20)
        
        if result_fits_images == {}:
            figure.text(0.5, 0.04, 'Right Ascension [deg]', ha='center', va='center')
            figure.text(0.04, 0.5, 'Declination [deg]', ha='center', va='center', rotation='vertical')
            
            axes_0 = figure.add_subplot(121)
            axes_0.invert_xaxis()
            axes_0.scatter(list(self.nearby_sources_table[self.ra]), list(self.nearby_sources_table[self.dec]), s=20, color='darkorange', label=f"Sources : {len(self.nearby_sources_table)}")
            axes_0.scatter(obj_ra, obj_dec, s=100, color='red', marker="*", label=f"{object_name}")
            axes_0.legend(loc='upper right', ncol=2, fontsize=7)
            axes_0.set_title(f"Sources close to {object_name}")
            
            axes_1 = figure.add_subplot(122)
            axes_1.invert_xaxis()
            axes_1.scatter(invar_ra, invar_dec, color='black', s=20, label=f"Invariant sources, {len(invar_ra)}")
            axes_1.scatter(obj_ra, obj_dec, color='red', marker='x', s=100, label=f"Position of {object_name}")
            axes_1.scatter(ra_in_x2a, dec_in_x2a, color='darkorange', marker="*", label=f"Variable sources in X2A, {len(ra_in_x2a)}")
            axes_1.scatter(ra_in_dr11, dec_in_dr11, color='royalblue', marker="*", label=f"Variable sources not in X2A, {len(ra_in_dr11)}")
            axes_1.legend(loc="lower right", ncol=2)
            axes_1.set_title(f"Variable and invariable sources close to {object_name} ")
            
        else:
            image = result_fits_images["XMM"][0][0].data[0, :, :]
            wcs = WCS(result_fits_images["XMM"][0][0].header)
            _wcs_ = wcs.dropaxis(2)
            
            figure.text(0.5, 0.04, 'Right Ascension [hms]', ha='center', va='center', fontsize=16)
            figure.text(0.04, 0.5, 'Declination [dms]', ha='center', va='center', rotation='vertical', fontsize=16)
            
            norm = ImageNormalize(image,interval=PercentileInterval(98.0), stretch=LinearStretch())
            
            axes_0 = figure.add_subplot(121, projection=_wcs_)
            axes_0.coords[0].set_format_unit(u.hourangle)
            axes_0.coords[1].set_format_unit(u.deg)  
            axes_0.imshow(image, cmap='gray', origin='lower', norm=norm, interpolation='nearest', aspect='equal')
            axes_0.scatter(list(self.nearby_sources_table[self.ra]), list(self.nearby_sources_table[self.dec]), s=30, transform=axes_0.get_transform('fk5'), facecolors='none', edgecolors='orange', label=f"Sources : {len(self.nearby_sources_table)}")
            axes_0.scatter(obj_ra, obj_dec, s=100, color='red', marker="*", transform=axes_0.get_transform('fk5'), facecolors='none', edgecolors='red', label=f"{object_name}")
            axes_0.legend(loc='upper right', ncol=2, fontsize=7)
            xlim, ylim = plt.xlim(), plt.ylim()
            value_x, value_y = 180, 180
            axes_0.set_xlim(xmin=xlim[0]+value_x, xmax=xlim[1]-value_x)
            axes_0.set_ylim(ymin=ylim[0]+value_y, ymax=ylim[1]-value_y)
            axes_0.set_xlabel(" ")
            axes_0.set_ylabel(" ")
            axes_0.set_title(f"Sources close to {object_name}")
            
            axes_1 = figure.add_subplot(122, projection=_wcs_, sharex=axes_0, sharey=axes_0)
            axes_1.imshow(image, cmap='gray', origin='lower', norm=norm, interpolation='nearest', aspect='equal')
            axes_1.scatter(invar_ra, invar_dec, s=30, transform=axes_1.get_transform('fk5'), facecolors='none', edgecolors='orange', label=f"Invar src : {len(invar_ra)}")
            axes_1.scatter(ra_in_dr11, dec_in_dr11, s=30, transform=axes_1.get_transform('fk5'), facecolors='none', edgecolors='blue', label=f"Var src not in x2a : {len(ra_in_dr11)} sources")
            axes_1.scatter(ra_in_x2a, dec_in_x2a, s=30, transform=axes_1.get_transform('fk5'), facecolors='none', edgecolors='hotpink', label=f"Var src in x2a : {len(ra_in_x2a)} sources")
            axes_1.scatter(obj_ra, obj_dec, s=100, marker="*", transform=axes_1.get_transform('fk5'), facecolors='none', edgecolors='red', label=f"{object_name}")
            axes_1.legend(loc='upper right', ncol=2, fontsize=7)
            axes_1.set_xlabel(" ")
            axes_1.set_ylabel(" ")
            axes_1.set_title(f"Variable and invariable sources close to {object_name} ")
            
        os_dictionary = simulation_data["os_dictionary"]
        plt.savefig(os.path.join(os_dictionary["img"], f"neighbourhood_of_{object_name}.png".replace(" ", "_")))
        plt.show()
        print("\n")


    def dictionary_model(self) -> Dict[str, Dict[str, Union[str, float]]]:
        """
        Creates a dictionary of models for each source in the nearby_sources_table, detailing the model type, its parameters, and source flux.

        This method compiles a dictionary where each entry corresponds to a source in the nearby_sources_table. 
        It includes details like the model used for analysis, the model's parameters, the observed flux, and the hydrogen column density (Nh).

        Returns:
            Dict[str, Dict[str, Union[str, float]]]: A dictionary with keys as source identifiers (e.g., 'src_0', 'src_1', etc.). 
            Each value is another dictionary containing the following keys:
                - 'model': The name of the model used for the source (e.g., 'power').
                - 'model_value': The value of a key parameter in the model (e.g., photon index for a power-law model).
                - 'flux': The observed flux of the source.
                - 'column_density': The hydrogen column density (Nh) associated with the source.

        The method performs the following operations:
        - Iterates through each source in the nearby_sources_table.
        - Determines the appropriate model and its parameters based on predefined criteria.
        - Compiles this information into a dictionary, alongside the source's observed flux and Nh value.

        Note:
            - Currently, the method only implements the 'power' model, with the photon index as the model value.
            - Future extensions may include additional models like 'black_body' or 'temp'.
            - The method assumes that flux and Nh values are available in the nearby_sources_table.
        """
        
        model_dictionary = {}
        number_source = len(self.nearby_sources_table)

        flux_obs = dict_cat.dictionary_catalog[self.key]["flux_obs"]
        
        model = np.array([], dtype=str)
        model_value = np.array([], dtype=float)
        xmm_flux = np.array([self.nearby_sources_table[flux_obs][item] for item in range(number_source)], dtype=float)
        nh_value = np.array([self.nearby_sources_table["Nh"][item] for item in range(number_source)], dtype=float)
        
        # Pour le moment seulement 'power' indiquant le modèle a utiliser pour la commande pimms
        for item in range(number_source):
            model = np.append(model, 'power')    
            
        for item in range(number_source):
            if model[item] == 'power':
                model_value = np.append(model_value, self.nearby_sources_table["Photon Index"][item])
            elif model[item] == 'black_body':
                pass # Pas de valeur pour le moment...
            elif model[item] == 'temp':
                pass # Pas de valeur pour le moment... (dernier model pimms)

        for item in range(number_source):

            dictionary = {
                "model": model[item],
                "model_value": model_value[item],
                "flux": xmm_flux[item],
                "column_dentsity": nh_value[item]
            }

            model_dictionary[f"src_{item}"] = dictionary
            
        return model_dictionary 


class Chandra:
    """
    A class for handling and analyzing data from the Chandra X-ray Observatory and associated cone search catalogs.

    This class is designed to work with Chandra X-ray data and perform various analyses including finding nearby sources, 
    assessing source variability, calculating photon indices, and generating models for each source.

    Attributes:
        ra (str): The key for right ascension in the Chandra dictionary.
        dec (str): The key for declination in the Chandra dictionary.
        ra_cs (str): The key for right ascension in the cone search Chandra dictionary.
        dec_cs (str): The key for declination in the cone search Chandra dictionary.
        chandra_catalog (Table): The catalog of Chandra data loaded from the specified path.
        cone_search_catalog (Table): The catalog of data loaded from a cone search based on the Chandra data.
        cs_nearby_sources_position (SkyCoord): The coordinates of sources found in the cone search catalog.
        nearby_sources_table (Table): A table of nearby sources found in the Chandra catalog.
        nearby_sources_position (SkyCoord): The coordinates of nearby sources found in the Chandra catalog.
        cs_photon_index (array): Photon index values for sources in the cone search catalog.
        photon_index (array): Photon index values for sources in the Chandra catalog.
        cs_model_dictionary (dict): A dictionary of models for sources in the cone search catalog.
        model_dictionary (dict): A dictionary of models for sources in the Chandra catalog.

    The class includes methods for opening and loading catalogs, finding nearby sources, analyzing source variability, 
    calculating photon indices, visualizing data, and generating source models.

    Args:
        catalog_path (str): Path to the primary Chandra catalog file.
        radius (Quantity): The search radius for finding nearby sources.
        simulation_data (Dict): A dictionary containing simulation data, including object data and OS dictionary for saving plots.
        user_table (Table): An optional user-provided table to include in the analysis.
    """
    
    def __init__(self, catalog_path: str, radius: Quantity, simulation_data: Dict, user_table: Table) -> None:
        """
        Initializes the Chandra class, setting up the necessary attributes for data analysis.

        This constructor loads the Chandra catalog from the given path and performs various initializations, including loading a cone search catalog,
        finding nearby sources, and preparing data for further analysis such as variability assessment, photon index calculation, and model generation.

        Args:
            catalog_path (str): Path to the primary Chandra catalog file. This file is used to initialize the Chandra catalog.
            radius (Quantity): The search radius for finding nearby sources. This radius is utilized for cone searches and finding nearby sources.
            simulation_data (Dict): A dictionary containing simulation data. This includes 'object_data' for object-specific information and 'os_dictionary' for operating system-specific information.
            user_table (Table): An optional Astropy Table provided by the user to include in the analysis.

        The constructor performs several key operations:
        - Initializes coordinate keys for Chandra and cone search catalogs from a global dictionary.
        - Loads the Chandra catalog from the specified path.
        - Loads a cone search catalog based on the provided radius and object data.
        - Finds nearby sources based on the Chandra catalog and additional user-provided data.
        - Prepares the data for further analysis, including calculating photon indices and generating source models.

        Attributes:
            ra (str): Right ascension key for the Chandra catalog.
            dec (str): Declination key for the Chandra catalog.
            ra_cs (str): Right ascension key for the cone search Chandra catalog.
            dec_cs (str): Declination key for the cone search Chandra catalog.
            chandra_catalog (Table): Catalog data loaded from the Chandra catalog file.
            cone_search_catalog (Table): Catalog data loaded from the cone search.
            cs_nearby_sources_position (SkyCoord): Positions of sources found in the cone search catalog.
            nearby_sources_table (Table): Table of nearby sources found in the Chandra catalog.
            nearby_sources_position (SkyCoord): Positions of nearby sources found in the Chandra catalog.
            cs_photon_index (list): List of photon index values for sources in the cone search catalog.
            photon_index (list): List of photon index values for sources in the Chandra catalog.
            cs_model_dictionary (dict): Dictionary of models for sources in the cone search catalog.
            model_dictionary (dict): Dictionary of models for sources in the Chandra catalog.
        """    
        
        # ---------- coord ---------- #
        self.ra = dict_cat.dictionary_coord["Chandra"]["right_ascension"]
        self.dec = dict_cat.dictionary_coord["Chandra"]["declination"]
        
        self.ra_cs = dict_cat.dictionary_coord["CS_Chandra"]["right_ascension"]
        self.dec_cs = dict_cat.dictionary_coord["CS_Chandra"]["declination"]
        # --------------------------- #
        
        
        self.chandra_catalog = self.open_catalog(catalog_path=catalog_path)
        self.cone_search_catalog = self.load_cs_catalog(radius=radius, object_data=simulation_data["object_data"])
        self.cs_nearby_sources_position = SkyCoord(ra=list(self.cone_search_catalog[self.ra_cs]), dec=list(self.cone_search_catalog[self.dec_cs]), unit=u.deg)
        self.cone_search_catalog = self.variability_column()
        self.cone_search_catalog = self.threshold(self.cone_search_catalog)
        
        self.nearby_sources_table, self.nearby_sources_position = self.find_nearby_sources(radius=radius, object_data=simulation_data["object_data"], user_table=user_table)
        self.neighbourhood_of_object(radius=radius, simulation_data=simulation_data)
        self.cs_photon_index, self.photon_index = self.get_phoindex_nh()
        self.cs_model_dictionary, self.model_dictionary = self.dictionary_model()
        

    def open_catalog(self, catalog_path: str) -> Table:
        """
        Opens a FITS file from the specified path and converts it into an Astropy Table.

        This method is designed to load a catalog from a FITS file, which is a common file format used in astronomy for storing data tables and images. 
        The method utilizes the 'fits' module from Astropy to read the file and loads the data into a table format for easier manipulation and analysis.

        Args:
            catalog_path (str): The file path to the FITS file containing the catalog data. This file should conform to the FITS standard and typically contains astronomical data.

        Returns:
            Table: An Astropy Table object containing the data from the FITS file. The table format facilitates various operations like data filtering, sorting, and visualization.
        """
        
        with fits.open(catalog_path, memmap=True) as data:
            return Table(data[1].data)
        
        
    def load_cs_catalog(self, radius: Quantity, object_data: dict) -> Table:
        """
        Conducts a cone search in the Chandra Source Catalog using VO SCS protocol.

        This method performs a cone search in the Chandra Source Catalog by initiating a VO SCS (Simple Cone Search) 
        service request. The search is centered on the coordinates of a celestial object, which are determined using the 
        object's name provided in `object_data`. The search radius is specified by the `radius` parameter.

        Parameters:
        radius (Quantity): The radius of the cone search. Should be an astropy Quantity object, specifying both the value 
                        and the unit (e.g., in degrees or arcminutes).
        object_data (dict): A dictionary containing information about the celestial object. This must include a key 
                            'object_name' with the object's name as its value, which is used to determine the object's sky 
                            coordinates for the search.

        Returns:
        Table: An astropy Table containing the results of the cone search. This table includes various details about the 
            astronomical sources found within the specified search radius.

        """
        
        cone = vo.dal.SCSService('http://cda.cfa.harvard.edu/csc2scs/coneSearch') 
        name = SkyCoord.from_name(object_data['object_name'])
        cone_search_catalog = cone.search(pos=name, radius=radius, verbosity=3)
        return cone_search_catalog


    def find_nearby_sources(self, radius: Quantity, object_data: dict, user_table: Table) -> Tuple[Table, SkyCoord]:
        """
        Searches for astronomical sources near a specified object within the Chandra Source Catalog.

        This method identifies sources in the Chandra Source Catalog that are located within a specified radius of a given 
        celestial object. It uses the object's position and a defined field of view to filter sources from the catalog. 
        The method returns a table of nearby sources along with their sky coordinates.

        Parameters:
        radius (Quantity): The radius within which to search for sources, as an astropy Quantity object (value and unit).
        object_data (dict): A dictionary containing the object's information. It must have keys 'object_name' and 
                            'object_position', where 'object_position' should be a SkyCoord object.
        user_table (Table): An astropy Table object representing the user's data. This table is not used in the current 
                            implementation of the function.

        Returns:
        Tuple[Table, SkyCoord]: A tuple containing two elements. The first element is an astropy Table with the catalog of 
                                nearby sources. The second element is a SkyCoord object containing the positions of these 
                                sources.

        Notes:
        - The search radius is internally expanded by 5 arcminutes to define a field of view.
        - The function prints various messages during execution, including progress updates and results summary.
        - If no sources are found, the function exits the program.
        - In case of an error, the error message is printed.

        """
    
        field_of_view = radius + 5*u.arcmin
        object_name = object_data["object_name"]
        object_position = object_data['object_position']

        min_ra, max_ra = object_position.ra - field_of_view, object_position.ra + field_of_view
        min_dec, max_dec = object_position.dec - field_of_view, object_position.dec + field_of_view

        small_table = Table(names= self.chandra_catalog.colnames,
                            dtype= self.chandra_catalog.dtype)

        nearby_sources_table = Table(names= self.chandra_catalog.colnames,
                                     dtype= self.chandra_catalog.dtype)
        
        print(fr"{colored('Reducing Chandra catalog...', 'yellow')} to fov of {radius.value + 5 } arcmin")
        for number in tqdm(range(len(self.chandra_catalog))):
            if min_ra/u.deg < self.chandra_catalog[self.ra][number] < max_ra/u.deg and min_dec/u.deg < self.chandra_catalog[self.dec][number] < max_dec/u.deg:
                small_table.add_row(self.chandra_catalog[number])
                
        sources_position = SkyCoord(ra=small_table[self.ra], dec=small_table[self.dec], unit=u.deg)

        print(f"{colored(f'Find sources close to {object_name} with Chandra catalog', 'blue')}")
        for number in tqdm(range(len(small_table))):
            if f.ang_separation(object_position, sources_position[number]) < radius:
                nearby_sources_table.add_row(small_table[number])
                
        column_name = {"source_name": "Chandra_IAUNAME", 
                       "right_ascension": self.ra,
                       "declination": self.dec,
                       "catalog_name": "Chandra"}
            
        if len(nearby_sources_table) != 0:
            unique_table = f.create_unique_sources_catalog(nearby_sources_table=nearby_sources_table, column_name=column_name)
        else:
            print("Nearby sources table from Chandra catalog is empty.")
            sys.exit()
            
        nearby_sources_position = SkyCoord(ra=unique_table[self.ra], dec=unique_table[self.dec], unit=u.deg)
                
        try :
            if len(unique_table) != 0:
                print((f"We have detected {len(unique_table)} sources close to {object_name}"))
                return unique_table, nearby_sources_position
            else:
                print(f"No sources detected close to {object_name}.")
        except Exception as error:
            print(f"An error occured : {error}")
            
    
    def variability_column(self) -> Table:
        """
        Adds a 'Variability' column to the Chandra Source Catalog table based on intra and inter variability probabilities.

        This method calculates a variability score for each source in the Chandra Source Catalog. It uses the 
        'var_inter_prob_b' and 'var_intra_prob_b' columns from the catalog to compute this score. If both intra and inter 
        variability probabilities are available for a source, their mean is used as the variability score. If only one of 
        them is available, that value is used. If both are missing, the variability score is set to 0.0.

        Returns:
        Table: The modified astropy Table of the Chandra Source Catalog, now including the 'Variability' column. This column 
            contains the computed variability scores for each source.

        Notes:
        - The method assumes that the catalog has already been converted to an astropy Table and is accessible via 
        `self.cone_search_catalog`.
        - Variability probabilities that are masked (not available) are handled as described above.
        - The method appends the 'Variability' column to the existing table and returns the updated table.

        """
        
        cone_catalog = self.cone_search_catalog.to_table()        

        inter, intra = cone_catalog['var_inter_prob_b'], cone_catalog['var_intra_prob_b']
        var_column = np.array([])

        for inter_value, intra_value in zip(inter, intra):
            if isinstance(inter_value, np.ma.core.MaskedConstant) and isinstance(intra_value, np.ma.core.MaskedConstant):
                var_column = np.append(var_column, 0.0)
            elif isinstance(inter_value, np.ma.core.MaskedConstant) or isinstance(intra_value, np.ma.core.MaskedConstant):
                if isinstance(inter_value, np.ma.core.MaskedConstant):
                    var_column = np.append(var_column, intra_value)
                else:
                    var_column = np.append(var_column, inter_value)
            else:
                mean_value = np.mean([inter_value, intra_value])
                var_column = np.append(var_column, mean_value)
                
        cone_catalog["Variability"] = var_column
        
        return cone_catalog
    
    
    def neighbourhood_of_object(self, radius: Quantity, simulation_data: Dict) -> None:
        """
        Visualizes the neighborhood of a specified astronomical object within a given radius.

        This method creates a set of plots to visually represent the neighborhood of a celestial object. It uses data from 
        the Chandra Source Catalog and a simulation data dictionary. The plots include the positions of nearby sources, 
        differentiating between variable and invariant sources. The plots are saved as a PNG file and also displayed.

        Parameters:
        radius (Quantity): The search radius around the celestial object, specified as an astropy Quantity.
        simulation_data (Dict): A dictionary containing various simulation parameters and data. This must include keys 
                                'os_dictionary' for output settings and 'object_data' for the celestial object's data. 
                                'object_data' itself must be a dictionary containing the 'object_name' and 'object_position'.

        Notes:
        - The method creates and displays four scatter plots in a 2x2 grid. Each plot shows different aspects of the 
        neighborhood, such as sources from different catalogs and variability information.
        - The object's position is marked distinctly in each plot.
        - The generated plots are saved to a file in the directory specified in 'os_dictionary' within 'simulation_data'.
        - The method does not return any value.

        """
        
        os_dictionary = simulation_data["os_dictionary"]
        object_data = simulation_data["object_data"]
        object_name = object_data["object_name"]
        
        cs_csc_ra = np.array(list(self.cone_search_catalog[self.ra_cs]), dtype=float)
        cs_csc_dec = np.array(list(self.cone_search_catalog[self.dec_cs]), dtype=float)
        
        csc_ra = np.array(self.nearby_sources_table[self.ra], dtype=float)
        csc_dec = np.array(self.nearby_sources_table[self.dec], dtype=float)
        csc_ra = list(set(csc_ra))
        csc_dec = list(set(csc_dec))
        
        figure_1, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True, sharey=True)
        figure_1.suptitle(f"Neighbourhood of {object_name}, radius = {radius}", fontsize=20)
        figure_1.text(0.5, 0.04, 'Right Ascension [deg]', ha='center', va='center', fontsize=16)
        figure_1.text(0.04, 0.5, 'Declination [deg]', ha='center', va='center', rotation='vertical', fontsize=16)

        
        ax00 = axes[0][0]
        ax00.scatter(csc_ra, csc_dec, s=10, c='black', marker="*", label=f"Sources close to {object_name} : {len(csc_ra)}")
        ax00.scatter(object_data['object_position'].ra, object_data['object_position'].dec, marker='x', c='red', label=f"{object_name}")
        ax00.legend(loc='upper right')
        ax00.set_title("With chandra.fits")
        
        ax01 = axes[0][1]
        ax01.scatter(cs_csc_ra, cs_csc_dec, s=10, c='black', marker="*", label=f"Sources close to {object_name} : {len(cs_csc_ra)}")
        ax01.scatter(object_data['object_position'].ra, object_data['object_position'].dec, marker='x', c='red', label=f"{object_name}")
        ax01.legend(loc='upper right')
        ax01.set_title("With cone search")
        
        ax10 = axes[1][0]

        cs_ra_var = [ra for index, ra in enumerate(list(self.cone_search_catalog[self.ra_cs])) if self.cone_search_catalog['Variability'][index] != 0.0]
        cs_ra_invar = [ra for index, ra in enumerate(list(self.cone_search_catalog[self.ra_cs])) if self.cone_search_catalog['Variability'][index] == 0.0]

        cs_dec_var = [dec for index, dec in enumerate(list(self.cone_search_catalog[self.dec_cs])) if self.cone_search_catalog['Variability'][index] != 0.0]
        cs_dec_invar = [dec for index, dec in enumerate(list(self.cone_search_catalog[self.dec_cs])) if self.cone_search_catalog['Variability'][index] == 0.0]

        ax11 = axes[1][1]
        ax11.scatter(cs_ra_var, cs_dec_var, s=10, c='darkorange', marker='*', label=f"Var src : {len(cs_ra_var)} sources")
        ax11.scatter(cs_ra_invar, cs_dec_invar, s=10, c='blue', marker='*', label=f"Invar src : {len(cs_ra_invar)} sources")
        ax11.scatter(object_data['object_position'].ra, object_data['object_position'].dec, marker='+', s=50, c='red', label=f"{object_name}")
        ax11.legend(loc="upper right", ncol=2)
        
        plt.savefig(os.path.join(os_dictionary["img"], f"neighbourhood_of_{object_name}.png".replace(" ", "_")))
        plt.show()


    def threshold(self, cone_search_catalog):
        """
        Corrects flux values in the Chandra Source Catalog by replacing missing or invalid data with minimum valid values.

        This method processes the Chandra Source Catalog to handle missing or invalid flux values. It iterates over various 
        flux-related columns in the catalog, replacing masked or NaN values with the minimum numerical value found in each 
        respective column. The method corrects both observed flux values and their errors, across different bands.

        Parameters:
        cone_search_catalog : The Chandra Source Catalog, typically an astropy Table, which contains the flux data to be 
                            corrected.

        Returns:
        The modified Chandra Source Catalog, with corrected flux values.

        Notes:
        - The method relies on the 'dict_cat.dictionary_catalog' for definitions of the flux columns to be processed.
        - The flux columns include observed fluxes, their errors, and band-specific fluxes.
        - The method iterates through each source in the catalog, checking for masked constants or NaN values in these 
        columns and replacing them with the minimum valid value found in the same column.
        - This correction is applied to ensure more accurate and meaningful analysis of the flux data.

        """
        
        source_number = len(cone_search_catalog)
        key = "CS_Chandra"

        # name : flux_powlaw_aper_b
        # algo to replace -- by the min numerical value of the list
        flux_obs = dict_cat.dictionary_catalog[key]["flux_obs"]
        flux_data = []
        for item in range(source_number):
            # iterate for each item in the list
            if not isinstance(self.cone_search_catalog[flux_obs][item], np.ma.core.MaskedConstant):
                # put in flux_data list each item of type different to np.ma.core.MaskedConstant
                flux_data.append(cone_search_catalog[flux_obs][item])
        flux = list(cone_search_catalog[flux_obs])
        corrected_flux_obs = np.nan_to_num(flux, nan=np.min(flux_data))
        cone_search_catalog[flux_obs] = corrected_flux_obs

        for name in dict_cat.dictionary_catalog[key]["flux_obs_err"]:
            flux_err = []
            for item in range(source_number):
                # iterate for each item in the list
                if not isinstance(cone_search_catalog[name][item], np.ma.core.MaskedConstant):
                    # put in flux_err list each item of type different to np.ma.core.MaskedConstant
                    flux_err.append(cone_search_catalog[name][item])
            flux_err_obs = list(cone_search_catalog[name])
            corrected_flux_err_obs = np.nan_to_num(flux_err_obs, nan=np.min(flux_err))
            cone_search_catalog[name] = corrected_flux_err_obs

        # name : flux_powlaw_aper__s/m/h
        # algo to replace -- by the min numerical value of the list
        for name in dict_cat.dictionary_catalog[key]["band_flux_obs"]:
            # itera name in band_flux_obs 
            data = []
            for item in range(source_number):
                # iterate for each item in the list
                if not isinstance(cone_search_catalog[name][item], np.ma.core.MaskedConstant):
                    # put in data list each item of type different to np.ma.core.MaskedConstant
                    data.append(cone_search_catalog[name][item])
            flux = list(cone_search_catalog[name])
            corrected_flux = np.nan_to_num(flux, nan=np.min(data))
            cone_search_catalog[name] = corrected_flux


        # name : flux_powlaw_aper_lo/hi_lim_s/m/h
        err_flux_neg, err_flux_pos = dict_cat.dictionary_catalog[key]["band_flux_obs_err"][0], dict_cat.dictionary_catalog[key]["band_flux_obs_err"][1]
        # algo to replace -- by the min numerical value of the list
        for err_name_0, err_name_1 in zip(err_flux_neg, err_flux_pos):
            neg_data, pos_data = [], []
            for item in range(source_number):
                # iterate for each item in the list
                if not isinstance(cone_search_catalog[err_name_0][item], np.ma.core.MaskedConstant):
                    # put in neg_data list each item of type different to np.ma.core.MaskedConstant
                    neg_data.append(cone_search_catalog[err_name_0][item])
                if not isinstance(cone_search_catalog[err_name_1][item], np.ma.core.MaskedConstant):
                    # put in pos_data list each item of type different to np.ma.core.MaskedConstant
                    pos_data.append(cone_search_catalog[err_name_1][item])
                    
            neg_flux, pos_flux = list(cone_search_catalog[err_name_0]), list(cone_search_catalog[err_name_1])
            corrected_neg_flux = np.nan_to_num(neg_flux, nan=np.min(neg_data))
            corrected_pos_flux = np.nan_to_num(pos_flux, nan=np.min(pos_data))
            
            cone_search_catalog[err_name_0] = corrected_neg_flux
            cone_search_catalog[err_name_1] = corrected_pos_flux
            
        return cone_search_catalog
            

    def visualization_interp(self, optimization_parameters, photon_index, key) -> None:
        """
        Creates a logarithmic plot visualizing the interpolation of photon index values across different energy bands.

        This method generates a plot that visualizes the relationship between energy bands and fluxes, considering the 
        photon index values. It uses optimization parameters and photon index values for each energy band and displays the 
        results as an errorbar plot, with power-law models for absorption.

        Parameters:
        optimization_parameters (list): A list of tuples, each containing optimization parameters. Each tuple typically 
                                        includes flux observations, flux errors, and absorbed power-law values.
        photon_index (list): A list of photon index values corresponding to the optimization parameters.
        key (str): A key to access the energy band center values from 'dict_cat.dictionary_catalog'.

        Notes:
        - The method retrieves energy band center values from 'dict_cat.dictionary_catalog' using the provided key.
        - The plot is logarithmic, displaying energy in keV and flux in erg/cm^2/s/keV.
        - Each set of flux observations is plotted with error bars, and absorbed power-law models are plotted for each 
        photon index.
        - The plot is titled "Interpolation Photon Index plot" and includes appropriate axis labels.

        """
        
        energy_band = dict_cat.dictionary_catalog[key]["energy_band_center"]
        
        fig, axes = plt.subplots(1, 1, figsize=(15, 8))
        fig.suptitle("Interpolation Photon Index plot", fontsize=20)
        fig.text(0.5, 0.04, 'Energy [keV]', ha='center', va='center')
        fig.text(0.04, 0.5, 'Flux $[erg.cm^{-2}.s^{-1}.keV^{-1}]$', ha='center', va='center', rotation='vertical')
        
        for item in range(len(optimization_parameters)):
            flux_obs = optimization_parameters[item][1]
            flux_obs_err = optimization_parameters[item][2]
            absorbed_power_law = optimization_parameters[item][3]
            absorb_pho_index = photon_index[item]
            
            axes.errorbar(energy_band, flux_obs, flux_obs_err, fmt='*', color='red', ecolor='black')
            axes.plot(energy_band, absorbed_power_law, label=f"$\Gamma$ = {absorb_pho_index:.8f}")
    
        axes.legend(loc="upper left", ncol=4, fontsize=6)
        axes.loglog()
        
        plt.show()


    def get_photon_index(self, key, table, index):
        """
        Calculates the photon index for a given source using absorbed power-law fitting.

        This method computes the photon index for an astronomical source based on its flux observations across different 
        energy bands. It uses an absorbed power-law model for the fit. The method selects the appropriate data based on the 
        provided catalog key and then performs the fitting process to determine the photon index and other optimization 
        parameters.

        Parameters:
        key (str): The key to identify the catalog and retrieve the relevant data from 'dict_cat.dictionary_catalog'.
        table (Table): An astropy Table containing flux data for the source in various energy bands.
        index (int): The index of the source in the table for which the photon index is to be calculated.

        Returns:
        Tuple[float, Tuple]: A tuple containing the photon index and a tuple of optimization parameters. The optimization 
                            parameters include energy band centers, observed fluxes, flux errors, and absorbed power-law 
                            values.

        Notes:
        - The method supports different keys for different catalogs (e.g., 'Chandra', 'CS_Chandra').
        - The absorbed power-law fitting is done using the curve_fit function from scipy.optimize.
        - If the fitting process fails, a default photon index value of 1.7 is returned.
        - The optimization parameters are used for plotting in the 'visualization_interp' method.

        """
        
        if key == "Chandra":
            interp_data = {"band_flux_obs": dict_cat.dictionary_catalog[key]["band_flux_obs"],
                           "band_flux_obs_err": dict_cat.dictionary_catalog[key]["band_flux_obs_err"],
                           "energy_band_center": dict_cat.dictionary_catalog[key]["energy_band_center"],
                           "energy_band_half_width": dict_cat.dictionary_catalog[key]["energy_band_half_width"]}
        
        if key == "CS_Chandra":
            interp_data = {"band_flux_obs": dict_cat.dictionary_catalog[key]["band_flux_obs"],
                           "band_flux_obs_err": dict_cat.dictionary_catalog[key]["band_flux_obs_err"],
                           "energy_band_center": dict_cat.dictionary_catalog[key]["energy_band_center"],
                           "energy_band_half_width": dict_cat.dictionary_catalog[key]["energy_band_half_width"]}
            
        def absorbed_power_law(energy_band, constant, gamma):
            sigma = np.array([1e-20, 1e-22, 1e-24], dtype=float)
            return (constant * energy_band **(-gamma)) * (np.exp(-sigma*3e20))
        
        tab_width = 2 * interp_data["energy_band_half_width"]
        
        flux_obs = [table[band_flux][index] for band_flux in interp_data["band_flux_obs"]]
        
        flux_err = [[table[err_0][index] for err_0 in interp_data["band_flux_obs_err"][0]],
                    [table[err_1][index] for err_1 in interp_data["band_flux_obs_err"][1]]]
        
        flux_err_obs = [np.mean([err_0, err_1]) for (err_0, err_1) in zip(flux_err[0], flux_err[1])]
        
        y_array = [num/det for num, det in zip(flux_obs, tab_width)]
        yerr_array = [num/det for num, det in zip(flux_err_obs, tab_width)]
        
        try:
            popt, pcov = curve_fit(absorbed_power_law, interp_data["energy_band_center"], y_array, sigma=yerr_array)
            constant, photon_index = popt
        except Exception as error:
            photon_index = 1.7
            
        optimization_parameters = (interp_data["energy_band_center"], y_array, yerr_array, absorbed_power_law(interp_data["energy_band_center"], *popt))
            
        return photon_index, optimization_parameters 
        

    def get_phoindex_nh(self):
        """
        Computes and assigns photon index and hydrogen column density (Nh) values for sources in the Chandra Source Catalog and nearby sources.

        This method calculates the photon index for each source in the Chandra Source Catalog and a table of nearby sources. 
        It uses the 'powlaw_gamma' and 'nh_gal' columns from the catalog, along with additional computation for sources 
        lacking this data. The method updates the tables with new columns for photon index and Nh values and visualizes 
        the interpolation of photon indices.

        Returns:
        Tuple[List[float], List[float]]: A tuple containing two lists. The first list contains photon index values for 
                                        sources in the Chandra Source Catalog. The second list contains photon index 
                                        values for nearby sources.

        Notes:
        - The method iterates through the Chandra Source Catalog and the nearby sources table, computing photon indices 
        where necessary using the 'get_photon_index' method.
        - Nh values are calculated based on 'nh_gal' data, with a default value assigned where this data is missing.
        - The computed photon indices and Nh values are added to the respective tables as new columns.
        - The method visualizes the photon indices using the 'visualization_interp' method for both the Chandra Source 
        Catalog and nearby sources.
        - Default photon index and Nh values are used for cases with missing or invalid data.

        """
        
        key = "Chandra"
        cs_key = "CS_Chandra"
        
        cs_photon_index_list, photon_index_list = [], []
        cs_parameters_list, parameters_list = [], []
        self.cs_nh_list, self.nh_list = [], []
        
        for (index, item), nh_value in zip(enumerate(self.cone_search_catalog["powlaw_gamma"]), self.cone_search_catalog["nh_gal"]):
            # Photon Index 
            if item != 0:
                cs_photon_index_list.append(item)
            else:
                photon, params = self.get_photon_index(key=cs_key, table=self.cone_search_catalog, index=index)
                photon = photon if photon > 0.0 else 1.7
                cs_parameters_list.append(params)
                cs_photon_index_list.append(photon)
            if nh_value != 0:
                self.cs_nh_list.append(nh_value*1e20)
            else:
                self.cs_nh_list.append(3e20)
                
        self.visualization_interp(optimization_parameters=cs_parameters_list, photon_index=cs_photon_index_list, key=cs_key)

        for index, name in enumerate(list(self.cone_search_catalog['name'])): 
            if name in self.nearby_sources_table["Chandra_IAUNAME"]:
                nearby_index = list(self.nearby_sources_table["Chandra_IAUNAME"]).index(name)
                if self.cone_search_catalog["powlaw_gamma"][index] != 0.0:
                    photon_index_list.append(self.cone_search_catalog["powlaw_gamma"][index])
                else:
                    photon , params = self.get_photon_index(key=key, table=self.nearby_sources_table, index=nearby_index) #TODO modifier index ici pour avoir celui de nearby sources table
                    photon = photon if photon > 0.0 else 1.7
                    parameters_list.append(params)
                    photon_index_list.append(photon)
                    
                if self.cone_search_catalog["nh_gal"][index] != 0.0:
                    self.nh_list.append(self.cone_search_catalog["nh_gal"][index]*1e20)
                else:
                    self.nh_list.append(3e20)
        
        self.visualization_interp(optimization_parameters=parameters_list, photon_index=photon_index_list, key=key)
        
        self.nearby_sources_table["Photon Index"] = photon_index_list
        self.nearby_sources_table["Nh"] = self.nh_list
        
        self.cone_search_catalog["Photon Index"] = cs_photon_index_list
        self.cone_search_catalog['Nh'] = self.cs_nh_list
            
        return cs_photon_index_list, photon_index_list
                
  
    def dictionary_model(self) -> Dict[str, Dict[str, Union[str, float]]]:
        """
        Constructs dictionaries with model parameters for sources in the Chandra Source Catalog and nearby sources.

        This method creates two dictionaries, one for the Chandra Source Catalog and another for nearby sources. Each 
        dictionary contains entries for individual sources, detailing their astrophysical model type (e.g., 'power'), 
        model values (e.g., photon index), flux values, and column density.

        Returns:
        Dict[str, Dict[str, Union[str, float]]]: A tuple of two dictionaries. The first dictionary corresponds to the 
                                                Chandra Source Catalog, and the second to the nearby sources. Each entry 
                                                in these dictionaries represents a source, keyed by a unique identifier 
                                                (e.g., 'src_0'), and contains information about the source's model, 
                                                model value, flux, and column density.

        Notes:
        - The method assumes that the 'Photon Index' and 'Nh' values have already been calculated and added to the 
        respective tables.
        - Currently, the method only handles the 'power' model type. Placeholder code exists for other model types 
        ('black_body', 'temp'), but these are not yet implemented.
        - Each source is assigned a unique key in the dictionary (e.g., 'src_0', 'src_1', etc.).
        - The method utilizes 'dict_cat.dictionary_catalog' to access flux observation column names for the respective 
        catalogs.

        """
        
        model_dictionary, cs_model_dictionary = {}, {}
        cs_number_source, number_source = len(self.cone_search_catalog), len(self.nearby_sources_table)

        cs_model = np.array([], dtype=str)
        cs_model_value = []
        for item in range(cs_number_source):
            cs_model = np.append(cs_model, 'power')

        for item in range(cs_number_source):
            if cs_model[item] == 'power':
                cs_model_value = np.append(cs_model_value, self.cone_search_catalog["Photon Index"][item])
            elif cs_model[item] == 'black_body':
                pass # Pas de valeur pour le moment...
            elif cs_model[item] == 'temp':
                pass # Pas de valeur pour le moment... (dernier model pimms)
                
        model = np.array([], dtype=str)
        model_value = []
        for item in range(number_source):
            model = np.append(model, 'power')

        for item in range(number_source):
            if model[item] == 'power':
                model_value = np.append(model_value, self.nearby_sources_table["Photon Index"][item])
            elif model[item] == 'black_body':
                pass # Pas de valeur pour le moment...
            elif model[item] == 'temp':
                pass # Pas de valeur pour le moment... (dernier model pimms)

        cs_flux = list(self.cone_search_catalog[dict_cat.dictionary_catalog["CS_Chandra"]["flux_obs"]])
        flux = list(self.nearby_sources_table[dict_cat.dictionary_catalog["Chandra"]["flux_obs"]])

        for item in range(cs_number_source):
            cs_dictionary = {
                "model": cs_model[item],
                "model_value": cs_model_value[item],
                "flux": cs_flux[item],
                "column_dentsity": self.cs_nh_list[item],
            }
            cs_model_dictionary[f"src_{item}"] = cs_dictionary
        
        for item in range(number_source):
            dictionary = {
                "model": model[item],
                "model_value": model_value[item],
                "flux": flux[item],
                "column_dentsity": self.nh_list[item]
            }
            model_dictionary[f"src_{item}"] = dictionary

        return cs_model_dictionary, model_dictionary


class Swift:
    """
    A class for analyzing astronomical data from the Swift catalog.

    This class provides methods for opening and analyzing data from the Swift catalog. It includes functionalities 
    for finding nearby sources, visualizing the neighborhood of a specific object, calculating photon indices, 
    and building a dictionary model for the sources.

    Attributes:
        ra (str): Right ascension column name in the Swift catalog.
        dec (str): Declination column name in the Swift catalog.
        swi_catalog (Table): The Swift catalog data as an astropy Table.
        nearby_sources_table (Table): Table of nearby sources.
        nearby_sources_position (SkyCoord): Sky coordinates of nearby sources.
        photon_index (List[float]): List of photon index values for sources.
        model_dictionary (Dict[str, Dict[str, Union[str, float]]]): Dictionary of source models.

    Parameters:
        catalog_path (str): Path to the Swift catalog file.
        radius (Quantity): Radius for searching nearby sources.
        simulation_data (dict): Dictionary containing simulation data.
        user_table (Table): User-provided astropy Table.

    Methods:
        open_catalog(catalog_path): Opens the Swift catalog file and returns it as an astropy Table.
        find_nearby_sources(radius, object_data): Finds and returns nearby sources and their positions.
        neighbourhood_of_object(radius, simulation_data): Visualizes the neighborhood of a specified object.
        visualization_inter(optimization_parameters, photon_index, key): Visualizes the interpolation of photon index values.
        get_photon_index(table, key, index): Calculates the photon index for a given source.
        get_phoindex_nh(): Computes and assigns photon index and Nh values for sources.
        dictionary_model(): Constructs a dictionary with model parameters for sources.
    """

    def __init__(self, catalog_path: str, radius: Quantity, simulation_data: dict, user_table: Table) -> None:
        """
        Initializes the Swift class with specific catalog data, search radius, and simulation parameters.

        This constructor loads the Swift astronomical catalog from a given path, searches for nearby sources within a 
        specified radius around a provided celestial object, and initiates several analyses including neighborhood 
        visualization, photon index calculation, and model dictionary creation.

        Parameters:
            catalog_path (str): The file path to the Swift catalog.
            radius (Quantity): The radius within which to search for nearby sources, specified as an astropy Quantity.
            simulation_data (dict): A dictionary containing simulation data, including details about the celestial object.
            user_table (Table): An astropy Table provided by the user, not used in the current implementation.

        Attributes created:
            ra (str): Right ascension column name as specified in dict_cat.dictionary_coord for Swift.
            dec (str): Declination column name as specified in dict_cat.dictionary_coord for Swift.
            swi_catalog (Table): The Swift catalog data as an astropy Table, loaded from the specified catalog_path.
            nearby_sources_table (Table): Table of sources found near the specified celestial object.
            nearby_sources_position (SkyCoord): Sky coordinates of the sources found near the specified celestial object.
            photon_index (List[float]): List of photon index values for sources.
            model_dictionary (Dict[str, Dict[str, Union[str, float]]]): Dictionary of model parameters for each source.
        """    
        # ---------- coord ---------- #
        self.ra = dict_cat.dictionary_coord["Swift"]["right_ascension"]
        self.dec = dict_cat.dictionary_coord["Swift"]["declination"]
        # --------------------------- #

        self.swi_catalog = self.open_catalog(catalog_path)
        self.nearby_sources_table, self.nearby_sources_position = self.find_nearby_sources(radius=radius, object_data=simulation_data["object_data"])
        
        self.neighbourhood_of_object(radius=radius, simulation_data=simulation_data)
        self.photon_index = self.get_phoindex_nh()
        self.model_dictionary = self.dictionary_model()


    def open_catalog(self, catalog_path: str)-> Table:
        """
        Opens a FITS file containing the Swift catalog and returns it as an astropy Table.

        Parameters:
        catalog_path (str): The file path to the Swift catalog.

        Returns:
        Table: An astropy Table containing the data from the Swift catalog.
        """
        with fits.open(catalog_path, memmap=True) as data:
            return Table(data[1].data)

            
    def find_nearby_sources(self, radius: Quantity, object_data: dict) -> Tuple[Table, SkyCoord]:
        """
        Finds sources in the Swift catalog that are near a specified celestial object.

        Parameters:
        radius (Quantity): The search radius around the object, specified as an astropy Quantity.
        object_data (dict): A dictionary containing information about the celestial object.

        Returns:
        Tuple[Table, SkyCoord]: A tuple containing an astropy Table of nearby sources and their SkyCoord positions.
        """
    
        field_of_view = radius + 5*u.arcmin
        name = object_data["object_name"]
        object_position = object_data['object_position']

        min_ra, max_ra = object_position.ra - field_of_view, object_position.ra + field_of_view
        min_dec, max_dec = object_position.dec - field_of_view, object_position.dec + field_of_view

        small_table = Table(names=self.swi_catalog.colnames,
                            dtype=self.swi_catalog.dtype)

        nearby_sources_table = Table(names= self.swi_catalog.colnames,
                                     dtype= self.swi_catalog.dtype)

        print(fr"{colored('Reducing Swift catalog...', 'yellow')} to fov of {radius.value + 5 } arcmin")
        for number in tqdm(range(len(self.swi_catalog))):
            if min_ra/u.deg < self.swi_catalog[self.ra][number] < max_ra/u.deg and min_dec/u.deg < self.swi_catalog[self.dec][number] < max_dec/u.deg:
                small_table.add_row(self.swi_catalog[number])
                
        src_position = SkyCoord(ra=small_table[self.ra], dec=small_table[self.dec], unit=u.deg)
                
        print(f"{colored(f'Find sources close to {name} with Swift catalog', 'blue')}")
        for number in tqdm(range(len(small_table))):
            if f.ang_separation(object_position, src_position[number]) < radius:
                nearby_sources_table.add_row(small_table[number])

        column_name = {"source_name": "Swift_IAUNAME", 
                       "right_ascension": self.ra,
                       "declination": self.dec,
                       "catalog_name": "Swift"}
        
        if len(nearby_sources_table) != 0:
            try:
                unique_table = f.create_unique_sources_catalog(nearby_sources_table=nearby_sources_table, column_name=column_name)
            except Exception as error:
                print(f"{colored('An error occured : ', 'red')} {error}")
        else:
            print("Nearby sources table from Swift catalog is empty.")
            sys.exit()
            
        nearby_src_position = SkyCoord(ra=unique_table[self.ra], dec=unique_table[self.dec], unit=u.deg)
                
        try :
            if len(unique_table) != 0:
                print((f"We have detected {len(unique_table)} sources close to {object_data['object_name']}"))
                return unique_table, nearby_src_position
            else:
                print(f"No sources detected close to {object_data['object_name']}.")
                sys.exit()
        except Exception as error:
            print(f"An error occured : {error}")


    def neighbourhood_of_object(self, radius: Quantity, simulation_data: dict) -> None:
        """
        Visualizes the neighborhood of a specified celestial object in the Swift catalog.

        This method creates a scatter plot showing the positions of sources found near the specified celestial object. 
        It highlights both the nearby sources and the object itself, helping to understand their spatial distribution.

        Parameters:
        radius (Quantity): The search radius around the celestial object, specified as an astropy Quantity.
        simulation_data (dict): A dictionary containing simulation data. This should include 'object_data' with the 
                                target object's information and 'os_dictionary' for output settings.

        The method plots the right ascension and declination of the nearby sources and the target object. It saves the 
        plot to a specified location and also displays it.

        Note:
        'object_data' within simulation_data should contain 'object_name' for the title of the plot and 
        'object_position' for the object's coordinates. 'os_dictionary' should contain the 'img' key with the path 
        to save the plot.
        """
        object_data = simulation_data["object_data"]
        object_position = object_data['object_position']
        os_dictionary = simulation_data["os_dictionary"]
    
        swi_ra = self.nearby_sources_table[self.ra]
        swi_dec = self.nearby_sources_table[self.dec]
        
        corrected_swi_ra = list(set(swi_ra))
        corrected_swi_dec = list(set(swi_dec))
        
        figure_1, axes = plt.subplots(1, 1, figsize=(12, 8))
        figure_1.suptitle(f"Neighbourhood of {object_data['object_name']}, radius = {radius}", fontsize=20)
        
        axes.scatter(corrected_swi_ra, corrected_swi_dec, s=30, facecolors='none', edgecolors='black', label=f"Sources : {len(self.nearby_sources_table)}")
        axes.scatter(object_position.ra, object_position.dec, c='red', s=100, marker='*', label=f"{object_data['object_name']}")
        axes.set_xlabel('Right Ascension [deg]', fontsize=16)
        axes.set_ylabel('Declination [deg]', fontsize=16)
        axes.legend(loc='upper right')
        
        img = os_dictionary["img"]
        img_path = os.path.join(img, f"neighbourhood_of_{object_data['object_name']}.png".replace(" ", "_"))
        plt.savefig(img_path)
        plt.show()
    
    
    def visualization_inter(self, optimization_parameters, photon_index, key) -> None:
        """
        Visualizes the interpolation of photon index values across different energy bands.

        This method creates a plot that illustrates the relationship between photon index values and energy bands. 
        It uses error bars to depict observed fluxes and plots absorbed power-law models for each energy band based 
        on the photon index values.

        Parameters:
        optimization_parameters: A list of tuples containing optimization parameters for each source. Each tuple should 
                                include observed fluxes, flux errors, and absorbed power-law values.
        photon_index (list): A list of photon index values corresponding to each set of optimization parameters.
        key (str): The key used to access energy band center values from 'dict_cat.dictionary_catalog'.

        The plot displays energy in keV and flux in erg/cm^2/s/keV. Each source's data is plotted with its photon index, 
        providing a visual representation of the spectral characteristics of the sources.
        """
        
        energy_band = dict_cat.dictionary_catalog[key]["energy_band_center"]
            
        fig, axes = plt.subplots(1, 1, figsize=(15, 8))
        fig.suptitle("Interpolation Photon Index plot", fontsize=20)
        fig.text(0.5, 0.04, 'Energy [keV]', ha='center', va='center')
        fig.text(0.04, 0.5, 'Flux $[erg.cm^{-2}.s^{-1}.keV^{-1}]$', ha='center', va='center', rotation='vertical')
        
        for item in range(len(optimization_parameters)):
            flux_obs = optimization_parameters[item][1]
            flux_obs_err = optimization_parameters[item][2]
            absorbed_power_law = optimization_parameters[item][3]
            absorb_pho_index = photon_index[item]
            
            axes.errorbar(energy_band, flux_obs, flux_obs_err, fmt='*', color='red', ecolor='black')
            axes.plot(energy_band, absorbed_power_law, label=f"$\Gamma$ = {absorb_pho_index:.8f}")

        axes.legend(loc="upper left", ncol=4, fontsize=6)
        axes.loglog()
        
        plt.show()
        

    def get_photon_index(self, table, key, index) -> Tuple[List, Tuple]:
        """
        Calculates the photon index for a specified source in the Swift catalog using absorbed power-law fitting.

        This method computes the photon index of a source by fitting its flux observations across different energy bands 
        to an absorbed power-law model. The method takes into account the flux errors and the energy band properties 
        while performing the fit.

        Parameters:
        table (Table): An astropy Table containing the catalog data.
        key (str): The key to access the required data from 'dict_cat.dictionary_catalog'.
        index (int): The index of the source in the table for which the photon index is to be calculated.

        Returns:
        Tuple[float, Tuple]: A tuple containing the photon index and a tuple of optimization parameters. The optimization 
                            parameters include energy band centers, observed fluxes, flux errors, and absorbed power-law 
                            values.

        The method uses the curve_fit function from scipy.optimize to perform the fitting. If the fitting fails, a 
        default photon index value is returned.
        """
        
        interp_data = {"band_flux_obs": dict_cat.dictionary_catalog[key]["band_flux_obs"],
                       "band_flux_obs_err": dict_cat.dictionary_catalog[key]["band_flux_obs_err"],
                       "energy_band_center": dict_cat.dictionary_catalog[key]["energy_band_center"],
                       "energy_band_half_width": dict_cat.dictionary_catalog[key]["energy_band_half_width"]}
        
        def absorbed_power_law(energy_band, constant, gamma):
            sigma = np.array(np.linspace(1e-20, 1e-24, len(energy_band)), dtype=float)
            return (constant * energy_band **(-gamma)) * (np.exp(-sigma*3e20))

        tab_width = 2 * interp_data["energy_band_half_width"]
        
        flux_obs = [table[band_flux][index] for band_flux in interp_data["band_flux_obs"]]
        
        flux_err = [[table[err_0][index] for err_0 in interp_data["band_flux_obs_err"][0]],
                    [table[err_1][index] for err_1 in interp_data["band_flux_obs_err"][1]]]
        
        flux_err_obs = [np.mean([err_0, err_1]) for (err_0, err_1) in zip(flux_err[0], flux_err[1])]
        
        y_array = [num/det for num, det in zip(flux_obs, tab_width)]
        yerr_array = [num/det for num, det in zip(flux_err_obs, tab_width)]
        
        try:
            popt, pcov = curve_fit(absorbed_power_law, interp_data["energy_band_center"], y_array, sigma=yerr_array)
            constant, photon_index = popt
        except Exception as error:
            popt = (1e-14, 1.7)
            
        optimization_parameters = (interp_data["energy_band_center"], y_array, yerr_array, absorbed_power_law(interp_data["energy_band_center"], *popt))
            
        return photon_index, optimization_parameters 


    def get_phoindex_nh(self) -> List:
        """
        Computes photon index and hydrogen column density (Nh) for sources in the Swift catalog's nearby sources table.

        This method calculates the photon index for each source in the nearby sources table using the absorbed power-law 
        model. It also assigns a default hydrogen column density (Nh) value to each source. The method then visualizes 
        the photon indices using a plot and updates the nearby sources table with the calculated photon indices and Nh values.

        Returns:
        List[float]: A list containing the photon index values for each source in the nearby sources table.

        The method assumes a default Nh value of 3e20. If the calculated photon index is negative or zero, a default 
        value of 1.7 is used. The visualization of photon indices is handled by the 'visualization_inter' method.
        """
        key = "Swift"
        photon_index_list, parameters_list, nh_list = [], [], []

        for index in range(len(self.nearby_sources_table)):
            nh_list.append(3e20)
            photon, params = self.get_photon_index(table=self.nearby_sources_table, key=key, index=index)
            photon_index_list.append(photon if photon > 0.0 else 1.7)
            parameters_list.append(params)
        
        self.visualization_inter(optimization_parameters=parameters_list, photon_index=photon_index_list, key=key)
        
        self.nearby_sources_table["Photon Index"] = photon_index_list
        self.nearby_sources_table["Nh"] = nh_list
        
        return photon_index_list
    
    
    def dictionary_model(self) -> Dict[str, Dict[str, Union[str, float]]]:
        """
        Creates a dictionary containing model parameters for each source in the nearby sources table.

        This method iterates over the nearby sources table and compiles a dictionary where each entry corresponds 
        to a source. The dictionary for each source includes its model type, model value (such as the photon index), 
        observed flux, and column density.

        Returns:
        Dict[str, Dict[str, Union[str, float]]]: A dictionary where each key represents a source (formatted as 'src_{index}')
        and the value is another dictionary containing the source's modeling parameters.

        The current implementation only handles the 'power' model type, with placeholders for 'black_body' and 'temp' 
        types for future expansion. The method assumes that the 'Photon Index' and 'Nh' values are already computed 
        and available in the nearby sources table.
        """
        
        model_dictionary = {}
        number_source = len(self.nearby_sources_table)

        model = np.array([], dtype=str)
        model_value = np.array([], dtype=float)
        flux_obs = dict_cat.dictionary_catalog["Swift"]["flux_obs"]
        
        swi_flux = np.array([self.nearby_sources_table[flux_obs][item] for item in range(number_source)], dtype=float)
        nh_value = np.array([self.nearby_sources_table["Nh"][item] for item in range(number_source)], dtype=float)
        
        # Pour le moment seulement 'power' indiquant le modèle a utiliser pour la commande pimms
        for item in range(number_source):
            model = np.append(model, 'power')    
            
        for item in range(number_source):
            if model[item] == 'power':
                model_value = np.append(model_value, self.nearby_sources_table["Photon Index"][item])
            elif model[item] == 'black_body':
                pass # Pas de valeur pour le moment...
            elif model[item] == 'temp':
                pass # Pas de valeur pour le moment... (dernier model pimms)

        for item in range(number_source):

            dictionary = {
                "model": model[item],
                "model_value": model_value[item],
                "flux": swi_flux[item],
                "column_dentsity": nh_value[item]
            }

            model_dictionary[f"src_{item}"] = dictionary
            
        return model_dictionary 


class eRosita:
    """
    A class for analyzing astronomical data from the eRosita catalog.

    This class provides methods for opening and analyzing data from the eRosita catalog. It includes functionalities 
    for finding nearby sources, visualizing the neighborhood of a specific object, calculating photon indices, 
    and building a dictionary model for the sources.

    Attributes:
    ra (str): Right ascension column name in the eRosita catalog.
    dec (str): Declination column name in the eRosita catalog.
    eRo_catalog (Table): The eRosita catalog data as an astropy Table.
    nearby_sources_table (Table): Table of nearby sources.
    nearby_sources_position (SkyCoord): Sky coordinates of nearby sources.
    photon_index (List[float]): List of photon index values for sources.
    model_dictionary (Dict[str, Dict[str, Union[str, float]]]): Dictionary of source models.

    Parameters:
    catalog_path (str): Path to the eRosita catalog file.
    radius (Quantity): Radius for searching nearby sources.
    simulation_data (dict): Dictionary containing simulation data.
    user_table (Table): User-provided astropy Table.

    Methods:
    open_catalog(catalog_path): Opens the eRosita catalog file and returns it as an astropy Table.
    find_nearby_sources(radius, simulation_data): Finds and returns nearby sources and their positions.
    neighbourhood_of_object(dictionary, radius): Visualizes the neighborhood of a specified object.
    visualization_inter(optimization_parameters, photon_index, key): Visualizes the interpolation of photon index values.
    get_photon_index(table, key, index): Calculates the photon index for a given source.
    get_phoindex_nh(): Computes and assigns photon index and Nh values for sources.
    dictionary_model(): Constructs a dictionary with model parameters for sources.

    """

    def __init__(self, catalog_path: str, radius: Quantity, simulation_data: dict, user_table: Table) -> None:
        """
        Initializes the eRosita class with the specified catalog, search radius, simulation data, and user table.

        This constructor loads the eRosita astronomical catalog from the given path, searches for nearby sources within a 
        specified radius around a provided celestial object, and performs various analyses such as neighborhood 
        visualization, photon index calculation, and model dictionary creation.

        Parameters:
            catalog_path (str): The file path to the eRosita catalog.
            radius (Quantity): The radius within which to search for nearby sources, specified as an astropy Quantity.
            simulation_data (dict): A dictionary containing simulation data, including details about the celestial object.
            user_table (Table): An astropy Table provided by the user, not used in the current implementation.

        Attributes created:
            ra (str): Right ascension column name as specified in dict_cat.dictionary_coord for eRosita.
            dec (str): Declination column name as specified in dict_cat.dictionary_coord for eRosita.
            eRo_catalog (Table): The eRosita catalog data as an astropy Table, loaded from the specified catalog_path.
            nearby_sources_table (Table): Table of sources found near the specified celestial object.
            nearby_sources_position (SkyCoord): Sky coordinates of the sources found near the specified celestial object.
            photon_index (List[float]): List of photon index values for sources.
            model_dictionary (Dict[str, Dict[str, Union[str, float]]]): Dictionary of model parameters for each source.
        """    
        # ---------- coord ---------- #
        self.ra = dict_cat.dictionary_coord["eRosita"]["right_ascension"]
        self.dec = dict_cat.dictionary_coord["eRosita"]["declination"]
        # --------------------------- #

        self.eRo_catalog = self.open_catalog(catalog_path)
        self.nearby_sources_table, self.nearby_sources_position = self.find_nearby_sources(radius=radius, simulation_data=simulation_data)
        
        self.neighbourhood_of_object(radius=radius, simulation_data=simulation_data)
        self.photon_index = self.get_phoindex_nh()
        self.model_dictionary = self.dictionary_model()
        
        
    def open_catalog(self, catalog_path: str) -> Table:
        """
        Opens a FITS file containing the eRosita catalog and returns it as an astropy Table.

        This method is responsible for loading the eRosita catalog data from a FITS file located at the specified path. 
        It utilizes memory mapping for efficient handling of large files.

        Parameters:
        catalog_path (str): The file path to the eRosita catalog.

        Returns:
        Table: An astropy Table containing the data from the eRosita catalog.
        """

        with fits.open(catalog_path, memmap=True) as data:
            return Table(data[1].data)

        
    def find_nearby_sources(self, radius: Quantity, simulation_data: dict) -> Tuple[Table, SkyCoord]:
        """
        Finds sources in the eRosita catalog that are near a specified celestial object.

        This method identifies sources within a specified radius around a celestial object's position. It searches the 
        eRosita catalog for sources within this field of view and compiles a list of nearby sources.

        Parameters:
        radius (Quantity): The search radius around the celestial object, specified as an astropy Quantity.
        simulation_data (dict): A dictionary containing information about the celestial object including its name and position.

        Returns:
        Tuple[Table, SkyCoord]: A tuple containing an astropy Table of nearby sources and their SkyCoord positions.

        The method calculates the field of view by adding an extra 5 arcminutes to the provided radius. It then filters 
        the eRosita catalog to find sources within this field of view. If no sources are found, or in case of an error, 
        the method provides appropriate feedback.
        """
        field_of_view = radius + 5*u.arcmin
        object_data = simulation_data["object_data"]
        object_name = object_data["object_name"]
        object_position = object_data['object_position']

        min_ra, max_ra = object_position.ra - field_of_view, object_position.ra + field_of_view
        min_dec, max_dec = object_position.dec - field_of_view, object_position.dec + field_of_view

        small_table = Table(names=self.eRo_catalog.colnames,
                            dtype=self.eRo_catalog.dtype)

        nearby_sources_table = Table(names=self.eRo_catalog.colnames,
                                     dtype=self.eRo_catalog.dtype)

        print(fr"{colored('Reducing Swift catalog...', 'yellow')} to fov of {radius.value + 5 } arcmin")
        for number in tqdm(range(len(self.eRo_catalog))):
            if min_ra/u.deg < self.eRo_catalog[self.ra][number] < max_ra/u.deg and min_dec/u.deg < self.eRo_catalog[self.dec][number] < max_dec/u.deg:
                small_table.add_row(self.eRo_catalog[number])
                
        src_position = SkyCoord(ra=small_table[self.ra], dec=small_table[self.dec], unit=u.deg)
                
        print(f"{colored(f'Find sources close to {object_name} with eRosita catalog', 'blue')}")
        for number in tqdm(range(len(small_table))):
            if f.ang_separation(object_position, src_position[number]) < radius:
                nearby_sources_table.add_row(small_table[number])
        
        column_name = {"source_name": "eRosita_IAUNAME",
                       "right_ascension": self.ra,
                       "declination": self.dec,
                       "catalog_name": "eRosita"}
        
        if len(nearby_sources_table) != 0:
            try:
                unique_table = f.create_unique_sources_catalog(nearby_sources_table=nearby_sources_table, column_name=column_name)
            except Exception as error:
                print(f"{colored('An error occured : ', 'red')} {error}")
        else:
            print("Nearby sources table from Swift catalog is empty.")
            sys.exit()
        
           
        nearby_sources_position = SkyCoord(ra=unique_table[self.ra], dec=unique_table[self.dec], unit=u.deg)
                
        try :
            if len(unique_table) != 0:
                print((f"We have detected {len(unique_table)} sources close to {object_data['object_name']}"))
                return unique_table, nearby_sources_position
            else:
                print(f"No sources detected close to {object_data['object_name']}.")
                sys.exit()
        except Exception as error:
            print(f"An error occured : {error}")
    
    
    def neighbourhood_of_object(self, dictionary: dict, radius: Quantity) -> None:
        """
        Visualizes the neighborhood of a specified celestial object in the eRosita catalog.

        This method creates a scatter plot showing the positions of sources found near the specified celestial object. 
        It highlights both the nearby sources and the object itself, helping to understand their spatial distribution.

        Parameters:
        dictionary (dict): A dictionary containing information about the celestial object, including its name and position.
        radius (Quantity): The search radius around the celestial object, specified as an astropy Quantity.

        The method plots the right ascension and declination of the nearby sources and the target object. The celestial 
        object's name and the number of nearby sources found are displayed as part of the plot's legend.

        The plot visually represents the distribution of sources within the specified radius from the celestial object, 
        helping to analyze the object's local astronomical neighborhood.
        """
        
        object_position = dictionary['object_position']
    
        ero_ra = self.nearby_sources_table[self.ra]
        ero_dec = self.nearby_sources_table[self.dec]
        
        figure_1, axes = plt.subplots(1, 1, figsize=(12, 8))
        figure_1.suptitle(f"Neighbourhood of {dictionary['object_name']}, radius = {radius}", fontsize=20)
        
        axes.scatter(ero_ra, ero_dec, c='black', s=1, marker='*', label=f"Sources close to {dictionary['object_name']}, nbr_src : {len(ero_ra)}")
        axes.scatter(object_position.ra, object_position.dec, c='red', s=100, marker='+', label=f"{dictionary['object_name']}")
        axes.set_xlabel('Right Ascension [deg]', fontsize=16)
        axes.set_ylabel('Declination [deg]', fontsize=16)
        axes.legend(loc='upper right')
        
        plt.show()


    def visualization_inter(self, optimization_parameters, photon_index, key) -> None:
        """
        Visualizes the interpolation of photon index values across different energy bands for astronomical sources.

        This method creates a plot illustrating the relationship between photon index values and energy bands for sources 
        in the eRosita catalog. It uses error bars to represent observed fluxes and plots the absorbed power-law models 
        for each energy band based on the photon index values.

        Parameters:
        optimization_parameters: A list of tuples, each containing optimization parameters for a source. These parameters 
                                typically include observed fluxes, flux errors, and absorbed power-law values.
        photon_index (list): A list of photon index values corresponding to each set of optimization parameters.
        key (str): The key used to access energy band center values from 'dict_cat.dictionary_catalog'.

        The plot displays energy in keV and flux in erg/cm^2/s/keV. It uses a logarithmic scale for both axes to 
        effectively visualize the spectral data. Each source's data is plotted with its corresponding photon index, 
        providing insights into the spectral characteristics of the sources in the catalog.
        """
        
        energy_band = dict_cat.dictionary_catalog[key]["energy_band_center"]
            
        fig, axes = plt.subplots(1, 1, figsize=(15, 8))
        fig.suptitle("Interpolation Photon Index plot", fontsize=20)
        fig.text(0.5, 0.04, 'Energy [keV]', ha='center', va='center')
        fig.text(0.04, 0.5, 'Flux $[erg.cm^{-2}.s^{-1}.keV^{-1}]$', ha='center', va='center', rotation='vertical')
        
        for item in range(len(optimization_parameters)):
            flux_obs = optimization_parameters[item][1]
            flux_obs_err = optimization_parameters[item][2]
            absorbed_power_law = optimization_parameters[item][3]
            absorb_pho_index = photon_index[item]
            
            axes.errorbar(energy_band, flux_obs, flux_obs_err, fmt='*', color='red', ecolor='black')
            axes.plot(energy_band, absorbed_power_law, label=f"$\Gamma$ = {absorb_pho_index:.8f}")

        axes.legend(loc="upper left", ncol=4, fontsize=6)
        axes.loglog()
        
        plt.show()
        

    def get_photon_index(self, table, key, index) -> Tuple[List, Tuple]:
        """
        Calculates the photon index for a specified source in the eRosita catalog using an absorbed power-law fitting.

        This method computes the photon index of a source by fitting its flux observations across different energy bands 
        to an absorbed power-law model. It considers the flux errors and the energy band properties during the fitting.

        Parameters:
        table (Table): An astropy Table containing the catalog data.
        key (str): The key to access the required data from 'dict_cat.dictionary_catalog'.
        index (int): The index of the source in the table for which the photon index is to be calculated.

        Returns:
        Tuple[float, Tuple]: A tuple containing the photon index and a tuple of optimization parameters. The optimization 
                            parameters include energy band centers, observed fluxes, flux errors, and absorbed power-law 
                            values.

        The method uses the curve_fit function from scipy.optimize to perform the fitting. If the fitting process 
        encounters any error, a default photon index value is returned.
        """
        interp_data = {"band_flux_obs": dict_cat.dictionary_catalog[key]["band_flux_obs"],
                       "band_flux_obs_err": dict_cat.dictionary_catalog[key]["band_flux_obs_err"],
                       "energy_band_center": dict_cat.dictionary_catalog[key]["energy_band_center"],
                       "energy_band_half_width": dict_cat.dictionary_catalog[key]["energy_band_half_width"]}
        
        def absorbed_power_law(energy_band, constant, gamma):
            sigma = np.array(np.linspace(1e-20, 1e-24, len(energy_band)), dtype=float)
            return (constant * energy_band **(-gamma)) * (np.exp(-sigma*3e20))

        tab_width = 2 * interp_data["energy_band_half_width"]
        
        flux_obs = [table[band_flux][index] for band_flux in interp_data["band_flux_obs"]]
        
        flux_err = [[table[err_0][index] for err_0 in interp_data["band_flux_obs_err"][0]],
                    [table[err_1][index] for err_1 in interp_data["band_flux_obs_err"][1]]]
        
        flux_err_obs = [np.mean([err_0, err_1]) for (err_0, err_1) in zip(flux_err[0], flux_err[1])]
        
        y_array = [num/det for num, det in zip(flux_obs, tab_width)]
        yerr_array = [num/det for num, det in zip(flux_err_obs, tab_width)]
        
        try:
            popt, pcov = curve_fit(absorbed_power_law, interp_data["energy_band_center"], y_array, sigma=yerr_array)
            constant, photon_index = popt
        except Exception as error:
            popt = (1e-14, 1.7)
            
        optimization_parameters = (interp_data["energy_band_center"], y_array, yerr_array, absorbed_power_law(interp_data["energy_band_center"], *popt))
            
        return photon_index, optimization_parameters 


    def get_phoindex_nh(self) -> List:
        """
        Computes the photon index and hydrogen column density (Nh) for sources in the eRosita catalog's nearby sources table.

        This method calculates the photon index for each source in the nearby sources table using an absorbed power-law model. 
        It also assigns a default hydrogen column density (Nh) value to each source. The photon indices are visualized using a 
        plot, and the nearby sources table is updated with these new 'Photon Index' and 'Nh' values.

        Returns:
        List[float]: A list containing the photon index values for each source in the nearby sources table.

        The method uses a default Nh value of 3e20. If a calculated photon index is negative or zero, a default value of 1.7 is used. 
        The visualization of photon indices is done using the 'visualization_inter' method.
        """
        key = "eRosita"
        photon_index_list, parameters_list, nh_list = [], [], []

        for index in range(len(self.nearby_sources_table)):
            nh_list.append(3e20)
            photon, params = self.get_photon_index(table=self.nearby_sources_table, key=key, index=index)
            photon_index_list.append(photon if photon > 0.0 else 1.7)
            parameters_list.append(params)
        
        self.visualization_inter(optimization_parameters=parameters_list, photon_index=photon_index_list, key=key)
        
        self.nearby_sources_table["Photon Index"] = photon_index_list
        self.nearby_sources_table["Nh"] = nh_list
        
        return photon_index_list
    
    
    def dictionary_model(self) -> Dict[str, Dict[str, Union[str, float]]]:
        """
        Creates a dictionary containing the model parameters for each source in the eRosita catalog's nearby sources table.

        This method compiles a dictionary where each entry corresponds to a source from the nearby sources table. The 
        dictionary for each source includes its astrophysical model type (such as 'power'), model values (like the photon index), 
        observed flux, and column density (Nh).

        Returns:
        Dict[str, Dict[str, Union[str, float]]]: A dictionary where each key represents a source and the value is a 
        dictionary containing the source's modeling parameters.

        The current implementation primarily handles the 'power' model type. Placeholders exist for 'black_body' and 'temp' types, 
        indicating potential future expansion. The method assumes that 'Photon Index' and 'Nh' values are already computed and 
        available in the nearby sources table.
        """
        model_dictionary = {}
        number_source = len(self.nearby_sources_table)

        model = np.array([], dtype=str)
        model_value = np.array([], dtype=float)
        flux_obs = dict_cat.dictionary_catalog["eRosita"]["flux_obs"]
        
        swi_flux = np.array([self.nearby_sources_table[flux_obs][item] for item in range(number_source)], dtype=float)
        nh_value = np.array([self.nearby_sources_table["Nh"][item] for item in range(number_source)], dtype=float)
        
        # Pour le moment seulement 'power' indiquant le modèle a utiliser pour la commande pimms
        for item in range(number_source):
            model = np.append(model, 'power')    
            
        for item in range(number_source):
            if model[item] == 'power':
                model_value = np.append(model_value, self.nearby_sources_table["Photon Index"][item])
            elif model[item] == 'black_body':
                pass # Pas de valeur pour le moment...
            elif model[item] == 'temp':
                pass # Pas de valeur pour le moment... (dernier model pimms)

        for item in range(number_source):

            dictionary = {
                "model": model[item],
                "model_value": model_value[item],
                "flux": swi_flux[item],
                "column_dentsity": nh_value[item]
            }

            model_dictionary[f"src_{item}"] = dictionary
            
        return model_dictionary 

    
class CompareCatalog:
    """
    A class to compare and analyze astronomical catalogs, with a focus on calculating photon index and nh values for sources within these catalogs.

    Attributes:
        catalog_path (List[str]): Paths to the two catalogs being compared.
        radius (float): The radius around the object position to consider for analysis.
        simulation_data (dict): A dictionary containing simulation data, including object data, operating system information, and telescope data.
        exp_time (int): The exposure time for the analysis.
        nearby_sources_table_1 (Table): Table containing nearby sources from the first catalog.
        nearby_sources_table_2 (Table): Table containing nearby sources from the second catalog.
        nearby_sources_position_1 (SkyCoord): Sky coordinates of nearby sources from the first catalog.
        nearby_sources_position_2 (SkyCoord): Sky coordinates of nearby sources from the second catalog.
        index_table (List): List containing indices for variable sources in catalogs.
        vignet_data_1 (List): List of vignetting-related data for the first catalog.
        vignet_data_2 (List): List of vignetting-related data for the second catalog.
        var_index_1 (List): List of indices for variable sources in the first catalog.
        var_index_2 (List): List of indices for variable sources in the second catalog.
        count_rates_1 (List): Count rates for the first catalog.
        count_rates_2 (List): Count rates for the second catalog.
        vector_dictionary_1 (Dict): Dictionary containing vector data for the first catalog.
        vector_dictionary_2 (Dict): Dictionary containing vector data for the second catalog.
        OptimalPointingIdx_1 (int): Optimal pointing index for the first catalog.
        OptimalPointingIdx_2 (int): Optimal pointing index for the second catalog.
        master_source_path (str): Path to the master source file.
        total_spectra_1 (List): Total spectra information for the first catalog.
        total_spectra_2 (List): Total spectra information for the second catalog.
        total_var_spectra_1 (List): Total variable spectra information for the first catalog.
        total_var_spectra_2 (List): Total variable spectra information for the second catalog.
        instrument (Instrument): Instrument data for the analysis.

    Methods:
        open_catalog: Opens and processes catalogs based on provided keys and paths.
        find_nearby_sources: Finds nearby sources based on object position, radius, and catalog data.
        photon_index_nh_for_xmm: Calculates photon index and nh values for the XMM catalog.
        photon_index_nh_for_csc: Calculates photon index and nh values for the Chandra catalog.
        photon_index_nh_for_other_catalog: Calculates photon index and nh values for other catalogs like Swift and eRosita.
        neighbourhood_of_object: Plots the neighborhood of an object based on catalog data.
        dictionary_model: Creates a dictionary model for source analysis.
        count_rate: Calculates count rates for sources in the catalogs.
        xslx_to_py: Converts xlsx data to Python format for further processing.
        calculate_opti_point: Calculates optimal pointing positions for telescopes.
        variability_table: Creates and processes a variability table for sources.
        variability_index: Identifies variable sources in the catalogs.
        write_fits_table: Writes data to a FITS table for storage and further analysis.
        modeling_source_spectra: Models source spectra based on catalog data.
        total_spectra_plot: Plots total spectra for sources in the catalogs.
        write_txt_file: Writes data to a text file for storage and documentation.
    """
   
    def __init__(self, catalog_path: List, radius, simulation_data: dict, exp_time: int) -> None:
        """
        Initializes the CompareCatalog class with necessary parameters for catalog comparison and analysis.

        This method processes two astronomical catalogs, computes photon index and nh values, and prepares data for further analysis, including generating tables of nearby sources, calculating vignetting factors, and modeling source spectra.

        Parameters:
            catalog_path (List[str]): A list containing paths and keys for two catalogs to be compared. The list should contain four elements: path_1, path_2, key_1, and key_2, where path_x is the file path and key_x is the catalog key.
            radius (float): The radius around the object position to consider for analysis, typically in arcminutes.
            simulation_data (dict): A dictionary containing simulation data, including object data, operating system information, and telescope data.
            exp_time (int): The exposure time used in the analysis.

        The method performs several steps:
        - Opens and processes the provided catalogs based on the keys and paths.
        - Finds nearby sources from both catalogs within the specified radius.
        - Calculates photon index and nh values for Xmm_DR13, CS_Chandra, Swift, and eRosita catalogs.
        - Calculates the optimal pointing position for telescopic observations.
        - Computes the vignetting factor for the sources in each catalog.
        - Identifies variable sources in the catalogs and generates a master source path.
        - Writes FITS tables and text files for the analyzed data.
        - Models the source spectra and plots the total spectra for visualization.

        The method also handles different operating systems for specific functionalities.
        """
        
        path_1, path_2 = catalog_path[0], catalog_path[1]
        key_1, key_2 = catalog_path[2], catalog_path[3]

        table_1, table_2 = self.open_catalog(key=(key_1, key_2), path=(path_1, path_2), radius=radius, object_data=simulation_data["object_data"])
        self.nearby_sources_table_1, self.nearby_sources_table_2, self.nearby_sources_position_1, self.nearby_sources_position_2 = self.find_nearby_sources(table=(table_1, table_2), radius=radius, simulation_data=simulation_data, key=(key_1, key_2))

        # ---------- Xmm_DR13 photon index and nh value --------- #
        if "Xmm_DR13" in [key_1, key_2]:
            print(f"\nCalculation of photon index and nh value for {colored('Xmm DR13 catalog', 'yellow')}.")
            #TODO add method to find variable sources in xmm 2 athena catalog with index_table
            xmm_index = [key_1, key_2].index("Xmm_DR13")
            if xmm_index == 0:
                self.nearby_sources_table_1, self.index_table = self.photon_index_nh_for_xmm(os_dictionary=simulation_data["os_dictionary"], xmm_index=xmm_index)
                vignet_data_1 = ["SC_RA", "SC_DEC", "IAUNAME"]
            elif xmm_index == 1:
                self.nearby_sources_table_2, self.index_table = self.photon_index_nh_for_xmm(os_dictionary=simulation_data["os_dictionary"], xmm_index=xmm_index)
                vignet_data_2 = ["SC_RA", "SC_DEC", "IAUNAME"]

        # ---------- CS_Chandra photon index and nh value ---------- #
        if "CSC_2.0" in [key_1, key_2]:
            print(f"\nCalculation of photon index and nh value for {colored('Chandra cone search catalog', 'yellow')}.")
            csc_index = [key_1, key_2].index("CSC_2.0")
            #TODO add method to find variable sources in chandra cone search catalog
            if csc_index == 0:
                self.nearby_sources_table_1 = self.threshold(cone_search_catalog=self.nearby_sources_table_1)
                self.nearby_sources_table_1 = self.photon_index_nh_for_csc(csc_index=csc_index)
                vignet_data_1 = ["ra", "dec", "name"]
            elif csc_index == 1:
                self.nearby_sources_table_2 = self.threshold(cone_search_catalog=self.nearby_sources_table_2)
                self.nearby_sources_table_2 = self.photon_index_nh_for_csc(csc_index=csc_index)
                vignet_data_2 = ["ra", "dec", "name"]

        # ---------- Swift photon index and nh value ---------- #
        if "Swift" in [key_1, key_2]:
            print(f"\nCalculation of photon index and nh value for {colored('Swift catalog', 'yellow')}.")
            swi_index = [key_1, key_2].index("Swift")
            if swi_index == 0:
                self.nearby_sources_table_1 = self.photon_index_nh_for_other_catalog(key="Swift", table=self.nearby_sources_table_1)
                vignet_data_1 = ["RA", "DEC", "Swift_IAUNAME"]
            if swi_index == 1:
                self.nearby_sources_table_2 = self.photon_index_nh_for_other_catalog(key="Swift", table=self.nearby_sources_table_2)
                vignet_data_2 = ["RA", "DEC", "Swift_IAUNAME"]

        # ---------- eRosita photon index and nh value ---------- #
        if "eRosita" in [key_1, key_2]:
            print(f"\nCalculation of photon index and nh value for {colored('eRosita catalog', 'yellow')}.")
            ero_index = [key_1, key_2].index("eRosita")
            if ero_index == 0:
                self.nearby_sources_table_1 = self.photon_index_nh_for_other_catalog(key="eRosita", table=self.nearby_sources_table_1)
                vignet_data_1 = ["RA", "DEC", "eRosita_IAUNAME"]
            if ero_index == 1:
                self.nearby_sources_table_2 = self.photon_index_nh_for_other_catalog(key="eRosita", table=self.nearby_sources_table_2)
                vignet_data_2 = ["RA", "DEC", "eRosita_IAUNAME"]

        self.neighbourhood_of_object(key=(key_1, key_2), simulation_data=simulation_data, radius=radius)
        self.model_dictionary_1, self.model_dictionary_2 = self.dictionary_model(key=(key_1, key_2))

        if platform.system() == "Linux" or platform.system() == "Darwin":
            self.count_rates_1, self.count_rates_2 = self.count_rate()
        elif platform.system() == "Windows":
            self.count_rates_1, self.nearby_sources_table_1 = self.xslx_to_py(args=key_1, table=self.nearby_sources_table_1, simulation_data=simulation_data, radius=radius.value)
            self.count_rates_2, self.nearby_sources_table_2 = self.xslx_to_py(args=key_2, table=self.nearby_sources_table_2, simulation_data=simulation_data, radius=radius.value)


        self.vector_dictionary_1, self.vector_dictionary_2, self.OptimalPointingIdx_1, self.OptimalPointingIdx_2 = self.calculate_opti_point(simulation_data=simulation_data, key=(key_1, key_2))


        self.vignetting_factor_1, self.nearby_sources_table_1 = f.vignetting_factor(OptimalPointingIdx=self.OptimalPointingIdx_1, vector_dictionary=self.vector_dictionary_1, simulation_data=simulation_data, data=vignet_data_1, nearby_sources_table=self.nearby_sources_table_1)
        self.vignetting_factor_2, self.nearby_sources_table_2 = f.vignetting_factor(OptimalPointingIdx=self.OptimalPointingIdx_2, vector_dictionary=self.vector_dictionary_2, simulation_data=simulation_data, data=vignet_data_2, nearby_sources_table=self.nearby_sources_table_2)

        self.master_source_path = self.variability_table(simulation_data=simulation_data, radius=radius.value)
        
        print(f"{colored(f'Find variable index for {key_1} catalog', 'blue')}")
        self.var_index_1 = self.variability_index(key=key_1, iauname=vignet_data_1[2], nearby_sources_table=self.nearby_sources_table_1)
        print(f"{colored(f'Find variable index for {key_2} catalog', 'blue')}")
        self.var_index_2 = self.variability_index(key=key_2, iauname=vignet_data_2[2], nearby_sources_table=self.nearby_sources_table_2)
        
        # ---------- fits table ---------- #
        self.write_fits_table(table=self.nearby_sources_table_1, key=key_1, os_dictionary=simulation_data["os_dictionary"])
        self.write_fits_table(table=self.nearby_sources_table_2, key=key_2, os_dictionary=simulation_data["os_dictionary"])
        # -------------------------------- #
        
        self.total_spectra_1, self.total_spectra_2, self.total_var_spectra_1, self.total_var_spectra_2, self.instrument = self.modeling_source_spectra(simulation_data=simulation_data, exp_time=exp_time, key=(key_1, key_2))
        self.data_1, self.data_2 = self.total_spectra_plot(simulation_data=simulation_data, radius=radius.value, key=(key_1, key_2))
        
        self.write_txt_file(simulation_data=simulation_data, data_1=self.data_1, data_2=self.data_2, key=(key_1, key_2))
        
    
    def open_catalog(self, key: Tuple, path:Tuple, radius, object_data: Dict) -> Tuple[Table, Table]:
        """
        Opens and processes astronomical catalogs for further analysis.

        This method is responsible for loading data from specified catalogs using given keys and paths. It handles different catalogs by checking the keys and applying appropriate procedures to load and convert the data into a usable format.

        Parameters:
            key (Tuple[str, str]): A tuple containing the keys for the catalogs to be opened. The keys are used to identify the specific catalogs and the corresponding procedures for loading the data.
            path (Tuple[str, str]): A tuple containing the paths to the files of the catalogs to be opened.
            radius (float): The radius around the object's position, used in catalog searching, typically in arcminutes.
            object_data (Dict): A dictionary containing information about the object of interest, including its name.

        Returns:
            Tuple[Table, Table]: A tuple of two `Table` objects containing the data from the opened catalogs. 

        The method performs the following operations:
        - If one of the catalogs is "CSC_2.0", it uses the Chandra Source Catalog cone search service to find sources within the specified radius of the object's position.
        - Opens FITS files for the specified paths to load data for other catalogs.
        - Converts the loaded data into `Table` objects for further analysis.
        - Prints a confirmation once the catalogs are loaded successfully.

        This method simplifies the process of accessing astronomical data from various sources, making it easier to conduct comparative analyses between different catalogs.
        """
        
        if key[0] == "CSC_2.0":
            cone = vo.dal.SCSService('http://cda.cfa.harvard.edu/csc2scs/coneSearch') 
            name = SkyCoord.from_name(object_data['object_name'])
            cone_search_catalog = cone.search(pos=name, radius=radius, verbosity=3)
            table_1 = cone_search_catalog.to_table()
            self.cone_search_table = table_1
            
            with fits.open(path[1]) as data:
                table_2 = Table(data[1].data)
                
            print(f"{key[0]} and {key[1]} catalog are loaded !")
            
        elif key[1] == "CSC_2.0":
            cone = vo.dal.SCSService('http://cda.cfa.harvard.edu/csc2scs/coneSearch') 
            name = SkyCoord.from_name(object_data['object_name'])
            cone_search_catalog = cone.search(pos=name, radius=radius, verbosity=3)
            table_2 = cone_search_catalog.to_table()
            self.cone_search_table = table_2
            
            with fits.open(path[0]) as data:
                table_1 = Table(data[1].data)
                
            print(f"{key[0]} and {key[1]} catalog are loaded !")
        
        else:
            with fits.open(path[0]) as data_1, fits.open(path[1]) as data_2:
                table_1 = Table(data_1[1].data)
                table_2 = Table(data_2[1].data)
                
            print(f"{key[0]} and {key[1]} catalog are loaded !")
                
        return table_1, table_2
    
    
    def find_nearby_sources(self, table, radius, simulation_data, key: Tuple) -> Tuple[Table, Table, SkyCoord, SkyCoord]:
        """
        Identifies and processes nearby sources from astronomical catalogs based on a specified object's position.

        Parameters:
            table (Tuple[Table, Table]): A tuple containing two `Table` objects that hold the data from the catalogs to be processed.
            radius (float): The radius around the object's position to consider for finding nearby sources, typically in arcminutes.
            simulation_data (dict): A dictionary containing simulation data, including the object's position and other relevant information.
            key (Tuple[str, str]): A tuple containing the keys identifying the specific catalogs to be processed.

        Returns:
            Tuple[Table, Table, SkyCoord, SkyCoord]: A tuple containing two `Table` objects with the nearby sources from each catalog and two `SkyCoord` objects representing the positions of these sources.

        The method performs several steps:
        - Determines the field of view based on the specified radius and the object's position.
        - Filters the sources in each catalog within the field of view and closer than the specified radius to the object's position.
        - Handles different combinations of catalogs (e.g., CSC_2.0, Xmm_DR13, Swift, eRosita) and applies specific procedures for each.
        - Calculates sky coordinates for the nearby sources.
        - Returns the processed tables and coordinates for further analysis.

        This method is crucial for narrowing down the focus to sources in the vicinity of a specified object, enabling detailed analysis and comparison of these sources across different catalogs.
        """
        
        object_data = simulation_data["object_data"]
        object_position = object_data['object_position']
        object_name = object_data["object_name"]
        
        table_1, table_2 = table
        
        field_of_view = radius + 5*u.arcmin
        min_ra, max_ra = object_position.ra - field_of_view, object_position.ra + field_of_view
        min_dec, max_dec = object_position.dec - field_of_view, object_position.dec + field_of_view
        
        if key[0] == "CSC_2.0":
            if key [1] == "Xmm_DR13":
                # ---------- Xmm catalog ---------- #
                small_table_2 = Table(names=table_2.colnames,
                                      dtype=table_2.dtype)
                nearby_sources_table_2 = Table(names=table_2.colnames,
                                               dtype=table_2.dtype)
                
                print(fr"{colored('Reducing Xmm catalog...', 'yellow')} to fov of {radius.value + 5 } arcmin")
                for number in tqdm(range(len(table_2))):
                    if min_ra/u.deg < table_2["SC_RA"][number] < max_ra/u.deg and min_dec/u.deg < table_2["SC_DEC"][number] < max_dec/u.deg:
                        small_table_2.add_row(table_2[number])
                        
                src_position = SkyCoord(ra=small_table_2['SC_RA'], dec=small_table_2['SC_DEC'], unit=u.deg)

                print(f"{colored(f'Find sources close to {object_name} with Chandra catalog', 'blue')}")
                for number in tqdm(range(len(small_table_2))):
                    if f.ang_separation(object_position, src_position[number]) < radius:
                        nearby_sources_table_2.add_row(small_table_2[number])
                nearby_sources_position_2 = SkyCoord(ra=nearby_sources_table_2["SC_RA"], dec=nearby_sources_table_2["SC_DEC"], unit=u.deg)
                
                # ---------- Chandra cone search catalog ---------- #
                nearby_sources_table_1 = self.cone_search_table
                nearby_sources_position_1 = SkyCoord(ra=nearby_sources_table_1["ra"], dec=nearby_sources_table_1["dec"], unit=u.deg)
                
                print(f"{len(nearby_sources_table_1)} sources was detected with Chandra catalog and {len(nearby_sources_table_2)} sources with Xmm catalog close to {object_name}")
                return nearby_sources_table_1, nearby_sources_table_2, nearby_sources_position_1, nearby_sources_position_2
            
            else:
                # ---------- catalog Swift or eRosita ---------- #
                small_table_2 = Table(names=table_2.colnames,
                                      dtype=table_2.dtype)
                not_unique_table = Table(names=table_2.colnames,
                                               dtype=table_2.dtype)
                
                print(f"{colored(f'Reducing {key[1]} catalog...', 'yellow')} to fov of {radius.value + 5 } arcmin")
                for number in tqdm(range(len(table_2))):
                    if min_ra/u.deg < table_2["RA"][number] < max_ra/u.deg and min_dec/u.deg < table_2["DEC"][number] < max_dec/u.deg:
                        small_table_2.add_row(table_2[number])
                        
                src_position = SkyCoord(ra=small_table_2['RA'], dec=small_table_2['DEC'], unit=u.deg)

                print(f"{colored(f'Find sources close to {object_name} with {key[1]} catalog', 'blue')}")
                for number in tqdm(range(len(small_table_2))):
                    if f.ang_separation(object_position, src_position[number]) < radius:
                        not_unique_table.add_row(small_table_2[number])
                
                if len(not_unique_table) != 0:
                    if key[1] == "Swift":
                        column_name = {"source_name": "Swift_IAUNAME",
                                       "right_ascension": "RA",
                                       "declination": "DEC",
                                       "catalog_name": "Swift"}
                    elif key[1] == "eRosita":
                        column_name = {"source_name": "eRosita_IAUNAME",
                                       "right_ascension": "RA",
                                       "declination": "DEC",
                                       "catalog_name": "eRosita"}

                    nearby_sources_table_2 = f.create_unique_sources_catalog(nearby_sources_table=not_unique_table, column_name=column_name)
                    nearby_sources_position_2 = SkyCoord(ra=nearby_sources_table_2["RA"], dec=nearby_sources_table_2["DEC"], unit=u.deg)
                    
                    # ---------- Chandra cone search catalog ---------- #
                    nearby_sources_table_1 = self.cone_search_table
                    nearby_sources_position_1 = SkyCoord(ra=nearby_sources_table_1["ra"], dec=nearby_sources_table_1["dec"], unit=u.deg)
                    
                    print(f"{len(nearby_sources_table_1)} sources was detected with Chandra catalog and {len(nearby_sources_table_2)} sources with {key[1]} catalog close to {object_name}")
                    return nearby_sources_table_1, nearby_sources_table_2, nearby_sources_position_1, nearby_sources_position_2
                else:
                    print(f"No sources detected in {colored(key[1], 'red')} catalog !")
                    sys.exit()
                
        elif key[1] == "CSC_2.0":
            if key[0] == "Xmm_DR13":
                # ---------- Xmm catalog ---------- #
                small_table_1 = Table(names=table_1.colnames,
                                      dtype=table_1.dtype)
                nearby_sources_table_1 = Table(names=table_1.colnames,
                                               dtype=table_1.dtype)
                
                print(fr"{colored(f'Reducing {key[0]} catalog...', 'yellow')} to fov of {radius.value + 5 } arcmin")
                for number in tqdm(range(len(table_1))):
                    if min_ra/u.deg < table_1["SC_RA"][number] < max_ra/u.deg and min_dec/u.deg < table_1["SC_DEC"][number] < max_dec/u.deg:
                        small_table_1.add_row(table_1[number])
                        
                src_position = SkyCoord(ra=small_table_1['SC_RA'], dec=small_table_1['SC_DEC'], unit=u.deg)

                print(f"{colored(f'Find sources close to {object_name} with {key[0]} catalog', 'blue')}")
                for number in tqdm(range(len(small_table_1))):
                    if f.ang_separation(object_position, src_position[number]) < radius:
                        nearby_sources_table_1.add_row(small_table_1[number])
                nearby_sources_position_1 = SkyCoord(ra=nearby_sources_table_1["SC_RA"], dec=nearby_sources_table_1["SC_DEC"], unit=u.deg)
                
                # ---------- Chandra cone search catalog ---------- #
                nearby_sources_table_2 = self.cone_search_table
                nearby_sources_position_2 = SkyCoord(ra=nearby_sources_table_2["ra"], dec=nearby_sources_table_2["dec"], unit=u.deg)
                
                print(f"{len(nearby_sources_table_1)} sources was detected with Xmm catalog and {len(nearby_sources_table_2)} sources with Chandra catalog close to {object_name}")
                return nearby_sources_table_1, nearby_sources_table_2, nearby_sources_position_1, nearby_sources_position_2
                
            else:
                # ---------- catalog Swift or eRosita ---------- #
                small_table_1 = Table(names=table_1.colnames,
                                      dtype=table_1.dtype)
                not_unique_table = Table(names=table_1.colnames,
                                         dtype=table_1.dtype)
                
                print(f"{colored(f'Reducing {key[0]} catalog...', 'yellow')} to fov of {radius.value + 5 } arcmin")
                for number in tqdm(range(len(table_1))):
                    if min_ra/u.deg < table_1["RA"][number] < max_ra/u.deg and min_dec/u.deg < table_1["DEC"][number] < max_dec/u.deg:
                        small_table_1.add_row(table_1[number])
                        
                src_position = SkyCoord(ra=small_table_1["RA"], dec=small_table_1["DEC"], unit=u.deg)

                print(f"{colored(f'Find sources close to {object_name} with {key[0]} catalog', 'blue')}")
                for number in tqdm(range(len(small_table_1))):
                    if f.ang_separation(object_position, src_position[number]) < radius:
                        not_unique_table.add_row(small_table_1[number])
                if len(not_unique_table) != 0:
                    if key[0] == "Swift":
                        column_name = {"source_name": "Swift_IAUNAME",
                                       "right_ascension": "RA",
                                       "declination": "DEC",
                                       "catalog_name": "Swift"}
                    elif key[0] == "eRosita":
                        column_name = {"source_name": "eRosita_IAUNAME",
                                       "right_ascension": "RA",
                                       "declination": "DEC",
                                       "catalog_name": "eRosita"}
                    
                    nearby_sources_table_1 = f.create_unique_sources_catalog(nearby_sources_table=not_unique_table, column_name=column_name)
                    nearby_sources_position_1 = SkyCoord(ra=nearby_sources_table_1["RA"], dec=nearby_sources_table_1["DEC"], unit=u.deg)
                    
                    # ---------- Chandra cone search catalog ---------- #
                    nearby_sources_table_2 = self.cone_search_table
                    nearby_sources_position_2 = SkyCoord(ra=nearby_sources_table_2["ra"], dec=nearby_sources_table_2["dec"], unit=u.deg)
                    
                    print(f"{len(nearby_sources_table_1)} sources was detected with {key[0]} catalog and {len(nearby_sources_table_2)} sources with Chandra catalog close to {object_name}")
                    return nearby_sources_table_1, nearby_sources_table_2, nearby_sources_position_1, nearby_sources_position_2
                else:
                    print(f"No sources detected in {colored(key[0], 'red')} catalog !")
                    sys.exit()
        else:
            if key[0] == "Xmm_DR13" :
                # ---------- Xmm catalog ---------- #
                small_table_1 = Table(names=table_1.colnames,
                                      dtype=table_1.dtype)
                nearby_sources_table_1 = Table(names=table_1.colnames,
                                               dtype=table_1.dtype)
                
                # On gère ici la première table dans le cas ou elle est Xmm_DR13
                print(f"{colored(f'Reducing {key[0]} catalog...', 'yellow')} to fov of {radius.value + 5 } arcmin")
                for number in tqdm(range(len(table_1))):
                    if min_ra/u.deg < table_1["SC_RA"][number] < max_ra/u.deg and min_dec/u.deg < table_1["SC_DEC"][number] < max_dec/u.deg:
                        small_table_1.add_row(table_1[number])
                        
                src_position = SkyCoord(ra=small_table_1["SC_RA"], dec=small_table_1["SC_DEC"], unit=u.deg)

                print(f"{colored(f'Find sources close to {object_name} with {key[0]} catalog', 'blue')}")
                for number in tqdm(range(len(small_table_1))):
                    if f.ang_separation(object_position, src_position[number]) < radius:
                        nearby_sources_table_1.add_row(small_table_1[number])
                        
                nearby_sources_position_1 = SkyCoord(ra=nearby_sources_table_1["SC_RA"], dec=nearby_sources_table_1["SC_DEC"], unit=u.deg)
                
                # ---------- catalog Swift or eRosita ---------- #
                
                small_table_2 = Table(names=table_2.colnames,
                                      dtype=table_2.dtype)
                not_unique_table = Table(names=table_2.colnames,
                                         dtype=table_2.dtype)
                
                print(f"{colored(f'Reducing {key[1]} catalog...', 'yellow')} to fov of {radius.value + 5 } arcmin")
                for number in tqdm(range(len(table_2))):
                    if min_ra/u.deg < table_2["RA"][number] < max_ra/u.deg and min_dec/u.deg < table_2["DEC"][number] < max_dec/u.deg:
                        small_table_2.add_row(table_2[number])
                        
                src_position = SkyCoord(ra=small_table_2["RA"], dec=small_table_2["DEC"], unit=u.deg)

                print(f"{colored(f'Find sources close to {object_name} with {key[1]} catalog', 'blue')}")
                for number in tqdm(range(len(small_table_2))):
                    if f.ang_separation(object_position, src_position[number]) < radius:
                        not_unique_table.add_row(small_table_2[number])
                if len(not_unique_table) != 0:
                    if key[1] == "Swift":
                        column_name = {"source_name": "Swift_IAUNAME",
                                       "right_ascension": "RA",
                                       "declination": "DEC",
                                       "catalog_name": "Swift"}
                    elif key[1] == "eRosita":
                        column_name = {"source_name": "eRosita_IAUNAME",
                                       "right_ascension": "RA",
                                       "declination": "DEC",
                                       "catalog_name": "eRosita"}
                                    
                    nearby_sources_table_2 = f.create_unique_sources_catalog(nearby_sources_table=not_unique_table, column_name=column_name)
                    nearby_sources_position_2 = SkyCoord(ra=nearby_sources_table_2["RA"], dec=nearby_sources_table_2["DEC"], unit=u.deg)
                    
                    print(f"{len(nearby_sources_table_1)} sources was detected with {key[0]} catalog and {len(nearby_sources_table_2)} sources with {key[1]} catalog close to {object_name}")
                    return nearby_sources_table_1, nearby_sources_table_2, nearby_sources_position_1, nearby_sources_position_2
                else:
                    print(f"No sources detected in {colored(key[1], 'red')} catalog !")
                    sys.exit()
                
            elif key[1] == "Xmm_DR13":
                # ---------- catalog Swift or eRosita ---------- #
                small_table_1 = Table(names=table_1.colnames,
                                      dtype=table_1.dtype)
                not_unique_table = Table(names=table_1.colnames,
                                         dtype=table_1.dtype)
                
                print(f"{colored(f'Reducing {key[0]} catalog...', 'yellow')} to fov of {radius.value + 5 } arcmin")
                for number in tqdm(range(len(table_1))):
                    if min_ra/u.deg < table_1["RA"][number] < max_ra/u.deg and min_dec/u.deg < table_1["DEC"][number] < max_dec/u.deg:
                        small_table_1.add_row(table_1[number])
                        
                src_position = SkyCoord(ra=small_table_1["RA"], dec=small_table_1["DEC"], unit=u.deg)

                print(f"{colored(f'Find sources close to {object_name} with {key[0]} catalog', 'blue')}")
                for number in tqdm(range(len(small_table_1))):
                    if f.ang_separation(object_position, src_position[number]) < radius:
                        not_unique_table.add_row(small_table_1[number])
                
                if len(not_unique_table) != 0:
                    if key[0] == "Swift":
                        column_name = {"source_name": "Swift_IAUNAME",
                                       "right_ascension": "RA",
                                       "declination": "DEC",
                                       "catalog_name": "Swift"}
                    elif key[0] == "eRosita":
                        column_name = {"source_name": "eRosita_IAUNAME",
                                       "right_ascension": "RA",
                                       "declination": "DEC",
                                       "catalog_name": "eRosita"}
                    
                    nearby_sources_table_1 = f.create_unique_sources_catalog(nearby_sources_table=not_unique_table, column_name=column_name)
                    nearby_sources_position_1 = SkyCoord(ra=nearby_sources_table_1["RA"], dec=nearby_sources_table_1["DEC"], unit=u.deg)
                    
                    # ---------- Xmm catalog ---------- #
                    
                    small_table_2 = Table(names=table_2.colnames,
                                        dtype=table_2.dtype)
                    nearby_sources_table_2 = Table(names=table_2.colnames,
                                                dtype=table_2.dtype)
                    
                    print(f"{colored(f'Reducing {key[1]} catalog...', 'yellow')} to fov of {radius.value + 5 } arcmin")
                    for number in tqdm(range(len(table_2))):
                        if min_ra/u.deg < table_2["SC_RA"][number] < max_ra/u.deg and min_dec/u.deg < table_2["SC_DEC"][number] < max_dec/u.deg:
                            small_table_2.add_row(table_2[number])
                            
                    src_position = SkyCoord(ra=small_table_2['SC_RA'], dec=small_table_2['SC_DEC'], unit=u.deg)

                    print(f"{colored(f'Find sources close to {object_name} with {key[1]} catalog', 'blue')}")
                    for number in tqdm(range(len(small_table_2))):
                        if f.ang_separation(object_position, src_position[number]) < radius:
                            nearby_sources_table_2.add_row(small_table_2[number])
                    nearby_sources_position_2 = SkyCoord(ra=nearby_sources_table_2["SC_RA"], dec=nearby_sources_table_2["SC_DEC"], unit=u.deg)
                    
                    print(f"{len(nearby_sources_table_1)} sources was detected with {key[0]} catalog and {len(nearby_sources_table_2)} sources with {key[1]} catalog close to {object_name}")
                    return nearby_sources_table_1, nearby_sources_table_2, nearby_sources_position_1, nearby_sources_position_2
                else:
                    print(f"No sources detected in {colored(key[0], 'red')} catalog !")
                    sys.exit()
                
            else:
                # ---------- eRosita or Swift catalog ---------- #
                small_table_1 = Table(names=table_1.colnames,
                                      dtype=table_1.dtype)
                not_unique_table_1 = Table(names=table_1.colnames,
                                           dtype=table_1.dtype)
                
                print(f"{colored(f'Reducing {key[0]} catalog...', 'yellow')} to fov of {radius.value + 5 } arcmin")
                for number in tqdm(range(len(table_1))):
                    if min_ra/u.deg < table_1["RA"][number] < max_ra/u.deg and min_dec/u.deg < table_1["DEC"][number] < max_dec/u.deg:
                        small_table_1.add_row(table_1[number])
                        
                src_position = SkyCoord(ra=small_table_1["RA"], dec=small_table_1["DEC"], unit=u.deg)

                print(f"{colored(f'Find sources close to {object_name} with {key[0]} catalog', 'blue')}")
                for number in tqdm(range(len(small_table_1))):
                    if f.ang_separation(object_position, src_position[number]) < radius:
                        not_unique_table_1.add_row(small_table_1[number])
                        
                if len(not_unique_table_1) != 0:
                    if key[0] == "Swift":
                        column_name = {"source_name": "Swift_IAUNAME",
                                       "right_ascension": "RA",
                                       "declination": "DEC",
                                       "catalog_name": "Swift"}
                    elif key[0] == "eRosita":
                        column_name = {"source_name": "eRosita_IAUNAME",
                                       "right_ascension": "RA",
                                       "declination": "DEC",
                                       "catalog_name": "eRosita"}
                        
                    nearby_sources_table_1 = f.create_unique_sources_catalog(nearby_sources_table=not_unique_table_1, column_name=column_name)
                    nearby_sources_position_1 = SkyCoord(ra=nearby_sources_table_1["RA"], dec=nearby_sources_table_1["DEC"], unit=u.deg)

                    # ---------- eRosita or Swift catalog ---------- #
                    small_table_2 = Table(names=table_2.colnames,
                                        dtype=table_2.dtype)
                    not_unique_table_2 = Table(names=table_2.colnames,
                                            dtype=table_2.dtype)
                    
                    print(f"{colored(f'Reducing {key[1]} catalog...', 'yellow')} to fov of {radius.value + 5 } arcmin")
                    for number in tqdm(range(len(table_2))):
                        if min_ra/u.deg < table_2["RA"][number] < max_ra/u.deg and min_dec/u.deg < table_2["DEC"][number] < max_dec/u.deg:
                            small_table_2.add_row(table_2[number])
                            
                    src_position = SkyCoord(ra=small_table_2["RA"], dec=small_table_2["DEC"], unit=u.deg)

                    print(f"{colored(f'Find sources close to {object_name} with {key[1]} catalog', 'blue')}")
                    for number in tqdm(range(len(small_table_2))):
                        if f.ang_separation(object_position, src_position[number]) < radius:
                            not_unique_table_2.add_row(small_table_2[number])
                    
                    if len(not_unique_table_2) != 0:
                        if key[1] == "Swift":
                            column_name = {"source_name": "Swift_IAUNAME",
                                           "right_ascension": "RA",
                                           "declination": "DEC",
                                           "catalog_name": "Swift"}
                        elif key[1] == "eRosita":
                            column_name = {"source_name": "eRosita_IAUNAME",
                                           "right_ascension": "RA",
                                           "declination": "DEC",
                                           "catalog_name": "eRosita"}
                            
                        nearby_sources_table_2 = f.create_unique_sources_catalog(nearby_sources_table=not_unique_table_2, column_name=column_name)
                        nearby_sources_position_2 = SkyCoord(ra=nearby_sources_table_2["RA"], dec=nearby_sources_table_2["DEC"], unit=u.deg)
                        
                        print(f"{len(nearby_sources_table_1)} sources was detected with {key[0]} catalog and {len(nearby_sources_table_2)} sources with {key[1]} catalog close to {object_name}")
                        return nearby_sources_table_1, nearby_sources_table_2, nearby_sources_position_1, nearby_sources_position_2
                    else:
                        print(f"No sources detected in {colored(key[1], 'red')} catalog !")
                        sys.exit()
                else:
                    print(f"No sources detected in {colored(key[0], 'red')} catalog !")
                    sys.exit()
                    
    
    def optimization(self, index: int, key: str, table: Table) -> Tuple[List, List]:
        """
        Optimizes the parameters of an absorbed power-law model based on the observed fluxes in different energy bands.

        This method is used to fit the observed flux data from astronomical catalogs to an absorbed power-law model, which is commonly used in astrophysics to describe the spectral energy distribution of sources.

        Parameters:
            index (int): The index of the source in the catalog for which the optimization is being performed.
            key (str): The key identifying the catalog (e.g., 'XMM', 'CS_Chandra', 'Swift', 'eRosita') from which the data is taken.
            table (Table): The table containing the observed flux data and other relevant information for the sources.

        Returns:
            Tuple[List, List]: A tuple where the first element is a list containing the optimized parameters and the second element is the absorbed photon index.

        The method performs the following steps:
        - Retrieves the observed fluxes, their errors, energy band centers, and widths from the catalog data based on the given key.
        - Defines an absorbed power-law function to model the source's spectrum.
        - Uses the curve fitting technique to find the best-fit parameters for the absorbed power-law model based on the observed data.
        - Returns the optimized parameters and the absorbed photon index for the specified source.

        This optimization is crucial for understanding the physical processes in astronomical sources by analyzing their energy spectra.
        """
        
        if key == "XMM":
            interp_data = {"band_flux_obs": dict_cat.dictionary_catalog[key]["band_flux_obs"],
                           "band_flux_obs_err": dict_cat.dictionary_catalog[key]["band_flux_obs_err"],
                           "energy_band_center": dict_cat.dictionary_catalog[key]["energy_band_center"],
                           "energy_band_half_width": dict_cat.dictionary_catalog[key]["energy_band_half_width"]}
        
        if key == "CS_Chandra":
            interp_data = {"band_flux_obs": dict_cat.dictionary_catalog[key]["band_flux_obs"],
                           "band_flux_obs_err": dict_cat.dictionary_catalog[key]["band_flux_obs_err"],
                           "energy_band_center": dict_cat.dictionary_catalog[key]["energy_band_center"],
                           "energy_band_half_width": dict_cat.dictionary_catalog[key]["energy_band_half_width"]}
        
        if key == "Swift":
            interp_data = {"band_flux_obs": dict_cat.dictionary_catalog[key]["band_flux_obs"],
                           "band_flux_obs_err": dict_cat.dictionary_catalog[key]["band_flux_obs_err"],
                           "energy_band_center": dict_cat.dictionary_catalog[key]["energy_band_center"],
                           "energy_band_half_width": dict_cat.dictionary_catalog[key]["energy_band_half_width"]}
        
        if key == "eRosita":
            interp_data = {"band_flux_obs": dict_cat.dictionary_catalog[key]["band_flux_obs"],
                           "band_flux_obs_err": dict_cat.dictionary_catalog[key]["band_flux_obs_err"],
                           "energy_band_center": dict_cat.dictionary_catalog[key]["energy_band_center"],
                           "energy_band_half_width": dict_cat.dictionary_catalog[key]["energy_band_half_width"]}

        def absorbed_power_law(energy_band, constant, gamma):
            sigma = np.array(np.linspace(1e-20, 1e-24, len(energy_band)), dtype=float)
            return (constant * energy_band ** (-gamma)) * (np.exp(-sigma * 3e20))
        
        tab_width = 2 * interp_data["energy_band_half_width"]
        
        flux_obs = [table[name][index] for name in interp_data["band_flux_obs"]]
        flux_err = [[table[err_0][index] for err_0 in interp_data["band_flux_obs_err"][0]],
                    [table[err_1][index] for err_1 in interp_data["band_flux_obs_err"][1]]]

        flux_err_obs = [np.mean([flux_err_0, flux_err_1]) for flux_err_0, flux_err_1 in zip(flux_err[0], flux_err[1])]
        
        y_array = [num/det for num, det in zip(flux_obs, tab_width)]
        yerr_array = [num/det for num, det in zip(flux_err_obs, tab_width)]
        
        popt, pcov = curve_fit(absorbed_power_law, interp_data["energy_band_center"], y_array, sigma=yerr_array)
        constant, absorb_pho_index = popt

        optimization_parameters = (interp_data["energy_band_center"], y_array, yerr_array, absorbed_power_law(interp_data["energy_band_center"], *popt))
        
        return optimization_parameters, absorb_pho_index
    
    
    def visualization_interp(self, optimization_parameters, photon_index, key) -> None:
        """
        Visualizes the interpolation of photon indices using an absorbed power-law model.

        This method plots the observed fluxes and the best-fit absorbed power-law model for sources in an astronomical catalog. It helps in understanding the spectral characteristics of these sources.

        Parameters:
            optimization_parameters (List): A list containing the optimization parameters obtained from fitting the absorbed power-law model. It includes energy band centers, observed fluxes, flux errors, and the model function.
            photon_index (List): A list of photon indices corresponding to each source or observation in the catalog.
            key (str): The key identifying the catalog (e.g., 'XMM', 'CS_Chandra', 'Swift', 'eRosita') from which the data is taken.

        The method performs the following steps:
        - Retrieves the energy band centers from the catalog data based on the given key.
        - Creates a plot with energy on the x-axis and flux on the y-axis.
        - Plots error bars for the observed fluxes and overlays the best-fit absorbed power-law model.
        - Annotates the plot with the photon index for each source or observation.
        - Displays the plot with logarithmic axes for better visualization of the spectral data.

        This visualization is important for assessing the quality of the fit and for comparing the spectral properties across different sources or observations.
        """
        
        energy_band = dict_cat.dictionary_catalog[key]["energy_band_center"]
        
        fig, axes = plt.subplots(1, 1, figsize=(15, 8))
        fig.suptitle("Interpolation Photon Index plot", fontsize=20)
        fig.text(0.5, 0.04, 'Energy [keV]', ha='center', va='center')
        fig.text(0.04, 0.5, 'Flux $[erg.cm^{-2}.s^{-1}.keV^{-1}]$', ha='center', va='center', rotation='vertical')
        
        for item in range(len(optimization_parameters)):
            flux_obs = optimization_parameters[item][1]
            flux_obs_err = optimization_parameters[item][2]
            absorbed_power_law = optimization_parameters[item][3]
            absorb_pho_index = photon_index[item]
            
            axes.errorbar(energy_band, flux_obs, flux_obs_err, fmt='*', color='red', ecolor='black')
            axes.plot(energy_band, absorbed_power_law, label=f"$\Gamma$ = {absorb_pho_index:.8f}")
    
        axes.legend(loc="upper left", ncol=4, fontsize=6)
        axes.loglog()
        
        plt.show()
    
    
    def photon_index_nh_for_xmm(self, os_dictionary: Dict, xmm_index: int) -> Tuple[Table, List]:
        """
        Calculates the photon index and hydrogen column density (NH) for sources in the XMM-Newton catalog.

        Parameters:
            os_dictionary (Dict): A dictionary containing paths and other operating system-related information needed to access catalog data.
            xmm_index (int): An index indicating which of the nearby sources tables (1 or 2) corresponds to the XMM-Newton catalog.

        Returns:
            Tuple[Table, List]: A tuple where the first element is the updated table with added 'Photon Index' and 'Nh' columns, and the second element is a table containing indexes mapping nearby sources to the XMM-Newton catalogs.

        The method performs the following steps:
        - Accesses the XMM-Newton DR11 and Athena catalogs using paths provided in the os_dictionary.
        - Matches sources in the nearby sources table with those in the XMM catalogs based on identifiers like 'IAUNAME' and 'DETID'.
        - For each source, calculates the photon index and NH either directly from the Athena catalog or through optimization if data is not found.
        - Visualizes the absorbed power-law fit for the calculated photon indices.
        - Adds the calculated photon indices and NH values to the nearby sources table.
        - Returns the updated nearby sources table and a table mapping the indexes of these sources in the XMM catalogs.

        This method is vital for extracting and computing key astrophysical parameters from the XMM-Newton catalog, aiding in the analysis of X-ray sources.
        """
        
        catalog_datapath = os_dictionary["catalog_datapath"]
        xmm_dr11_path = os.path.join(catalog_datapath, "4XMM_DR11cat_v1.0.fits").replace("\\", "/")
        xmm_2_athena_path = os.path.join(catalog_datapath, "xmm2athena_D6.1_V3.fits").replace("\\", "/")

        while True:
            try:
                with fits.open(xmm_dr11_path) as data:
                    xmm_dr11_table = Table(data[1].data)
                    print(f"{colored('Xmm DR11', 'green')} catalog is loaded.")
                    break
            except Exception as error:
                print(f"{colored('An error occured : ', 'red')} {error}")
                xmm_dr11_path = str(input("Enter a new path for xmm_dr11 catalog : \n"))
        while True:
            try:
                with fits.open(xmm_2_athena_path) as data:
                    xmm_2_athena_table = Table(data[1].data)
                    print(f"{colored('Xmm 2 Athena', 'green')} catalog is loaded.")
                    break
            except Exception as error:
                print(f"{colored('An error occured : ', 'red')} {error}")
                xmm_2_athena_path = str(input("Enter a new path for xmm_2_athena catalog : \n"))

        if xmm_index == 0:
            nearby_sources_table = self.nearby_sources_table_1
        else: 
            nearby_sources_table = self.nearby_sources_table_2

        sources_number = len(nearby_sources_table)
        name_list = nearby_sources_table["IAUNAME"]
        nearby_sources_table_dr11 = Table(names=xmm_dr11_table.colnames,
                                          dtype=xmm_dr11_table.dtype)

        index_dr11 = np.array([], dtype=int)
        for name in name_list:
            if name in xmm_dr11_table["IAUNAME"]:
                index_name = list(xmm_dr11_table["IAUNAME"]).index(name)
                index_dr11 = np.append(index_dr11, index_name)
                nearby_sources_table_dr11.add_row(xmm_dr11_table[index_name])

        index_x2a = np.array([], dtype=int)
        for det_id in nearby_sources_table_dr11["DETID"]:
            if det_id in xmm_2_athena_table["DETID"]:
                index_det_id = list(xmm_2_athena_table["DETID"]).index(det_id)
                index_x2a = np.append(index_x2a, index_det_id)
            else:
                index_x2a = np.append(index_x2a, "No data found")

        column_names = ["Index in nearby_sources_table", "Index in xmm_dr11", "Index in x2a"]
        column_data = [[item for item in range(sources_number)], index_dr11, index_x2a]
        index_table = Table(names=column_names,
                            data=column_data)
        
        nh_list, photon_index_list, optimization_parameters = [], [], []
        for index in range(sources_number):
            if index_table["Index in x2a"][index] != "No data found":
                nh = xmm_2_athena_table["logNH_med"][index]
                nh_list.append(np.exp(nh * np.log(10)))
                photon_index_list.append(xmm_2_athena_table['PhoIndex_med'][index])
            else:
                nh_list.append(3e20)
                params, photon_index = self.optimization(index=index, key="XMM", table=nearby_sources_table)
                photon_index_list.append(photon_index)
                optimization_parameters.append(params)
                
        self.visualization_interp(optimization_parameters=optimization_parameters, photon_index=photon_index_list, key="XMM")
        
        col_names = ["Photon Index", "Nh"]
        col_data = [photon_index_list, nh_list]
        
        for name, data in zip(col_names, col_data):
            nearby_sources_table[name] = data
            
        return nearby_sources_table, index_table
        

    def threshold(self, cone_search_catalog: Table) -> Table:
        """
        Corrects and standardizes flux values in an astronomical catalog, handling missing or undefined data points.

        This method is primarily used to process flux data from the Chandra Source Catalog (CSC), represented by the 'CS_Chandra' key, but can be adapted for similar catalogs.

        Parameters:
            cone_search_catalog (Table): The table containing flux data and other parameters from the CSC or a similar catalog.

        Returns:
            Table: The corrected and standardized table with flux values processed to replace undefined or missing data with meaningful numerical values.

        The method performs the following steps:
        - Iterates over each item in the catalog to check for undefined or missing flux values, represented as `np.ma.core.MaskedConstant`.
        - Replaces missing values in observed flux (`flux_obs`) and flux error (`flux_obs_err`) fields with the minimum numerical value found in the respective field.
        - Processes other flux-related fields (like `flux_powlaw_aper_b`, `flux_powlaw_aper__s/m/h`, and their error limits) in a similar manner.
        - Ensures that all flux-related fields in the catalog have consistent and usable numerical values, facilitating further analysis.

        This method is crucial for preparing astronomical data for analysis, ensuring that all flux-related fields are consistent and numerically valid, especially important in catalogs where missing or undefined values are common.
        """
        
        source_number = len(cone_search_catalog)
        key = "CS_Chandra"

        # name : flux_powlaw_aper_b
        # algo to replace -- by the min numerical value of the list
        flux_obs = dict_cat.dictionary_catalog[key]["flux_obs"]
        flux_data = []
        for item in range(source_number):
            # iterate for each item in the list
            if not isinstance(cone_search_catalog[flux_obs][item], np.ma.core.MaskedConstant):
                # put in flux_data list each item of type different to np.ma.core.MaskedConstant
                flux_data.append(cone_search_catalog[flux_obs][item])
        flux = list(cone_search_catalog[flux_obs])
        corrected_flux_obs = np.nan_to_num(flux, nan=np.min(flux_data))
        cone_search_catalog[flux_obs] = corrected_flux_obs

        for name in dict_cat.dictionary_catalog[key]["flux_obs_err"]:
            flux_err = []
            for item in range(source_number):
                # iterate for each item in the list
                if not isinstance(cone_search_catalog[name][item], np.ma.core.MaskedConstant):
                    # put in flux_err list each item of type different to np.ma.core.MaskedConstant
                    flux_err.append(cone_search_catalog[name][item])
            flux_err_obs = list(cone_search_catalog[name])
            corrected_flux_err_obs = np.nan_to_num(flux_err_obs, nan=np.min(flux_err))
            cone_search_catalog[name] = corrected_flux_err_obs

        # name : flux_powlaw_aper__s/m/h
        # algo to replace -- by the min numerical value of the list
        for name in dict_cat.dictionary_catalog[key]["band_flux_obs"]:
            # itera name in band_flux_obs 
            data = []
            for item in range(source_number):
                # iterate for each item in the list
                if not isinstance(cone_search_catalog[name][item], np.ma.core.MaskedConstant):
                    # put in data list each item of type different to np.ma.core.MaskedConstant
                    data.append(cone_search_catalog[name][item])
            flux = list(cone_search_catalog[name])
            corrected_flux = np.nan_to_num(flux, nan=np.min(data))
            cone_search_catalog[name] = corrected_flux


        # name : flux_powlaw_aper_lo/hi_lim_s/m/h
        err_flux_neg, err_flux_pos = dict_cat.dictionary_catalog[key]["band_flux_obs_err"][0], dict_cat.dictionary_catalog[key]["band_flux_obs_err"][1]
        # algo to replace -- by the min numerical value of the list
        for err_name_0, err_name_1 in zip(err_flux_neg, err_flux_pos):
            neg_data, pos_data = [], []
            for item in range(source_number):
                # iterate for each item in the list
                if not isinstance(cone_search_catalog[err_name_0][item], np.ma.core.MaskedConstant):
                    # put in neg_data list each item of type different to np.ma.core.MaskedConstant
                    neg_data.append(cone_search_catalog[err_name_0][item])
                if not isinstance(cone_search_catalog[err_name_1][item], np.ma.core.MaskedConstant):
                    # put in pos_data list each item of type different to np.ma.core.MaskedConstant
                    pos_data.append(cone_search_catalog[err_name_1][item])
                    
            neg_flux, pos_flux = list(cone_search_catalog[err_name_0]), list(cone_search_catalog[err_name_1])
            corrected_neg_flux = np.nan_to_num(neg_flux, nan=np.min(neg_data))
            corrected_pos_flux = np.nan_to_num(pos_flux, nan=np.min(pos_data))
            
            cone_search_catalog[err_name_0] = corrected_neg_flux
            cone_search_catalog[err_name_1] = corrected_pos_flux
            
        return cone_search_catalog


    def photon_index_nh_for_csc(self, csc_index: int) -> Table:
        """
        Calculates and updates the photon index and hydrogen column density (NH) for sources in the Chandra Source Catalog (CSC).

        Parameters:
            csc_index (int): An index indicating which of the nearby sources tables (1 or 2) corresponds to the CSC catalog.

        Returns:
            Table: The updated nearby sources table with new columns for 'Photon Index' and 'Nh' added.

        The method performs the following steps:
        - Determines the appropriate nearby sources table based on the csc_index.
        - Iterates through each source in the table. For each source:
            - If a valid photon index is already present, it is used; otherwise, it's calculated using the `optimization` method.
            - If a valid NH value is present, it is used; otherwise, a default value is assigned.
        - Each source's photon index and NH value are added to the photon_index_list and nh_list, respectively.
        - Calls the `visualization_interp` method to visualize the interpolated photon indices.
        - The 'Photon Index' and 'Nh' columns are added or updated in the nearby sources table with the calculated values.
        - Returns the updated nearby sources table.

        This method is crucial for enhancing the CSC data by calculating key astrophysical parameters (photon index and NH) that are essential for analyzing the X-ray spectra of astronomical sources.
        """
        
        if csc_index == 0:
            nearby_sources_table = self.nearby_sources_table_1
        else:
            nearby_sources_table = self.nearby_sources_table_2
            
        photon_index_list, nh_list, optimization_parameters = [], [], []
        
        for (index, photon), nh_value in zip(enumerate(nearby_sources_table["powlaw_gamma"]), nearby_sources_table["nh_gal"]):
            if photon != 0:
                photon_index_list.append(photon)
            else:
                params, photon_index = self.optimization(index=index, key="CS_Chandra", table=nearby_sources_table)
                photon_index_list.append(photon_index if photon_index > 0.0 else 1.7)
                optimization_parameters.append(params)
                
            if nh_value != 0:
                nh_list.append(nh_value)
            else:
                nh_list.append(3e20)

        self.visualization_interp(optimization_parameters=optimization_parameters, photon_index=photon_index_list, key="CS_Chandra")

        col_names = ["Photon Index", "Nh"]
        col_data = [photon_index_list, nh_list]
        
        for name, data in zip(col_names, col_data):
            nearby_sources_table[name] = data
            
        return nearby_sources_table
        
        
    def photon_index_nh_for_other_catalog(self, key: str, table: Table) -> Table:
        """
        Computes and updates photon index and hydrogen column density (NH) for sources in a given astronomical catalog other than XMM and CSC.

        Parameters:
            key (str): The key identifying the catalog (e.g., 'Swift', 'eRosita').
            table (Table): The table containing the catalog data.

        Returns:
            Table: The updated table with 'Photon Index' and 'Nh' columns added.

        Methodology:
        - Iterates over each source in the table.
        - For each source, calculates the photon index using the `optimization` method.
        - Assigns a default NH value of 3e20.
        - The calculated photon index and NH values are added to the photon_index_list and nh_list, respectively.
        - Calls `visualization_interp` for visualizing the photon index interpolation.
        - Updates the table with the new 'Photon Index' and 'Nh' columns.
        - Returns the updated table.

        This method is essential for spectral analysis in astrophysics, allowing for the enhancement of catalog data with key spectral parameters.
        """
        
        photon_index_list, nh_list, optimization_parameters = [], [], []
        
        number_source = len(table)
        
        for index in range(number_source):
            params, photon_index = self.optimization(index=index, key=key, table=table)
            photon_index_list.append(photon_index if photon_index > 0.0 else 1.7)
            optimization_parameters.append(params)
            nh_list.append(3e20)
        
        self.visualization_interp(optimization_parameters=optimization_parameters, photon_index=photon_index_list, key=key)

        col_names = ["Photon Index", "Nh"]
        col_data = [photon_index_list, nh_list]
        
        for name, data in zip(col_names, col_data):
            table[name] = data
            
        return table
        
    
    def neighbourhood_of_object(self, key: Tuple[str, str], simulation_data: Dict, radius) -> None:
        """
        Visualizes the astronomical sources in the vicinity of a given object, using data from two different catalogs.

        Parameters:
            key (Tuple[str, str]): A tuple containing the keys of the two catalogs being compared.
            simulation_data (Dict): A dictionary containing simulation data, including object information.
            radius (float): The radius around the object within which sources are considered.

        Returns:
            None: This method does not return anything but produces a visualization plot.

        Description:
        - Retrieves object data, including name and celestial position.
        - Based on the catalog keys, determines the right ascension (RA) and declination (DEC) column names for each catalog.
        - Creates a plot with two subplots, each representing one of the catalogs.
        - In each subplot:
            - Plots the positions of the sources from the respective catalog.
            - Highlights the position of the main object.
        - Saves the plot image to a specified path.

        This method is useful for astronomers to visually assess the distribution of sources around a particular object across different catalogs.
        """
        
        object_data = simulation_data["object_data"]
        object_name = object_data["object_name"]
        obj_ra, obj_dec = object_data["object_position"].ra, object_data["object_position"].dec
        
        if key[0] == "Xmm_DR13" and key[1] == "CSC_2.0":
            ra_1, ra_2 = "SC_RA", "ra"
            dec_1, dec_2 = "SC_DEC", "dec"
        elif key[0] == "CSC_2.0" and key[1] == "Xmm_DR13":
            ra_1, ra_2 = "ra", "SC_RA"
            dec_1, dec_2 = "dec", "SC_DEC"
        elif key[0] == "Xmm_DR13" and (key[1] == "Swift" or key[1] == "eRosita"):
            ra_1, ra_2 = "SC_RA", "RA"
            dec_1, dec_2 = "SC_DEC", "DEC"
        elif key[0] == "CSC_2.0" and (key[1] == "Swift" or key[1] == "eRosita"):
            ra_1, ra_2 = "ra", "RA"
            dec_1, dec_2 = "dec", "DEC"
        elif (key[0] == "Swift" or key[0] == "eRosita") and key[1] == "Xmm_DR13":
            ra_1, ra_2 = "RA", "SC_RA"
            dec_1, dec_2 = "DEC", "SC_DEC"
        elif (key[0] == "Swift" or key[0] == "eRosita") and key[1] == "CSC_2.0":
            ra_1, ra_2 = "RA", "ra"
            dec_1, dec_2 = "DEC", "dec"
        else:
            print("Invalid key !")
            sys.exit()
            
        figure_1, axes = plt.subplots(1, 2, figsize=(17, 9), sharey=True)
        figure_1.suptitle(f"Neighbourhood of {object_name}, radius = {radius}", fontsize=20)
        figure_1.text(0.5, 0.04, 'Right Ascension [deg]', ha='center', va='center', fontsize=16)
        figure_1.text(0.04, 0.5, 'Declination [deg]', ha='center', va='center', rotation='vertical', fontsize=16)
        
        ax0, ax1 = axes[0], axes[1]
        
        cat_1_ra, cat_1_dec = self.nearby_sources_table_1[ra_1], self.nearby_sources_table_1[dec_1]
        ax0.scatter(cat_1_ra, cat_1_dec, s=30, edgecolors='black', facecolors='none', label=f"{len(cat_1_ra)} sources")
        ax0.scatter(obj_ra, obj_dec, s=120, marker="*", facecolors='none', edgecolors='red', label=f"{object_name}")
        ax0.legend(loc="upper right")
        ax0.set_title(f"With {key[0]} catalog")
        
        cat_2_ra, cat_2_dec = self.nearby_sources_table_2[ra_2], self.nearby_sources_table_2[dec_2]
        ax1.scatter(cat_2_ra, cat_2_dec, s=30, edgecolors='black', facecolors='none', label=f"{len(cat_2_ra)} sources")
        ax1.scatter(obj_ra, obj_dec, s=120, marker="*", facecolors='none', edgecolors='red', label=f"{object_name}")
        ax1.legend(loc="upper right")
        ax1.set_title(f"With {key[1]} catalog")
        
        os_dictionary = simulation_data["os_dictionary"]
        img = os_dictionary["img"]
        img_path = os.path.join(img, f"{key[0]}_{key[1]}_neighbourhood_of_{object_name}.png".replace(" ", "_")).replace("\\", "/")
        plt.savefig(img_path)
        plt.show()


    def dictionary_model(self, key: Tuple[str, str]) -> Tuple[Dict, Dict]:
        """
        Creates dictionaries for each source in the nearby sources tables, specifying the astrophysical model parameters.

        Parameters:
            key (Tuple[str, str]): Tuple containing the keys identifying the two catalogs being compared.

        Returns:
            Tuple[Dict, Dict]: Two dictionaries, each containing the model parameters for the sources in the corresponding nearby sources table.

        Description:
        - Converts catalog keys to standard format if necessary.
        - Iterates over each source in both nearby sources tables.
        - For each source, creates a dictionary with model type ('power'), photon index, observed flux, and column density.
        - The model parameters for each source are stored in separate dictionaries for each catalog.
        - Returns a tuple containing these two dictionaries.

        This method is crucial for preparing data for further spectral analysis, encapsulating key parameters in an accessible format.
        """
        
        model_dictionary_1, model_dictionary_2 = {}, {}
        
        key_0 = key[0]
        key_1 = key[1]
        
        if key_0 == "Xmm_DR13":
            key_0 = "XMM"
        elif key_1 == "Xmm_DR13":
            key_1 = "XMM"
            
        if key_0 == "CSC_2.0":
            key_0 = "CS_Chandra"
        elif key_1 == "CSC_2.0":
            key_1 = "CS_Chandra"
        
        flux_obs_1 = dict_cat.dictionary_catalog[key_0]["flux_obs"]
        flux_obs_2 = dict_cat.dictionary_catalog[key_1]["flux_obs"]
        
        for index in range(len(self.nearby_sources_table_1)):

            dictionary = {
                "model": 'power',
                "model_value": self.nearby_sources_table_1["Photon Index"][index],
                "flux": self.nearby_sources_table_1[flux_obs_1][index],
                "column_dentsity": self.nearby_sources_table_1["Nh"][index]
            }

            model_dictionary_1[f"src_{index}"] = dictionary
            
        for index in range(len(self.nearby_sources_table_2)):

            dictionary = {
                "model": 'power',
                "model_value": self.nearby_sources_table_2["Photon Index"][index],
                "flux": self.nearby_sources_table_2[flux_obs_2][index],
                "column_dentsity": self.nearby_sources_table_2["Nh"][index]
            }

            model_dictionary_2[f"src_{index}"] = dictionary
            
        return model_dictionary_1, model_dictionary_2
        

    def count_rate(self) -> Tuple[List, List]:
        """
        Calculates the count rate for each source in both nearby sources tables using the PIMMS tool.

        Returns:
            Tuple[List, List]: Two lists containing the calculated count rates for each source in the corresponding nearby sources table.

        Description:
        - Iterates over each source in both nearby sources tables.
        - For each source, extracts the astrophysical model parameters from the model dictionaries.
        - Generates and runs PIMMS commands to calculate the count rate for each source based on its model parameters.
        - The resulting count rates are stored in two separate lists.
        - Updates the nearby sources tables with the calculated count rates.
        - Returns a tuple containing these two lists of count rates.

        This method enables the quantification of expected detector count rates based on the observed astrophysical parameters, which is essential for planning and analyzing astronomical observations.
        """
        
        number_source_1 = len(self.nearby_sources_table_1)
        number_source_2 = len(self.nearby_sources_table_2)
        
        ct_rates_1, ct_rates_2 = np.array([], dtype=float), np.array([], dtype=float)

        for item in range(number_source_1):
            model_1 = self.model_dictionary_1[f"src_{item}"]["model"]
            model_value_1 = self.model_dictionary_1[f"src_{item}"]["model_value"]
            flux_1 =  self.model_dictionary_1[f"src_{item}"]["flux"]
            nh_1 = self.model_dictionary_1[f"src_{item}"]["column_dentsity"]
                    
            pimms_cmds = f"instrument nicer 0.3-10.0\nfrom flux ERGS 0.2-12.0\nmodel galactic nh {nh_1}\nmodel {model_1} {model_value_1} 0.0\ngo {flux_1}\nexit\n"
            
            with open('pimms_script.xco', 'w') as file:
                file.write(pimms_cmds)
                file.close()
                
            result = subprocess.run(['pimms', '@pimms_script.xco'], stdout=subprocess.PIPE).stdout.decode(sys.stdout.encoding)
            ct_rate_value = float(result.split("predicts")[1].split('cps')[0])
            ct_rates_1 = np.append(ct_rates_1, ct_rate_value)
            
            self.nearby_sources_table_1["count_rate"] = ct_rates_1
            
        for item in range(number_source_2):
            model_2 = self.model_dictionary_2[f"src_{item}"]["model"]
            model_value_2 = self.model_dictionary_2[f"src_{item}"]["model_value"]
            flux_2 =  self.model_dictionary_2[f"src_{item}"]["flux"]
            nh_2 = self.model_dictionary_2[f"src_{item}"]["column_dentsity"]
                    
            pimms_cmds = f"instrument nicer 0.3-10.0\nfrom flux ERGS 0.2-12.0\nmodel galactic nh {nh_2}\nmodel {model_2} {model_value_2} 0.0\ngo {flux_2}\nexit\n"
            
            with open('pimms_script.xco', 'w') as file:
                file.write(pimms_cmds)
                file.close()
                
            result = subprocess.run(['pimms', '@pimms_script.xco'], stdout=subprocess.PIPE).stdout.decode(sys.stdout.encoding)
            ct_rate_value = float(result.split("predicts")[1].split('cps')[0])
            ct_rates_2 = np.append(ct_rates_2, ct_rate_value)
            
            self.nearby_sources_table_2["count_rate"] = ct_rates_2
        
        return ct_rates_1, ct_rates_2
    
    
    def xslx_to_py(self, args, table, simulation_data, radius) -> Tuple[List, Table]:
        """
        Reads count rate data from an Excel file based on specified parameters and updates the provided table with these values.

        Parameters:
        args (str): Identifier for the catalog to be used (e.g., "Xmm_DR13", "CSC_2.0", "Swift", "eRosita", "match").
        table (Table): The table to be updated with count rate data.
        simulation_data (Dict): Dictionary containing simulation parameters and paths.
        radius (float): The radius value used to define the file name for the Excel data.

        Returns:
        Tuple[List, Table]: A tuple where the first element is a list of count rates and the second element is the updated table with these rates.

        This method constructs the path to the relevant Excel file based on the catalog and object name, reads the count rates, and updates the table's 'count_rate' column.
        """
        
        data_path = simulation_data["os_dictionary"]["data_path"]
        object_data = simulation_data["object_data"]
        excel_data_path = os.path.join(data_path, "excel_data").replace("\\", "/")

        if args == "Xmm_DR13":
            cat = "xmm"
        elif args == "CSC_2.0":
            cat = "csc_CS_Chandra"
        elif args == "Swift":
            cat = "swi"
        elif args == "eRosita":
            cat = "ero"
        elif args == "match":
            cat = "xmmXchandra"
        
        ct_rates_path = os.path.join(excel_data_path, f"{cat}_{radius}_{object_data['object_name']}.xlsx".replace(" ", "_"))
        wb = openpyxl.load_workbook(ct_rates_path)
        sheet = wb.active

        count_rates = []
        for item in range(len(table)): 
            count_rates.append(sheet.cell(row = item + 1, column = 1).value)
            
        table["count_rate"] = count_rates

        return count_rates, table
    
    
    def calculate_opti_point(self, simulation_data, key) -> Tuple[Dict, Dict, int, int]:
        """
        Calculates the optimal pointing coordinates for an astronomical object to maximize the signal-to-noise ratio (S/N) based on nearby sources.

        Parameters:
        simulation_data (Dict): Dictionary containing simulation parameters including telescope and object data.
        key (Tuple[str, str]): A tuple of two catalog identifiers used for the analysis.

        Returns:
        Tuple[Dict, Dict, int, int]: A tuple containing two dictionaries for each catalog, and two integers representing the indices of optimal pointing coordinates.

        This method creates a grid of potential pointing coordinates around the object, calculates the S/N ratio for each point, and identifies the optimal pointing location. It also generates and saves a visualization of the S/N map for each catalog.
        """
        
        min_value, max_value, step = -7.0, 7.1, 0.05
        DeltaRA = Angle(np.arange(min_value, max_value, step), unit=u.deg)/60
        DeltaDEC = Angle(np.arange(min_value, max_value, step), unit=u.deg)/60
        
        telescop_data = simulation_data["telescop_data"]
        object_data = simulation_data["object_data"]
        object_name = object_data["object_name"]
        
        RA_grid, DEC_grid = np.meshgrid(DeltaRA, DeltaDEC)

        SampleRA = object_data["object_position"].ra.deg + RA_grid.flatten().deg
        SampleDEC = object_data["object_position"].dec.deg + DEC_grid.flatten().deg
        
        NICERpointing = SkyCoord(ra=SampleRA*u.deg, dec=SampleDEC*u.deg)
        NICERpointing = NICERpointing.reshape(-1, 1)
        
        PSRseparation = f.ang_separation(object_data["object_position"], NICERpointing).arcmin
        PSRcountrateScaled = f.scaled_ct_rate(PSRseparation, object_data['count_rate'], telescop_data["EffArea"], telescop_data["OffAxisAngle"])
        
        sources_1 = self.nearby_sources_position_1.reshape(1, -1)
        SRCseparation_1 = f.ang_separation(sources_1, NICERpointing).arcmin
        count_rate_1 = self.nearby_sources_table_1['count_rate']
        SRCcountrateScaled_1 = f.scaled_ct_rate(SRCseparation_1, count_rate_1, telescop_data["EffArea"], telescop_data["OffAxisAngle"])
        SNR_1, PSRrates, SRCrates_1  = np.zeros((3, len(DeltaRA) * len(DeltaDEC)))
        for item in range(len(PSRcountrateScaled)):
            PSRrates[item] = PSRcountrateScaled[item]
            SRCrates_1[item] = np.sum(SRCcountrateScaled_1[item])
            SNR_1[item] = f.signal_to_noise(PSRrates[item], SRCrates_1[item], simulation_data["INSTbkgd"], simulation_data["EXPtime"])
        
        sources_2 = self.nearby_sources_position_2.reshape(1, -1)
        SRCseparation_2 = f.ang_separation(sources_2, NICERpointing).arcmin
        count_rate_2 = self.nearby_sources_table_2['count_rate']
        SRCcountrateScaled_2 = f.scaled_ct_rate(SRCseparation_2, count_rate_2, telescop_data["EffArea"], telescop_data["OffAxisAngle"])
        SNR_2, PSRrates, SRCrates_2  = np.zeros((3, len(DeltaRA) * len(DeltaDEC)))
        for item in range(len(PSRcountrateScaled)):
            PSRrates[item] = PSRcountrateScaled[item]
            SRCrates_2[item] = np.sum(SRCcountrateScaled_2[item])
            SNR_2[item] = f.signal_to_noise(PSRrates[item], SRCrates_2[item], simulation_data["INSTbkgd"], simulation_data["EXPtime"])
        
        OptimalPointingIdx_1 = np.where(SNR_1==max(SNR_1))[0][0]
        SRCoptimalSEPAR_1 = f.ang_separation(self.nearby_sources_position_1, SkyCoord(ra=SampleRA[OptimalPointingIdx_1]*u.degree, dec=SampleDEC[OptimalPointingIdx_1]*u.degree)).arcmin
        SRCoptimalRATES_1 = f.scaled_ct_rate(SRCoptimalSEPAR_1, self.count_rates_1, telescop_data["EffArea"], telescop_data["OffAxisAngle"])
        
        OptimalPointingIdx_2 = np.where(SNR_2==max(SNR_2))[0][0]
        SRCoptimalSEPAR_2 = f.ang_separation(self.nearby_sources_position_2, SkyCoord(ra=SampleRA[OptimalPointingIdx_1]*u.degree, dec=SampleDEC[OptimalPointingIdx_2]*u.degree)).arcmin
        SRCoptimalRATES_2 = f.scaled_ct_rate(SRCoptimalSEPAR_2, self.count_rates_2, telescop_data["EffArea"], telescop_data["OffAxisAngle"])     
        
        vector_dictionary_1 = {'SampleRA': SampleRA,
                               'SampleDEC': SampleDEC,
                               'PSRrates': PSRrates,
                               'SRCrates_2': SRCrates_2,
                               'SRCoptimalRATES_1':SRCoptimalRATES_1,
                               'SNR_1': SNR_1}
        
        vector_dictionary_2 = {'SampleRA': SampleRA,
                               'SampleDEC': SampleDEC,
                               'PSRrates': PSRrates,
                               'SRCrates_2': SRCrates_2,
                               'SRCoptimalRATES_2':SRCoptimalRATES_2,
                               'SNR_2': SNR_2}

        opti_ra_1, opti_dec_1 = vector_dictionary_1['SampleRA'][OptimalPointingIdx_1], vector_dictionary_1['SampleDEC'][OptimalPointingIdx_1]
        opti_ra_2, opti_dec_2 = vector_dictionary_2['SampleRA'][OptimalPointingIdx_2], vector_dictionary_2['SampleDEC'][OptimalPointingIdx_2]
        
        figure_map, axes = plt.subplots(1, 2, figsize=(17, 9), sharey=True)
        figure_map.suptitle(f"S/N map for {object_data['object_name']}", fontsize=20)
        figure_map.text(0.5, 0.04, 'Right Ascension [deg]', ha='center', va='center', fontsize=16)
        figure_map.text(0.04, 0.5, 'Declination [deg]', ha='center', va='center', rotation='vertical', fontsize=16)

        ax0 = axes[0]
        ax0.plot(self.nearby_sources_position_1.ra, self.nearby_sources_position_1.dec, marker='.', color='black', linestyle='', label=f"{len(self.nearby_sources_position_1.ra)} sources")
        ax0.plot(object_data["object_position"].ra, object_data["object_position"].dec, marker='*', color="green", linestyle='', label=f"{object_data['object_name']}")
        ax0.plot(opti_ra_1, opti_dec_1, marker="+", color='red', linestyle='', label="Optimal pointing point")
        ax0.scatter(vector_dictionary_1['SampleRA'], vector_dictionary_1['SampleDEC'], c=vector_dictionary_1["SNR_1"], s=10, edgecolor='face')
        ax0.set_title(f"With {key[0]} catalog\n Optimal pointing point : {opti_ra_1} deg, {opti_dec_1} deg")
        ax0.legend(loc="upper right", ncol=2)

        ax1 = axes[1]
        ax1.plot(self.nearby_sources_position_2.ra, self.nearby_sources_position_2.dec, marker='.', color='black', linestyle='', label=f"{len(self.nearby_sources_position_2.ra)} sources")
        ax1.plot(object_data["object_position"].ra, object_data["object_position"].dec, marker='*', color="green", linestyle='', label=f"{object_data['object_name']}")
        ax1.plot(opti_ra_2, opti_dec_2, marker="+", color='red', linestyle='', label="Optimal pointing point")
        ax1.scatter(vector_dictionary_2['SampleRA'], vector_dictionary_2['SampleDEC'], c=vector_dictionary_2["SNR_2"], s=10, edgecolor='face')
        ax1.set_title(f"With {key[1]} catalog\n Optimal pointing point : {opti_ra_2} deg, {opti_dec_2} deg")
        ax1.legend(loc="upper right", ncol=2)
 
        norm = plt.Normalize(vmin=min(vector_dictionary_1["SNR_1"] + vector_dictionary_2["SNR_2"]), vmax=max(vector_dictionary_1["SNR_1"] + vector_dictionary_2["SNR_2"]))
        sm = ScalarMappable(cmap='viridis', norm=norm)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        color_bar = figure_map.colorbar(sm, cax=cax)
        color_bar.set_label("S/N")
        
        os_dictionary = simulation_data["os_dictionary"]
        img = os_dictionary["img"]
        img_path = os.path.join(img, f"{key[0]}_{key[1]}_SNR_{object_name}.png".replace(" ", "_")).replace("\\", "/")
        plt.savefig(img_path)
        plt.show()

        return vector_dictionary_1, vector_dictionary_2, OptimalPointingIdx_1, OptimalPointingIdx_2
    
    
    def variability_table(self, simulation_data: Dict, radius: int) -> str:
        """
        Creates a table of master sources around a specific astronomical object based on given radius and simulation data.

        Parameters:
        simulation_data (Dict): Dictionary containing various parameters and paths used in the simulation.
        radius (int): Radius in arcminutes defining the area around the object of interest.

        Returns:
        str: Path to the created master source FITS file.

        This method first extracts sources around the specified region (RA, Dec) within the given radius from the master source catalog. Then, it selects relevant catalog sources around this region. It uses STILTS (Software for the Treatment of Image Data from Large Telescopes) for data processing. Finally, it visualizes and saves the resulting master sources.

        Exceptions:
        Any exceptions during processing are caught and printed to the console.

        Note:
        This method relies on external software (STILTS) and assumes the existence of a master source catalog and additional catalogs defined in `simulation_data`.
        """
        
        os_dictionary = simulation_data["os_dictionary"]
        catalog_datapath = os_dictionary["catalog_datapath"]
        stilts_software_path = os_dictionary["stilts_software_path"]
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
            for cat in dict_cat.catalogs:
                path_to_cat_init = os.path.join(catalog_datapath, cat).replace("\\", "/")
                path_to_cat_final = os.path.join(output_name, cat).replace("\\", "/")
                command = (f"java -jar {stilts_software_path} tmatch2 matcher=exact \
                        in1='{master_cone_path}' in2='{path_to_cat_init}.fits' out='{path_to_cat_final}.fits'\
                            values1='{cat}' values2='{cat}_IAUNAME' find=all progress=none")
                command = shlex.split(command)
                subprocess.run(command)

        object_data = simulation_data["object_data"]
        right_ascension = object_data["object_position"].ra.value
        declination = object_data["object_position"].dec.value
        try:
            print(f"\n{colored('Load Erwan s code for :', 'yellow')} {object_data['object_name']}")
            select_master_sources_around_region(ra=right_ascension, dec=declination, radius=radius, output_name=output_name)
            select_catalogsources_around_region(output_name=output_name)
            master_sources = f.load_master_sources(output_name)
            f.master_source_plot(master_sources=master_sources, simulation_data=simulation_data, number_graph=len(master_sources))
        except Exception as error :
            print(f"{colored('An error occured : ', 'red')} {error}")
            
        return master_source_path
    
    
    def variability_index(self, key: str, iauname: str, nearby_sources_table: Table) -> List:
        """
        Identifies and returns the indices of variable sources from a nearby sources table based on a master source catalog.

        Parameters:
        key (str): Key representing the catalog name (e.g., 'CSC_2.0', 'Xmm_DR13').
        iauname (str): The column name in `nearby_sources_table` representing source names.
        nearby_sources_table (Table): A table containing data of nearby sources.

        Returns:
        List: A list of indices in `nearby_sources_table` corresponding to variable sources found in the master source catalog.

        This method opens the master source FITS file and filters out variable sources based on the catalog specified by `key`. It then matches these sources with those in the `nearby_sources_table` and compiles a list of indices representing these variable sources within the table.

        Note:
        This method assumes that the master source path is already set in the instance variable `self.master_source_path`.
        """
        
        with fits.open(self.master_source_path) as data:
            master_source_cone = Table(data[1].data)
        
        if key == "CSC_2.0":
            key = "Chandra"
        elif key == "Xmm_DR13":
            key = "XMM"
        
        msc_name = [name for name in master_source_cone[key] if name != ""]
        var_index_in_nearby_sources_table = []
        for name in msc_name:
            if name in nearby_sources_table[iauname]:
                index_in_table = list(nearby_sources_table[iauname]).index(name)
                var_index_in_nearby_sources_table.append(index_in_table)
                
        return var_index_in_nearby_sources_table
    
    
    def write_fits_table(self, table: Table, key: str, os_dictionary: Dict) -> None:
        """
        Writes a given table to a FITS file, using a specific catalog key and directory paths from a dictionary.

        Parameters:
        table (Table): The table to be written to the FITS file.
        key (str): Key representing the catalog name used for naming the FITS file.
        os_dictionary (Dict): Dictionary containing various file paths used in the operation.

        This method attempts to write the `table` to a FITS file in the directory specified in `os_dictionary["cloesest_dataset_path"]`. The file is named using the `key` parameter. If an error occurs during this process, it is caught and printed to the console.

        Note:
        The method overwrites any existing file with the same name.
        """
        
        try:            
            cloesest_dataset_path = os_dictionary["cloesest_dataset_path"]
            nearby_sources_table_path = os.path.join(cloesest_dataset_path, f"{key}_nearby_sources_table.fits").replace("\\", "/")
            table.write(nearby_sources_table_path, format='fits', overwrite=True)
            print(f"Nearby sources table was created in : {colored(nearby_sources_table_path, 'magenta')}")
        except Exception as error:
            print(f"{colored('An error occured : ', 'red')} {error}")
    
    
    def modeling_source_spectra(self, simulation_data: Dict, exp_time: int, key: Tuple[str, str]) -> Tuple[List, List]:
        """
        Generates modeled spectra for sources in nearby sources tables for given catalogs.

        Parameters:
        simulation_data (Dict): A dictionary containing simulation data including telescope data.
        exp_time (int): Exposure time used in the simulation.
        key (Tuple[str, str]): A tuple containing the keys of the catalogs to be used.

        Returns:
        Tuple[List, List, List, List, Instrument]: A tuple containing lists of total spectra, total variable spectra for both catalogs, and the instrument used.

        This method creates and models spectral data for each source in the nearby sources tables of the specified catalogs (key[0] and key[1]). It utilizes an X-ray spectrum model (Tbabs * Powerlaw) and simulates the spectra using the NICER instrument's ARF and RMF files. The method also accounts for the vignetting factor for each source. The result is a collection of spectra for all sources, as well as a separate collection for variable sources as identified by their indices.

        Note:
        This method assumes that the necessary data paths and instrumental information are provided in `simulation_data`.
        """
        
        model = Tbabs() * Powerlaw()
        telescop_data = simulation_data["telescop_data"]
        nicer_data_arf = telescop_data["nicer_data_arf"]
        nicer_data_rmf = telescop_data["nicer_data_rmf"]
        
        instrument = Instrument.from_ogip_file(nicer_data_arf, nicer_data_rmf, exposure=exp_time)
        
        print(f"\n{colored(f'Modeling spectra for {key[0]} catalog... ', 'yellow')}")
        size = 10_000
        
        total_spectra_1, total_var_spectra_1 = [], []
        for index, vignet_factor in tqdm(enumerate(self.nearby_sources_table_1["vignetting_factor"])):
            parameters = {}
            parameters = {
                "tbabs_1": {"N_H": np.full(size, self.nearby_sources_table_1["Nh"][index]/1e22)},
                "powerlaw_1": {
                    "alpha": np.full(size, self.nearby_sources_table_1["Photon Index"][index] if self.nearby_sources_table_1["Photon Index"][index] > 0.0 else 1.7),
                    "norm": np.full(size, 1e-5),
                }
            }
            
            spectra = fakeit_for_multiple_parameters(instrument=instrument, model=model, parameters=parameters) * vignet_factor
            
            if index in self.var_index_1:
                total_var_spectra_1.append(spectra)
            
            total_spectra_1.append(spectra)
        
        print(f"\n{colored(f'Modeling spectra for {key[1]} catalog... ', 'yellow')}")
        total_spectra_2, total_var_spectra_2 = [], []
        for index, vignet_factor in tqdm(enumerate(self.nearby_sources_table_2["vignetting_factor"])):
            parameters = {}
            parameters = {
                "tbabs_1": {"N_H": np.full(size, self.nearby_sources_table_2["Nh"][index]/1e22)},
                "powerlaw_1": {
                    "alpha": np.full(size, self.nearby_sources_table_2["Photon Index"][index] if self.nearby_sources_table_2["Photon Index"][index] > 0.0 else 1.7),
                    "norm": np.full(size, 1e-5),
                }
            }
            
            spectra = fakeit_for_multiple_parameters(instrument=instrument, model=model, parameters=parameters) * vignet_factor
            
            if index in self.var_index_1:
                total_var_spectra_2.append(spectra)
            
            total_spectra_2.append(spectra)
            
            
        return total_spectra_1, total_spectra_2, total_var_spectra_1, total_var_spectra_2, instrument
     
    
    def total_spectra_plot(self, simulation_data: Dict, radius: float, key: Tuple[str, str]):
        """
        Plots the modeled spectra for sources around a specific object from two different catalogs.

        Parameters:
        simulation_data (Dict): A dictionary containing simulation data including object data.
        radius (float): The radius within which the sources are considered.
        key (Tuple[str, str]): A tuple containing the keys of the catalogs to be used.

        This method plots the modeled spectra for sources in the vicinity of a specified object, using data from two different catalogs. The plots include individual spectra for each catalog, as well as a combined plot showcasing the summed spectra and variability errors. The method also calculates the upper and lower limits for the spectra to provide an envelope for the variability. Each subplot is appropriately labeled and the overall figure title indicates the object around which the spectra are modeled.

        Note:
        This method uses the total and variable spectra lists generated by the `modeling_source_spectra` method and the instrumental data from `simulation_data`.
        """
        
        object_name = simulation_data["object_data"]["object_name"]
        
        figure_spectra, axes = plt.subplots(2, 3, figsize=(17, 8), sharey=True, sharex=True)
        figure_spectra.suptitle(f"Spectral modeling close to {object_name}", fontsize=20)
        figure_spectra.text(0.5, 0.04, 'Energy [keV]', ha='center', va='center', fontsize=16)
        figure_spectra.text(0.04, 0.5, 'Counts [cts/s]', ha='center', va='center', rotation='vertical', fontsize=16)
        
        figure_spectra.text(0.07, 0.75, f'{key[0]}', ha='center', va='center', rotation='vertical', fontsize=14)
        figure_spectra.text(0.07, 0.25, f'{key[1]}', ha='center', va='center', rotation='vertical', fontsize=14)
        
        graph_data = {"min_lim_x": 0.2,
                      "max_lim_x": 10.0}
        
        for ax_row in axes:
            for ax in ax_row:
                ax.loglog()
                ax.set_xlim([graph_data["min_lim_x"], graph_data["max_lim_x"]])
        
        ax00, ax01, ax02 = axes[0][0], axes[0][1], axes[0][2]
        ax10, ax11, ax12 = axes[1][0], axes[1][1], axes[1][2]
        figure_spectra.delaxes(ax02)
        figure_spectra.delaxes(ax12)
        
        # ---------- first row ---------- #
        
        for spectra in self.total_spectra_1:
            ax00.step(self.instrument.out_energies[0],
                      np.median(spectra, axis=0),
                      where="post")
        ax00.set_title("All spectra of nearby sources table")
        
        spectrum_summed_1 = 0.0
        for index in range(len(self.total_spectra_1)):
            spectrum_summed_1 += self.total_spectra_1[index]

        spectrum_var_summed_1 = 0.0
        for index in range(len(self.total_var_spectra_1)):
            spectrum_var_summed_1 += self.total_var_spectra_1[index]
        
        ax01.errorbar(self.instrument.out_energies[0], y=np.median(spectrum_summed_1, axis=0), yerr=np.median(spectrum_var_summed_1, axis=0), 
                    fmt="none", ecolor='red', capsize=2, capthick=3,
                    label='error')
        ax01.step(self.instrument.out_energies[0], np.median(spectrum_summed_1, axis=0), color='black', label="sum powerlaw")
        ax01.set_title("Spectrum Summed with var sources error")
        ax01.legend(loc='upper right')
        
        # ---------- second row ---------- #
        
        for spectra in self.total_spectra_2:
            ax10.step(self.instrument.out_energies[0],
                      np.median(spectra, axis=0),
                      where="post")
        
        spectrum_summed_2 = 0.0
        for index in range(len(self.total_spectra_2)):
            spectrum_summed_2 += self.total_spectra_2[index]

        spectrum_var_summed_2 = 0.0
        for index in range(len(self.total_var_spectra_2)):
            spectrum_var_summed_2 += self.total_var_spectra_2[index]
            
        ax11.errorbar(self.instrument.out_energies[0], y=np.median(spectrum_summed_2, axis=0), yerr=np.median(spectrum_var_summed_2, axis=0), 
                    fmt="none", ecolor='red', capsize=2, capthick=3,
                    label='error')
        ax11.step(self.instrument.out_energies[0], np.median(spectrum_summed_2, axis=0), color='black', label="sum powerlaw")
        ax11.legend(loc='upper right')
        
        # ---------- big axes ---------- #
        
        new_ax = figure_spectra.add_subplot(1, 3, 3)
        new_ax.errorbar(self.instrument.out_energies[0], y=np.median(spectrum_summed_1, axis=0), yerr=np.median(spectrum_var_summed_1, axis=0),
                        fmt="none", ecolor='red', capsize=2, capthick=3, alpha=0.1, label='error_1')
        new_ax.step(self.instrument.out_energies[0], np.median(spectrum_summed_1, axis=0), color='black', label=f"sum powerlaw {key[0]}")
        new_ax.set_title("Spectrum Summed with var sources error")
        new_ax.legend(loc='upper right')
        
        new_ax.errorbar(self.instrument.out_energies[0], y=np.median(spectrum_summed_2, axis=0), yerr=np.median(spectrum_var_summed_2, axis=0),
                        fmt="none", ecolor='darkorange', capsize=2, capthick=3, alpha=0.1, label='error_2')
        new_ax.step(self.instrument.out_energies[0], np.median(spectrum_summed_2, axis=0), color='navy', label=f"sum powerlaw {key[1]}")
        new_ax.set_title("Spectrum Summed with var sources error")
        new_ax.legend(loc='upper right')
        
        new_ax.legend(loc="upper right", ncol=2)
        new_ax.set_xlim([graph_data["min_lim_x"], graph_data["max_lim_x"]])
        new_ax.loglog()
        
        # ---------- get envelop ---------- #
        
        y_upper_1 = np.median(spectrum_summed_1, axis=0) + np.median(spectrum_var_summed_1, axis=0)
        y_lower_1 = np.median(spectrum_summed_1, axis=0) - np.median(spectrum_var_summed_1, axis=0)
        
        y_upper_2 = np.median(spectrum_summed_2, axis=0) + np.median(spectrum_var_summed_2, axis=0)
        y_lower_2 = np.median(spectrum_summed_2, axis=0) - np.median(spectrum_var_summed_2, axis=0)
        
        data_1 = {
            "Energy": self.instrument.out_energies[0],
            "Counts": np.median(spectrum_summed_1, axis=0),
            "Upper limit": y_upper_1,
            "Lower limit": y_lower_1
        }
        
        data_2 = {
            "Energy": self.instrument.out_energies[0],
            "Counts": np.median(spectrum_summed_2, axis=0),
            "Upper limit": y_upper_2,
            "Lower limit": y_lower_2
        }

        os_dictionary = simulation_data["os_dictionary"]
        img = os_dictionary["img"]
        img_path = os.path.join(img, f"{key[0]}_{key[1]}_spectral_modeling_close_to_{object_name}.png".replace(" ", "_")).replace("\\", "/")
        plt.savefig(img_path)
        plt.show()
        
        return data_1, data_2
    
    
    def write_txt_file(self, simulation_data: Dict, data_1: Dict, data_2: Dict, key: Tuple[str, str]) -> None:
        """
        Writes the spectral modeling data into text files for each of the specified catalogs.

        Parameters:
        simulation_data (Dict): A dictionary containing simulation data including directory paths.
        data_1 (Dict): A dictionary containing the spectral data for the first catalog (key[0]).
        data_2 (Dict): A dictionary containing the spectral data for the second catalog (key[1]).
        key (Tuple[str, str]): A tuple containing the keys of the catalogs.

        This method exports the spectral data for two different catalogs into separate text files. Each file contains data such as energy, counts, and upper and lower limits of the spectra. The data is formatted into columns for readability. The files are named according to the catalogs' keys and saved in the specified directory.

        The method iterates through the provided spectral data, formats each row according to the given specifications, and writes the rows to the respective text files. The headers of the files include the names of the data columns.

        Note:
        The method assumes that the directory for saving the text files is provided in `simulation_data['os_dictionary']["catalog_directory"]`.

        Example of Output File Format:
        Energy        Counts        Upper Limit   Lower Limit
        [value]       [value]       [value]       [value]
        ...           ...           ...           ...
        """
    
        catalog_directory = simulation_data['os_dictionary']["catalog_directory"]
        txt_path_1 = os.path.join(catalog_directory, f'{key[0]}_output_modeling_plot.txt').replace("\\", "/")
        
        data_to_txt_1 = [
            list(data_1.keys())
        ]
        
        energy, counts, y_upper, y_lower = list(data_1.values())
        data_to_txt_1.extend([energy[index], counts[index], y_upper[index], y_lower[index]] for index in range(len(energy)))
        
        with open(txt_path_1, 'w') as file:
            header = "{:<15} {:<15} {:<15} {:<15}".format(*data_to_txt_1[0])
            file.write(header + "\n")

            for row in data_to_txt_1[1:]:
                new_row = "{:<10.5f}     {:<10.5f}       {:<10.5f}       {:<10.5f}".format(*row)
                file.write(new_row + "\n")
                
        print(f"\n{colored(f'{key[0]}_output_modeling_plot.txt', 'yellow')} has been created in {colored(txt_path_1, 'blue')}")
        
        
        txt_path_2 = os.path.join(catalog_directory, f'{key[1]}_output_modeling_plot.txt').replace("\\", "/")
        
        data_to_txt_2 = [
            list(data_2.keys())
        ]
        
        energy, counts, y_upper, y_lower = list(data_2.values())
        data_to_txt_2.extend([energy[index], counts[index], y_upper[index], y_lower[index]] for index in range(len(energy)))
        
        with open(txt_path_2, 'w') as file:
            header = "{:<15} {:<15} {:<15} {:<15}".format(*data_to_txt_2[0])
            file.write(header + "\n")

            for row in data_to_txt_2[1:]:
                new_row = "{:<10.5f}     {:<10.5f}       {:<10.5f}       {:<10.5f}".format(*row)
                file.write(new_row + "\n")
                
        print(f"\n{colored(f'{key[1]}_output_modeling_plot.txt', 'yellow')} has been created in {colored(txt_path_2, 'blue')}")
    
    
# --------------- Software Class --------------- #

    # --------------- Refactoring --------------- #

class BandFlux:
    """
    A class representing the observed flux and its error in a specific energy band.

    Attributes:
    flux (float): The observed flux value in the energy band.
    flux_err (float): The error associated with the observed flux value.

    The `BandFlux` class is designed to encapsulate the properties of an observed flux in a specific energy band of an astronomical object. It stores the flux value along with its corresponding error. This class can be used in scenarios where handling flux data and its uncertainty is required, such as in spectral analysis or photometric measurements.

    Methods:
    __init__: Initializes a new instance of the `BandFlux` class with the specified flux and error values.
    """
    
    def __init__(self, flux, flux_err) -> None:
        """
        Initializes a new instance of the `BandFlux` class.

        Parameters:
        flux (float): The observed flux value in the energy band.
        flux_err (float): The error associated with the observed flux value.

        This constructor method creates an instance of `BandFlux` with the provided flux and its error. 
        These values are expected to be numerical representations of the observed flux and its uncertainty.
        """
        
        self.flux = flux
        self.flux_err = flux_err


class SwiftData:
    """
    A class to represent and store the stacked flux data, its associated errors, 
    and observation times for astronomical observations, specifically from the Swift telescope.

    This class is tailored for handling time-series data from the Swift telescope, 
    commonly used in astrophysical studies. It stores stacked flux values, their corresponding 
    errors, and the observation times, facilitating data analysis and manipulation in 
    time-resolved studies.

    Attributes:
    stacked_flux (list of float): A list of stacked flux values observed by the Swift telescope.
    stacked_flux_err (list of float): A list of errors associated with the stacked flux values.
    stacked_times (list of float): A list of observation times corresponding to each flux value.

    Methods:
    __init__: Initializes a new instance of the `SwiftData` class with the specified data.
    """
    
    def __init__(self, stacked_flux, stacked_flux_err, stacked_times):
        """
        Initializes a new instance of the `SwiftData` class.

        Parameters:
        stacked_flux (list of float): A list of stacked flux values observed by the Swift telescope.
        stacked_flux_err (list of float): A list of errors associated with the stacked flux values.
        stacked_times (list of float): A list of observation times corresponding to each flux value.

        This constructor method creates an instance of `SwiftData` with the provided flux data, 
        errors, and times. These values are typically derived from observations conducted by the 
        Swift telescope and are crucial in time-series analyses of astronomical sources.

        """
        self.stacked_flux = stacked_flux
        self.stacked_flux_err = stacked_flux_err
        self.stacked_times = stacked_times
        
        
    # ------------------------------------------- #

    # --------------- Source Class --------------- #


class Source:
    """
    A class to encapsulate various observational data of an astronomical source, 
    including flux measurements, band flux data, observation times, and additional 
    source-specific parameters.

    This class is designed to provide a structured way to store and manipulate 
    observational data from different catalogs, making it easier to perform various 
    analyses like calculating hardness ratios or handling Swift telescope data.

    Attributes:
    catalog (str): Name of the catalog from which the source data is obtained.
    iau_name (str): International Astronomical Union (IAU) designated name of the source.
    flux (list of float): List of flux values of the source.
    flux_err (list of float): Corresponding errors of the flux values.
    time_steps (list of float): Time steps of the observations.
    obs_ids (list of int): Observation IDs, if available.
    band_flux (list of float): Flux values in specific energy bands.
    band_flux_err (list of float): Errors associated with the band flux values.
    swift_data (SwiftData): Object containing Swift telescope specific data.
    xmm_offaxis (list): Off-axis angles for XMM-Newton observations, if available.
    short_term_var (list): Information about short-term variability, if available.

    Methods:
    __init__: Initializes the Source object with given data.
    hardness_ratio: Calculates the hardness ratio for the source based on catalog-specific parameters.
    swift_modif: Modifies Swift data attributes based on provided flux and flux error.

    """
    
    def __init__(self, catalog, iau_name, flux, flux_err, time_steps, band_flux_data, **kwargs) -> None:
        """
        Initializes a new instance of the Source class.

        Parameters:
        catalog (str): Name of the catalog from which the source data is obtained.
        iau_name (str): IAU designated name of the source.
        flux (list of float): List of flux values of the source.
        flux_err (list of float): Corresponding errors of the flux values.
        time_steps (list of float): Time steps of the observations.
        band_flux_data (BandFlux): Object containing flux and flux error data for specific energy bands.
        kwargs: Additional parameters such as observation IDs (obs_id), Swift data (swift_data), etc.

        This constructor initializes the Source object with the given data, converting some of the data 
        like observation times and observation IDs into appropriate formats. It also sets up default 
        values for some attributes based on the provided keyword arguments.

        """
        self.catalog = catalog
        self.iau_name = iau_name
        self.flux = flux
        self.flux_err = flux_err
        self.time_steps = [float(time) for time in time_steps]
        self.obs_ids = [int(obs_id) for obs_id in kwargs.get('obs_id', [])]

        self.band_flux = band_flux_data.flux
        self.band_flux_err = band_flux_data.flux_err

        self.swift_data = kwargs.get('swift_data', SwiftData([], [], []))
        self.xmm_offaxis = kwargs.get('xmm_offaxis', [])
        self.short_term_var = kwargs.get('short_term_var', [])

        self.hardness_ratio(catalog=catalog)
        self.swift_modif(flux=flux, flux_err=flux_err)

    
    def hardness_ratio(self, catalog:str) -> None:
        """
        Calculates the hardness ratio for a source based on the given catalog's parameters.

        The hardness ratio is a measure of the spectral shape of the source, typically
        used in X-ray astronomy. It is calculated using soft and hard detections which are 
        derived from band flux data, considering the specific energy bands defined for the 
        given catalog.

        Parameters:
        catalog (str): Name of the catalog which defines the energy bands and conversion factors.

        This method modifies the source object by calculating and setting the hardness ratio
        along with the associated errors.

        """

        hr_bandlimit_index = dict_cat.dictionary_catalog[catalog]["hr_bandlimit_index"]
        band_conv_factor_soft = dict_cat.dictionary_catalog[catalog]["band_conv_factor_soft"]
        band_conv_factor_hard = dict_cat.dictionary_catalog[catalog]["band_conv_factor_hard"]

        self.soft_dets = [np.sum(det[:hr_bandlimit_index]) * band_conv_factor_soft for det in self.band_flux]
        self.soft_errors = [[np.sum(err_neg[:hr_bandlimit_index]) * band_conv_factor_soft for err_neg in self.band_flux_err[0]],
                            [np.sum(err_pos[:hr_bandlimit_index]) * band_conv_factor_soft for err_pos in self.band_flux_err[1]]]

        if catalog != "RASS" and catalog != "WAGAT":
            self.hard_dets = [np.sum(det[hr_bandlimit_index:]) * band_conv_factor_hard for det in self.band_flux]
            self.hard_errors = [
                [np.sum(err_neg[hr_bandlimit_index:]) * band_conv_factor_hard for err_neg in
                self.band_flux_err[0]],
                [np.sum(err_pos[hr_bandlimit_index:]) * band_conv_factor_hard for err_pos in
                self.band_flux_err[1]]]
        else:
            self.hard_dets = [np.nan for det in self.flux]
            self.hard_errors = [[np.nan for det in self.flux], [np.nan for det in self.flux]]

        self.hardness_ratio = [(hard - soft)/(hard + soft) for (soft, hard) in zip(self.soft_dets, self.hard_dets)]
        low_soft = np.where(np.array(self.soft_dets) - np.array(self.soft_errors[0]) < 0, 0,
                            np.array(self.soft_dets) - np.array(self.soft_errors[0]))
        low_hard = np.where(np.array(self.hard_dets) - np.array(self.hard_errors[0]) < 0, 0,
                            np.array(self.hard_dets) - np.array(self.hard_errors[0]))
        up_soft = np.where(np.array(self.soft_dets) + np.array(self.soft_errors[1]) < 0, 0,
                           np.array(self.soft_dets) + np.array(self.soft_errors[1]))
        up_hard = np.where(np.array(self.hard_dets) + np.array(self.hard_errors[1]) < 0, 0,
                        np.array(self.hard_dets) + np.array(self.hard_errors[1]))
        self.hardness_err = [[hr - (hard - soft)/(hard + soft) for (soft, hard, hr) in zip(up_soft, low_hard, self.hardness_ratio)],
                            [(hard - soft)/(hard + soft) - hr for (soft, hard, hr) in zip(low_soft, up_hard, self.hardness_ratio)]]


    def swift_modif(self, flux:list, flux_err:list) -> None:
        """
        Modifies and updates the Swift data attributes of the source object.

        This method is specifically designed to handle data from the Swift telescope.
        It involves updating stacked flux, error values, and checking for variability 
        based on the given flux and error parameters.

        Parameters:
        flux (list of float): List of flux values of the source.
        flux_err (list of float): Corresponding errors of the flux values.

        The method updates attributes related to Swift telescope data, including checking
        for variability and adjusting minimum and maximum flux values.

        """

        self.swift_stacked_flux = self.swift_data.stacked_flux
        self.swift_stacked_flux_err = self.swift_data.stacked_flux_err
        self.swift_stacked_times = self.swift_data.stacked_times
        self.swift_stacked_variable = False
        
        self.min_upper, self.max_lower = 1, 0
        self.var = 1
        if len(flux) > 0:
            min_upper = min(np.array(flux) + np.array(flux_err[1]))
            max_lower = max(np.array(flux) - np.array(flux_err[0]))
        if self.swift_stacked_flux != []:
            stacked_min = min(np.array(self.swift_stacked_flux)+np.array(self.swift_stacked_flux_err[1]))
            if stacked_min<0.5*self.min_upper:
                self.swift_stacked_variable = True
            self.min_upper = min(self.min_upper, stacked_min)
        if len(flux) + len(self.swift_stacked_flux) > 1:
            self.var = max_lower/min_upper


class MasterSource:
    """
    A class representing a master source, consolidating data from various sources.

    This class aggregates and processes data related to astronomical sources from 
    different catalogs. It handles the removal of redundant data, calculation of various 
    properties like hardness ratio, variability ratios, and maintains a comprehensive 
    record of the source's observations across different telescopes.

    Attributes:
    src_id (int): Unique identifier for the master source.
    sources (dict): Dictionary storing Source objects from different catalogs.
    sources_flux (np.ndarray): Array of flux values from all sources.
    sources_error_bar (np.ndarray): Array of flux error values from all sources.
    sources_time_steps (list): List of time steps corresponding to each observation.
    sources_var (list): List of variability flags for each observation.
    tab_hr (list): List of hardness ratios.
    tab_hr_err (list): List of errors in hardness ratios.
    never_on_axis_xmm (bool): Indicates if source never appeared on-axis in XMM observations.
    has_short_term_var (bool): Flag for the presence of short-term variability.
    min_time (float): Minimum observation time across all sources.
    max_time (float): Maximum observation time across all sources.
    min_upper (float): Minimum upper limit of the source's flux.
    max_lower (float): Maximum lower limit of the source's flux.
    var_ratio (float): Variability ratio of the source.
    var_amplitude (float): Variability amplitude.
    var_significance (float): Significance of the variability.
    hr_min (float): Minimum hardness ratio.
    hr_max (float): Maximum hardness ratio.
    hr_var (float): Variability in hardness ratio.
    hr_var_signif (float): Significance of hardness ratio variability.
    xmm_ul (list): List of upper limits from XMM observations.
    xmm_ul_dates (list): Dates corresponding to XMM upper limits.
    xmm_ul_obsids (list): Observation IDs for XMM upper limits.
    slew_ul, slew_ul_dates, slew_ul_obsids, chandra_ul, chandra_ul_dates (lists): Similar attributes for other telescopes.
    ra (float): Right ascension of the source.
    dec (float): Declination of the source.
    pos_err (float): Positional error of the source.
    glade_distance (list): Distances from GLADE catalog.
    simbad_type (str): Source type from the SIMBAD database.
    has_sdss_widths (bool): Flag indicating the presence of SDSS widths.

    Methods:
    __init__(self, src_id, sources_table, ra, dec, poserr): Initializes the MasterSource object.

    The class primarily focuses on aggregating and processing the source data for 
    further analysis, particularly in the context of astronomical research.
    """
    
    def __init__(self, src_id, sources_table, ra, dec, poserr) -> None:
        self.src_id = src_id
        self.sources, self.sources_flux, self.sources_error_bar = {}, [], [[], []]
        self.sources_time_steps, self.sources_var = [], []
        self.tab_hr, self.tab_hr_err = [], [[], []]
        self.never_on_axis_xmm, self.has_short_term_var = False, False
        self.min_time, self.max_time = 60000, 0
        
        for source in sources_table:
            if ("XMM" in self.sources.keys()) and (source.catalog == "Stacked"):
                # We remove the Stacked detection that correspond to a clean XMM detection
                xmm_obs_id = self.sources["XMM"].obs_ids
                stacked_obs_id = source.obs_ids
                new_det_ind = [item for item in range(len(stacked_obs_id)) if stacked_obs_id[item] not in xmm_obs_id]
                
                source.flux = source.flux[new_det_ind]
                source.flux_err[0] = source.flux_err[0][new_det_ind]
                source.flux_err[1] = source.flux_err[1][new_det_ind]
                
                source.time_steps = np.array(source.time_steps)[new_det_ind]
                source.obs_ids = np.array(source.obs_ids)[new_det_ind]
                
                source.hardness_ratio = np.array(source.hardness_ratio)[new_det_ind]
                source.hardness_err[0] = np.array(source.hardness_err[0])[new_det_ind]
                source.hardness_err[1] = np.array(source.hardness_err[1])[new_det_ind]
                
                source.band_flux = source.band_flux[new_det_ind]
                source.band_flux_err[0] = source.band_flux_err[0][new_det_ind]
                source.band_flux_err[1] = source.band_flux_err[1][new_det_ind]
                
            source.master_source = self
            self.sources[source.catalog] = source
            
            for (flux, flux_err_neg, flux_err_pos, time_step) in zip(source.flux, source.flux_err[0], source.flux_err[1], source.time_steps):
                self.sources_flux.append(flux)
                self.sources_error_bar[0].append(flux_err_neg)
                self.sources_error_bar[1].append(flux_err_pos)
                self.sources_var.append(source.var)
                self.sources_time_steps.append(time_step)
            self.tab_hr += list(source.hardness_ratio)
            self.tab_hr_err[0] += list(source.hardness_err[0])
            self.tab_hr_err[1] += list(source.hardness_err[1])
            
            for (flux, flux_err_neg, flux_err_pos, start, stop) in zip(source.swift_stacked_flux, source.swift_stacked_flux_err[0], source.swift_stacked_flux_err[1], source.swift_stacked_times[0], source.swift_stacked_times[1]):
                self.sources_flux.append(flux)
                self.sources_error_bar[0].append(flux_err_neg)
                self.sources_error_bar[1].append(flux_err_pos)
                self.min_time = min(start, self.min_time)
                self.max_time = max(stop, self.max_time)
                self.sources_time_steps.append((start + stop)/2)
                
            if source.xmm_offaxis!=[]:
                if np.nanmin(source.xmm_offaxis)>1:
                    self.never_on_axis_xmm = True
            if source.time_steps!=[]:
                self.min_time = min(min(source.time_steps), self.min_time)
                self.max_time = max(max(source.time_steps), self.max_time)
            for var_flag in source.short_term_var:
                if var_flag>0:
                    self.has_short_term_var=True
        self.sources_flux = np.array(self.sources_flux)
        self.sources_error_bar = np.array(self.sources_error_bar)
        
        self.min_upper, self.max_lower, self.var_ratio = 1, 0, 1
        self.var_amplitude, self.var_significance = 0, 0

        if len(self.sources_flux)>0 and (not np.isnan(self.sources_flux).all()):
            min_upper_ind = np.argmin(self.sources_flux + self.sources_error_bar[1])
            self.min_upper = (self.sources_flux + self.sources_error_bar[1])[min_upper_ind]
            max_lower_tab = np.where(self.sources_flux - self.sources_error_bar[0]>0,
                                     self.sources_flux - self.sources_error_bar[0],
                                     self.sources_flux)
            max_lower_ind = np.argmax(max_lower_tab)
            self.max_lower = max_lower_tab[max_lower_ind]
            self.var_ratio = self.max_lower/self.min_upper
            self.var_amplitude = self.max_lower - self.min_upper
            self.var_optimistic = self.sources_flux[max_lower_ind]/self.sources_flux[min_upper_ind]
            self.var_significance = self.var_amplitude/np.sqrt(self.sources_error_bar[1][max_lower_ind]**2 + self.sources_error_bar[0][min_upper_ind]**2)
            #self.frac_var = np.sqrt((np.var(self.sources_flux, ddof=1)-np.mean(np.array(self.sources_error_bar)**2))/(np.mean(self.sources_flux)**2))
            
        self.hr_min, self.hr_max = np.nan, np.nan
        self.hr_var, self.hr_var_signif = np.nan, np.nan
        
        if len(self.tab_hr) > 1 and (not np.isnan(self.tab_hr).all()) and (not np.isnan(self.tab_hr_err).all()):
            index_hr_min = np.nanargmin(np.array(self.tab_hr) + np.array(self.tab_hr_err[1]))
            index_hr_max = np.nanargmax(np.array(self.tab_hr) - np.array(self.tab_hr_err[0]))
            self.hr_min = (np.array(self.tab_hr) + np.array(self.tab_hr_err[1]))[index_hr_min]
            self.hr_max = (np.array(self.tab_hr) - np.array(self.tab_hr_err[0]))[index_hr_max]
            self.hr_var = self.hr_max - self.hr_min
            if self.tab_hr_err[1][index_hr_min]**2 + self.tab_hr_err[0][index_hr_max]**2 > 0:
                self.hr_var_signif = self.hr_var/np.sqrt(self.tab_hr_err[1][index_hr_min]**2 + self.tab_hr_err[0][index_hr_max]**2)
            else:
                self.hr_var_signif = np.nan
            
        self.xmm_ul, self.xmm_ul_dates, self.xmm_ul_obsids = [], [], []

        self.slew_ul, self.slew_ul_dates, self.slew_ul_obsids = [], [], []

        self.chandra_ul, self.chandra_ul_dates = [], []

        self.ra, self.dec = float(ra), float(dec)
        self.pos_err = float(poserr)

        self.glade_distance=[]

        self.simbad_type = ''
        self.has_sdss_widths = False

          
# ---------------------------------------------- #

# --------------- CatalogMatch --------------- #

class CatalogMatch:
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

        unique_sources_dict = f.unique_dict(nearby_sources_table[column_name["source_name"]])
        
        new_row = []
        for index, name in enumerate(nearby_sources_table[column_name["source_name"]]):
            if name not in unique_sources_dict.keys():
                new_row.append((name, index))
            
        sources_dict = f.insert_row(unique_sources_dict=unique_sources_dict, new_row=new_row)
        
        
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
                if f.ang_separation(object_position, sources_position_1[number]) < radius:
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
                if f.ang_separation(object_position, sources_position_2[number]) < radius:
                    nearby_sources_table_2.add_row(small_table_2[number])
                    
            column_name = {"source_name": "Chandra_IAUNAME",
                           "right_ascension": "RA",
                           "declination": "DEC",
                           "catalog_name": "Chandra"}
                        
            unique_table = f.create_unique_sources_catalog(nearby_sources_table=nearby_sources_table_2, column_name=column_name)
            
            os_dictionary = simulation_data["os_dictionary"]
            catalog = os_dictionary["cloesest_dataset_path"]
            chandra_catalog = os.path.join(catalog, "unique_table.fits").replace("\\", "/")
            
            unique_table.write(chandra_catalog, format='fits', overwrite=True)
            topcat_path = os.path.join(os_dictionary["active_workflow"], 'softwares/topcat-extra.jar').replace("\\", "/")
            command = f"java -jar {topcat_path} {chandra_catalog}"
            subprocess.run(command)
        
            unique_table = f.create_unique_sources_catalog(nearby_sources_table=nearby_sources_table_2, column_name=column_name)
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
            count_rates, self.nearby_sources_table = f.count_rates(self.nearby_sources_table, self.model_dictionary, telescop_data)
            # f.py_to_xlsx(excel_data_path=excel_data_path, count_rates=count_rates, object_data=object_data, args=(catalog, key), radius=radius)
        elif platform.system() == "Windows":
            count_rates, self.nearby_sources_table = f.xlsx_to_py(excel_data_path=excel_data_path, nearby_sources_table=self.nearby_sources_table, object_data=object_data, args=(catalog, key), radius=radius.value)
        else:
            sys.exit()
            
        simulation_data['nearby_sources_table'] = self.nearby_sources_table
        
        f.nominal_pointing_info(simulation_data, self.nearby_sources_position)
        self.OptimalPointingIdx, self.SRCoptimalSEPAR, self.SRCoptimalRATES, self.vector_dictionary = f.calculate_opti_point(simulation_data, self.nearby_sources_position)
        f.optimal_point_infos(self.vector_dictionary, self.OptimalPointingIdx, self.SRCoptimalRATES)
        f.data_map(simulation_data, self.vector_dictionary, self.OptimalPointingIdx, self.nearby_sources_position)
        
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
            SRCnominalDIST = f.ang_separation(SRCposition, SkyCoord(ra=optipoint_ra, dec=optipoint_dec, unit=u.deg)).arcmin
            distance = np.append(distance, SRCnominalDIST)
            vignetting = calculate_vignetting_factor(SRCnominalDIST, EffArea, OffAxisAngle)
            vignetting_factor = np.append(vignetting_factor, vignetting)
    
        optimal_pointing_point = SkyCoord(ra=optipoint_ra, dec=optipoint_dec, unit=u.deg)
        psr_position = SkyCoord(ra=object_data['object_position'].ra, dec=object_data['object_position'].dec, unit=u.deg)
        distance_psr_to_optipoint = f.ang_separation(psr_position, optimal_pointing_point).arcmin
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
            master_sources = f.load_master_sources(output_name)
            f.master_source_plot(master_sources=master_sources, simulation_data=simulation_data, number_graph=len(master_sources))
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
   

# -------------------------------------------- #