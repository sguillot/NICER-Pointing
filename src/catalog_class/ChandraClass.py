# ------------------------------ #
        # Python's packages
        
from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from typing import Tuple, Dict, Union
from termcolor import colored
from scipy.optimize import curve_fit
from astropy.units import Quantity
from tqdm import tqdm

# ---------- import function ---------- #

import function.calculation_function as c_f
import function.unique_function as u_f

# ------------------------------------- #

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import catalog_information as dict_cat
import pyvo as vo

# ------------------------------ #

class ChandraCatalog:
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
            if c_f.ang_separation(object_position, sources_position[number]) < radius:
                nearby_sources_table.add_row(small_table[number])
                
        column_name = {"source_name": "Chandra_IAUNAME", 
                       "right_ascension": self.ra,
                       "declination": self.dec,
                       "catalog_name": "Chandra"}
            
        if len(nearby_sources_table) != 0:
            unique_table = u_f.create_unique_sources_catalog(nearby_sources_table=nearby_sources_table, column_name=column_name)
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

