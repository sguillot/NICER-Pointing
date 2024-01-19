# ------------------------------ #
        # Python's packages
        
from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from typing import Tuple, Dict, Union, List
from termcolor import colored
from scipy.optimize import curve_fit
from astropy.units import Quantity
from tqdm import tqdm

# ---------- import function ---------- #

import function.unique_function as u_f
import function.calculation_function as c_f

# ------------------------------------- #

import sys
import numpy as np
import matplotlib.pyplot as plt
import catalog_information as dict_cat

# ------------------------------ #

# ---------- for documentation ---------- #

# import src.function.unique_function as u_f
# import src.function.calculation_function as c_f
# import src.catalog_information as dict_cat

# --------------------------------------- #

class eRositaCatalog:
    """
    A class dedicated to handling and analyzing astronomical data from the eRosita catalog.

    The eRositaCatalog class offers a suite of methods for efficiently managing and interpreting data from the eRosita space telescope. 
    This includes capabilities for opening the catalog, identifying nearby astronomical sources, visualizing spatial distributions 
    of these sources, and conducting detailed spectral analysis through photon index calculations. Additionally, the class provides 
    functionality for constructing detailed model dictionaries for each identified source.

    Important:
        - This class is specifically tailored for the eRosita data, providing astronomers and researchers with tools 
          necessary for X-ray astronomy data analysis.
        - Advanced functionalities such as variability assessment and photon index calculation are integral to the 
          class, facilitating a deeper understanding of the high-energy universe.
        - The class bridges the gap between raw observational data and meaningful astronomical insights, playing a 
          critical role in astrophysical research.

    Note:
        - The eRositaCatalog class is part of a broader toolkit aimed at enhancing the analysis and interpretation of 
          X-ray data from various space telescopes.
        - Integration with other data sources and catalogs is feasible, allowing for comprehensive and comparative 
          astronomical studies.
        - Users should possess a basic understanding of X-ray astronomy and data formats common in the field for optimal use.

    Example:
        Instantiate the eRositaCatalog with necessary parameters:
        
        >>> eRo_catalog = eRositaCatalog("path/to/eRosita_catalog.fits", 5*u.arcmin, simulation_data, user_table)
    """

    def __init__(self, catalog_path: str, radius: Quantity, simulation_data: dict, user_table: Table) -> None:
        """
        Initializes the eRosita class with the specified catalog, search radius, simulation data, and user table.

        This constructor loads the eRosita astronomical catalog from the given path, searches for nearby sources within a 
        specified radius around a provided celestial object, and performs various analyses such as neighborhood 
        visualization, photon index calculation, and model dictionary creation.

        Args:
            catalog_path (str): The file path to the eRosita catalog.
            radius (Quantity): The radius within which to search for nearby sources, specified as an astropy Quantity.
            simulation_data (dict): A dictionary containing simulation data, including details about the celestial object.
            user_table (Table): An astropy Table provided by the user, not used in the current implementation.

        Attributes:
    
        .. attribute:: ra
            :type: str
            :value: RA
            
            Right ascension column name as specified in dict_cat.dictionary_coord for eRosita.
        
        .. attribute:: dec
            :type: str
            :value: DEC
            
            Declination column name as specified in dict_cat.dictionary_coord for eRosita.
            
        .. attribute:: eRo_catalog
            :type: Table
            
            The eRosita catalog data as an astropy Table, loaded from the specified catalog_path.
            
        .. attribute:: nearby_sources_table
            :type: Table
            
            Table of sources found near the specified celestial object.
            
        .. attribute:: nearby_sources_position
            :type: List[SkyCoord]
            
            Sky coordinates of the sources found near the specified celestial object.
            
        .. attribute:: photon_index
            :type: List[float] 
            
            List of photon index values for sources.
            
        .. attribute:: model_dictionary
            :type: Dict[str, Dict[str, Union[str, float]]]
        
            Dictionary of model parameters for each source.
            
        Important:
            - Catalog Initialization: The eRosita astronomical catalog is loaded from the specified path, setting the foundation for all subsequent analyses.
            - Nearby Sources Identification: The constructor employs a search radius to identify astronomical sources in close proximity to a specified celestial object, enhancing the relevance of the analysis.
            - Data Analysis Initiation: It initiates key analyses such as neighborhood visualization, which helps in understanding the spatial distribution of astronomical objects, and photon index calculation, crucial for characterizing the spectral properties of these objects.
            - Model Dictionary Creation: A model dictionary for the sources is created, which is vital for detailed astrophysical analysis and interpretation.
            - Versatility in Data Handling: The class is designed to handle user-provided data tables, although this functionality is not implemented in the current version, indicating potential for future expansion and customization.
            - Coordinate System Specification: The class initializes specific coordinate columns ('RA' and 'DEC') for eRosita data, ensuring accurate spatial referencing and alignment with astronomical standards.

        The eRositaCatalog class, through its constructor, lays the groundwork for sophisticated astronomical data analysis, enabling researchers to unlock deeper insights from eRosita telescope observations.
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
        Opens a FITS file containing the eRosita catalog and converts it into an astropy Table.

        This method is essential for loading the eRosita catalog data. It utilizes the astropy library to handle FITS 
        files, a common format for storing astronomical data. By converting the FITS file into an astropy Table, 
        the data becomes more accessible and easier to manipulate for further analysis.

        Args:
            catalog_path (str): The file path to the eRosita catalog FITS file. This path should point to a valid 
                                FITS file containing the eRosita catalog data.

        Returns:
            Table: An astropy Table containing the data from the eRosita catalog. This table structure allows for 
                convenient handling and analysis of the catalog data, utilizing the rich feature set of astropy.
        """

        with fits.open(catalog_path, memmap=True) as data:
            return Table(data[1].data)

        
    def find_nearby_sources(self, radius: Quantity, simulation_data: dict) -> Tuple[Table, SkyCoord]:
        """
        Identifies and compiles a list of astronomical sources from the eRosita catalog that are located near a specified 
        celestial object.

        This method is pivotal for studies focusing on specific regions of the sky. It filters the eRosita catalog to 
        identify sources within a defined radius from the target celestial object, using its astronomical coordinates. 
        The method enhances the data's relevance to specific astronomical queries or studies.

        Args:
            radius (Quantity): The radius within which to search for nearby sources. This radius should be specified 
                            as an astropy Quantity object, indicating both the value and the unit (e.g., degrees, 
                            arcminutes, etc.).
            object_data (dict): A dictionary containing essential information about the celestial object. Key elements 
                                include 'object_name' for identification and 'object_position', an astropy SkyCoord 
                                object representing the object's coordinates.

        Returns:
            Tuple[Table, SkyCoord]: A tuple containing two elements:
                - An astropy Table listing the sources found near the specified celestial object. This table includes 
                various details from the eRosita catalog for each identified source.
                - A SkyCoord object containing the coordinates of these nearby sources.

        Note:
            - The method dynamically adjusts the search area based on the specified radius, ensuring comprehensive 
            coverage of the surrounding region.
            - It is capable of handling additional user-provided data (not implemented in the current version), 
            which could be integrated in future enhancements.
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
            if c_f.ang_separation(object_position, src_position[number]) < radius:
                nearby_sources_table.add_row(small_table[number])
        
        column_name = {"source_name": "eRosita_IAUNAME",
                       "right_ascension": self.ra,
                       "declination": self.dec,
                       "catalog_name": "eRosita"}
        
        if len(nearby_sources_table) != 0:
            try:
                unique_table = u_f.create_unique_sources_catalog(nearby_sources_table=nearby_sources_table, column_name=column_name)
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

        Args:
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
        Visualizes the interpolation of photon index values for sources in the eRosita catalog.

        This method plots the relationship between photon index values and energy bands for sources in the eRosita catalog. 
        It showcases the spectral characteristics of these sources through a plot that combines error bars for observed 
        fluxes and absorbed power-law models.

        Args:
            optimization_parameters (list): A list of tuples containing optimization parameters (fluxes, errors, power-law values) for each source.
            photon_index (list): A list of photon index values for the sources.
            key (str): The key to retrieve energy band center values from 'dict_cat.dictionary_catalog'.

        The plot is on a logarithmic scale, displaying energy in keV and flux in erg/cm²/s/keV. It provides a visual summary 
        of how photon index values vary across different energy bands, aiding in the spectral analysis of the sources.
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
        

    def optim_index(self, table, key, index) -> Tuple[List, Tuple]:
        """
        Calculates the photon index for a source in the eRosita catalog using an absorbed power-law model.

        This method fits the observed flux data of a source across various energy bands to an absorbed power-law model. 
        The fitting process takes into account both the flux values and their respective errors, resulting in a photon 
        index that characterizes the source's spectral properties.

        Args:
            table (Table): An astropy Table containing the eRosita catalog data.
            key (str): The key to access necessary data from 'dict_cat.dictionary_catalog'.
            index (int): The index of the source in the table for photon index calculation.

        Returns:
            Tuple[float, Tuple]: The photon index of the source and a tuple of optimization parameters, including energy band 
                                centers, observed fluxes, flux errors, and fitted power-law values.

        The photon index is a critical parameter in X-ray astronomy, offering insights into the energy distribution of 
        celestial sources. The method employs curve fitting techniques, defaulting to a standard value in case of 
        fitting failures.
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


    def get_phoindex_nh(self) -> List[float]:
        """
        Computes the photon index and hydrogen column density (Nh) for each source in the eRosita catalog's nearby sources table.

        This method calculates the photon index using an absorbed power-law model and assigns a default Nh value to each source. 
        It also includes a visualization step for the photon indices and updates the nearby sources table with these calculated values.

        Returns:
            List[float]: A list of photon index values for each source in the nearby sources table.

        Note:
            - A default Nh value of 3e20 is used if specific Nh data is not available.
            - The photon index calculation defaults to a value of 1.7 if the computed index is negative or zero.
            - Visualization of photon indices aids in understanding the spectral characteristics of the sources.

        The method plays a crucial role in spectral analysis, allowing for a detailed examination of the high-energy properties 
        of astronomical sources in the eRosita catalog.
        """
        key = "eRosita"
        photon_index_list, parameters_list, nh_list = [], [], []

        for index in range(len(self.nearby_sources_table)):
            nh_list.append(3e20)
            photon, params = self.optim_index(table=self.nearby_sources_table, key=key, index=index)
            photon_index_list.append(photon if photon > 0.0 else 1.7)
            parameters_list.append(params)
        
        self.visualization_inter(optimization_parameters=parameters_list, photon_index=photon_index_list, key=key)
        
        self.nearby_sources_table["Photon Index"] = photon_index_list
        self.nearby_sources_table["Nh"] = nh_list
        
        return photon_index_list
    
    
    def dictionary_model(self) -> Dict[str, Dict[str, Union[str, float]]]:
        """
        Constructs a dictionary with detailed modeling parameters for each source in the nearby sources table.

        This method creates a comprehensive dictionary where each entry corresponds to a source, detailing its astrophysical 
        model type, model values (like photon index), observed flux, and hydrogen column density.

        Returns:
            Dict[str, Dict[str, Union[str, float]]]: A dictionary with each source represented by a unique key (e.g., 'src_0'), containing a dictionary of its modeling parameters.

        Note:
            - Currently, only the 'power' model type is implemented, with placeholders for other models like 'black_body' and 'temp'.
            - Assumes photon index ('Photon Index') and hydrogen column density ('Nh') values are pre-computed and present in the nearby sources table.

        This method provides a structured approach to organizing and accessing model data for each source, supporting further 
        analysis and interpretation in X-ray astrophysics.
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

    