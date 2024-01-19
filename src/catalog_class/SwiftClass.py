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

import os
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

class SwiftCatalog:
    """
    A specialized class for processing and analyzing data from the Swift X-ray Telescope and related astronomical sources.

    This class is meticulously designed to handle data from the Swift X-ray Telescope, a pivotal instrument in contemporary astrophysics for studying gamma-ray bursts and other high-energy phenomena. It adeptly extends its analytical prowess to include data from related astronomical sources, offering comprehensive functionalities. Key features include the identification of nearby celestial sources, evaluation of their variability, computation of photon indices, and the development of intricate models for thorough scientific analysis.

    SwiftCatalog simplifies the intricate process of deciphering complex X-ray data, transforming it into intelligible and scientifically valuable insights. Its ability to integrate and analyze data from associated sources significantly broadens its scope in astronomical research and exploration.

    Important:
        - Tailored specifically for Swift X-ray data, the class provides advanced functionalities like detailed variability analysis, precise photon index computation, and sophisticated source modeling.
        - It is an indispensable tool in the realm of X-ray astrophysics, facilitating deep explorations into the high-energy aspects of the universe.
        - Users leveraging this class should ideally possess a foundational knowledge of X-ray astronomy and the data structures typical in this scientific domain.

    Note:
        - SwiftCatalog transcends being merely a data processing utility; it represents a conduit linking raw space observations with profound astronomical insights.
        - Its integration with additional astronomical data sources paves the way for expansive studies and cross-referencing, enriching the field of astrophysics.
        - While primarily designed for Swift data, the methodologies and approaches are versatile, applicable to data from other X-ray observatories.
        - This class is a crucial component of a larger toolkit aimed at augmenting our comprehension of space through X-ray astrophysics.

    Example:
        Instantiate the SwiftCatalog with necessary parameters:
        
        >>> swift_catalog = SwiftCatalog("path/to/swift_catalog.fits", 5*u.arcmin, simulation_data, user_table)

    SwiftCatalog embodies the fusion of comprehensive observational data with advanced analytical methodologies, propelling astrophysical research into new frontiers of discovery and understanding.
    """

    def __init__(self, catalog_path: str, radius: Quantity, simulation_data: dict, user_table: Table) -> None:
        """
        Initializes the Swift class with specific catalog data, search radius, and simulation parameters.

        This constructor loads the Swift astronomical catalog from a given path, searches for nearby sources within a 
        specified radius around a provided celestial object, and initiates several analyses including neighborhood 
        visualization, photon index calculation, and model dictionary creation.

        Args:
            catalog_path (str): The file path to the Swift catalog.
            radius (Quantity): The radius within which to search for nearby sources, specified as an astropy Quantity.
            simulation_data (dict): A dictionary containing simulation data, including details about the celestial object.
            user_table (Table): An astropy Table provided by the user, not used in the current implementation.

        Attributes:
        
        .. attribute:: ra
            :type: str
            :value: RA
            
            Right ascension column name as specified in dict_cat.dictionary_coord for Swift.
        
        .. attribute:: dec
            :type: str
            :value: DEC
            
            Declination column name as specified in dict_cat.dictionary_coord for Swift.
            
        .. attribute:: swi_catalog
            :type: Table
            
            The Swift catalog data as an astropy Table, loaded from the specified catalog_path.
            
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
            - Catalog Initialization: The Swift astronomical catalog is loaded from the specified path, setting the foundation for all subsequent analyses.
            - Nearby Sources Identification: The constructor employs a search radius to identify astronomical sources in close proximity to a specified celestial object, enhancing the relevance of the analysis.
            - Data Analysis Initiation: It initiates key analyses such as neighborhood visualization, which helps in understanding the spatial distribution of astronomical objects, and photon index calculation, crucial for characterizing the spectral properties of these objects.
            - Model Dictionary Creation: A model dictionary for the sources is created, which is vital for detailed astrophysical analysis and interpretation.
            - Versatility in Data Handling: The class is designed to handle user-provided data tables, although this functionality is not implemented in the current version, indicating potential for future expansion and customization.
            - Coordinate System Specification: The class initializes specific coordinate columns ('RA' and 'DEC') for Swift data, ensuring accurate spatial referencing and alignment with astronomical standards.

        The SwiftCatalog class, through its constructor, lays the groundwork for sophisticated astronomical data analysis, enabling researchers to unlock deeper insights from Swift telescope observations.
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
        Opens a FITS file containing the Swift catalog and converts it into an astropy Table.

        This method is essential for loading the Swift catalog data. It utilizes the astropy library to handle FITS 
        files, a common format for storing astronomical data. By converting the FITS file into an astropy Table, 
        the data becomes more accessible and easier to manipulate for further analysis.

        Args:
            catalog_path (str): The file path to the Swift catalog FITS file. This path should point to a valid 
                                FITS file containing the Swift catalog data.

        Returns:
            Table: An astropy Table containing the data from the Swift catalog. This table structure allows for 
                convenient handling and analysis of the catalog data, utilizing the rich feature set of astropy.
        """
        with fits.open(catalog_path, memmap=True) as data:
            return Table(data[1].data)

            
    def find_nearby_sources(self, radius: Quantity, object_data: dict) -> Tuple[Table, SkyCoord]:
        """
        Identifies and compiles a list of astronomical sources from the Swift catalog that are located near a specified 
        celestial object.

        This method is pivotal for studies focusing on specific regions of the sky. It filters the Swift catalog to 
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
                various details from the Swift catalog for each identified source.
                - A SkyCoord object containing the coordinates of these nearby sources.

        Note:
            - The method dynamically adjusts the search area based on the specified radius, ensuring comprehensive 
            coverage of the surrounding region.
            - It is capable of handling additional user-provided data (not implemented in the current version), 
            which could be integrated in future enhancements.
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
            if c_f.ang_separation(object_position, src_position[number]) < radius:
                nearby_sources_table.add_row(small_table[number])

        column_name = {"source_name": "Swift_IAUNAME", 
                       "right_ascension": self.ra,
                       "declination": self.dec,
                       "catalog_name": "Swift"}
        
        if len(nearby_sources_table) != 0:
            try:
                unique_table = u_f.create_unique_sources_catalog(nearby_sources_table=nearby_sources_table, column_name=column_name)
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
            
        Here is an example of the plot create by this method:
        
        .. image:: C:/Users/plamb_v00y0i4/OneDrive/Bureau/Optimal_Pointing_Point_Code/modeling_result/PSR_J0437-4715/Swift/img/neighbourhood_of_PSR_J0437-4715.png
            :align: center
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
        Visualizes the interpolation of photon index values for sources in the Swift catalog.

        This method plots the relationship between photon index values and energy bands for sources in the Swift catalog. 
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
        Calculates the photon index for a source in the Swift catalog using an absorbed power-law model.

        This method fits the observed flux data of a source across various energy bands to an absorbed power-law model. 
        The fitting process takes into account both the flux values and their respective errors, resulting in a photon 
        index that characterizes the source's spectral properties.

        Args:
            table (Table): An astropy Table containing the Swift catalog data.
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
        Computes the photon index and hydrogen column density (Nh) for each source in the Swift catalog's nearby sources table.

        This method calculates the photon index using an absorbed power-law model and assigns a default Nh value to each source. 
        It also includes a visualization step for the photon indices and updates the nearby sources table with these calculated values.

        Returns:
            List[float]: A list of photon index values for each source in the nearby sources table.

        Note:
            - A default Nh value of 3e20 is used if specific Nh data is not available.
            - The photon index calculation defaults to a value of 1.7 if the computed index is negative or zero.
            - Visualization of photon indices aids in understanding the spectral characteristics of the sources.

        The method plays a crucial role in spectral analysis, allowing for a detailed examination of the high-energy properties 
        of astronomical sources in the Swift catalog.
        """
        key = "Swift"
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

