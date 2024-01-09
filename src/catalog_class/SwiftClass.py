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

class SwiftCatalog:
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

