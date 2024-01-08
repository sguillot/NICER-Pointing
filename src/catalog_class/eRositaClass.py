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
class eRositaCatalog:
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

    