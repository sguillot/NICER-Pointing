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
    A class to manage and analyze data from the XMM-Newton X-ray observatory.

    This class provides functionality to load, analyze, and visualize data from XMM catalogs. 
    It finds nearby sources in the catalog based on a specified radius and performs analysis 
    on photon indices, variability, and neighborhood of objects.

    Attributes:
        xmm_catalog (Table): The loaded XMM catalog data.
        nearby_sources_table (Table): A table of sources found within the specified radius.
        nearby_sources_position (List[Tuple]): List of positions of the nearby sources.
        xmm_dr11_catalog (Table): The loaded XMM DR11 catalog data.
        x2a_catalog (Table): The loaded XMM to Athena catalog data.
        index_table (Table): A table containing information related to photon indices.
        variability_table (Table): A table containing information related to source variability.
        model_dictionary (Dict): A dictionary containing model-related information.

    Methods:
        open_catalog(catalog_path: str) -> Table:
            Loads and returns the data from a given catalog file path.
        
        find_nearby_sources(radius: float, dictionary: Dict, user_table: Table) -> Tuple[Table, List[Tuple]]:
            Finds and returns nearby sources within the specified radius and the positions of those sources.

        optimization_phoindex(number: int):
            Optimizes the photon index for a given number of sources.

        visualization_interp(optimization_parameters: List[float], photon_index: float):
            Visualizes the interpolated photon indices based on optimization parameters.

        get_phoindex_nh() -> Tuple[Table, Table]:
            Gets and returns the photon index and hydrogen column density (nh) from the catalog data.

        variability_table(dictionary: Dict) -> Table:
            Generates and returns a table containing variability information of sources.

        neighbourhood_of_object(radius: float, dictionary: Dict):
            Analyzes and displays information about the neighbourhood of a given object within the specified radius.

        dictionary_model() -> Dict:
            Generates and returns a dictionary containing model-related information.
    """


    def __init__(self, catalog_path: str, radius: Quantity, dictionary: dict, user_table: Table, os_dictionary: dict) -> None:
        """
        Initializes the XmmCatalog instance.

        This constructor initializes an XmmCatalog object by loading catalog data from the specified path,
        finding nearby sources within the given radius, analyzing photon indices, variability, and neighborhood 
        of objects, and generating a model dictionary.

        Parameters:
            catalog_path (str): The file path to the XMM catalog to be loaded.
            radius (float): The radius within which to search for nearby sources.
            dictionary (Dict): A dictionary containing user-defined parameters for analysis.
            user_table (Table): A user-defined table for additional data and analysis.

        Attributes:
            xmm_catalog (Table): The loaded XMM catalog data.
            nearby_sources_table (Table): A table of sources found within the specified radius.
            nearby_sources_position (List[Tuple]): List of positions of the nearby sources.
            xmm_dr11_catalog (Table): The loaded XMM DR11 catalog data.
            x2a_catalog (Table): The loaded XMM to Athena catalog data.
            index_table (Table): A table containing information related to photon indices.
            variability_table (Table): A table containing information related to source variability.
            model_dictionary (Dict): A dictionary containing model-related information.
        """
        self.xmm_catalog = self.open_catalog(catalog_path=catalog_path)
        self.nearby_sources_table, self.nearby_sources_position = self.find_nearby_sources(radius=radius, dictionary=dictionary, user_table=user_table)
        self.os_dictionary = os_dictionary
        
        test_dr11_path = "catalog_data/4XMM_DR11cat_v1.0.fits"
        test_x2a_path = "catalog_data/xmm2athena_D6.1_V3.fits"
        xmm_dr11_path = f.get_valid_file_path(test_dr11_path)
        x2a_path = f.get_valid_file_path(test_x2a_path)
        
        self.xmm_dr11_catalog = self.open_catalog(catalog_path=xmm_dr11_path)
        self.x2a_catalog = self.open_catalog(catalog_path=x2a_path)
        self.nearby_sources_table, self.index_table = self.get_phoindex_nh()
        self.variability_table = self.variability_table(dictionary)
        self.neighbourhood_of_object(radius=radius, dictionary=dictionary, os_dictionary=os_dictionary)
        self.model_dictionary = self.dictionary_model()
        
    
    def open_catalog(self, catalog_path: str) -> Table:
        """
        Opens and reads the catalog data from the specified file path.

        This method opens a FITS file from the given path, reads the data from the first HDU extension,
        converts it into an Astropy Table, and returns the resulting table.

        Parameters:
            catalog_path (str): The file path to the catalog to be opened and read.

        Returns:
            result_table (astropy.table.Table): The catalog data converted into an Astropy Table.
        """
        with fits.open(catalog_path, memmap=True) as data:
            result_table = Table(data[1].data)
            data.close()
            return result_table
        
    
    def find_nearby_sources(self, radius: Quantity, dictionary: dict, user_table: Table) -> Tuple[Table, SkyCoord]:
        """
        Searches and returns nearby sources from the XMM catalog based on a specified radius and position.

        This method searches the XMM catalog for sources within a specified radius from a given object position.
        It first filters the catalog based on RA (Right Ascension) and DEC (Declination) within a pointing area,
        and then refines the search based on angular separation. It also allows for additional sources from a user-defined
        table to be included in the search. The results, if found, are returned as tables of nearby sources and their positions.

        Parameters:
            radius (float or astropy.units.Quantity): The search radius around the object position.
            dictionary (Dict): A dictionary containing parameters such as 'object_position' (SkyCoord) and 'object_name' (str).
            user_table (astropy.table.Table): An additional table of user-defined sources.

        Returns:
            nearby_src_table (astropy.table.Table): A table of sources found within the specified radius.
            nearby_src_position (astropy.coordinates.SkyCoord): Sky coordinates of the nearby sources.

        Raises:
            Exception: An exception is raised and the program exits if no sources are found or if an error occurs.
        """   
        pointing_area = radius + 5*u.arcmin
        name = dictionary["object_name"]
        object_position = dictionary['object_position']
        min_ra, max_ra = object_position.ra - pointing_area, object_position.ra + pointing_area
        min_dec, max_dec = object_position.dec - pointing_area, object_position.dec + pointing_area
        
        small_table = Table(names=self.xmm_catalog.colnames,
                            dtype=self.xmm_catalog.dtype)
        nearby_src_table = Table(names=self.xmm_catalog.colnames,
                                 dtype=self.xmm_catalog.dtype)
        
        print(fr"{colored('Reducing XMM catalog...', 'yellow')} to fov of {radius.value + 5 } arcmin")
        for number in tqdm(range(len(self.xmm_catalog))):
            if min_ra/u.deg < self.xmm_catalog['SC_RA'][number] < max_ra/u.deg and min_dec/u.deg < self.xmm_catalog['SC_DEC'][number] < max_dec/u.deg:
                small_table.add_row(self.xmm_catalog[number])
                
        if len(user_table) == 0:
            src_position = SkyCoord(ra=small_table['SC_RA'], dec=small_table['SC_DEC'], unit=u.deg)
            print(f"{colored(f'Find sources close to {name} with XMM catalog', 'blue')}")
            for number in tqdm(range(len(small_table))):
                if f.ang_separation(object_position, src_position[number]) < radius:
                    nearby_src_table.add_row(small_table[number])
            nearby_src_position = SkyCoord(ra=nearby_src_table['SC_RA'], dec=nearby_src_table['SC_DEC'], unit=u.deg)

        else:
            for number in range(len(user_table)):
                small_table.add_row(user_table[number])
                
            src_position = SkyCoord(ra=small_table['SC_RA'], dec=small_table['SC_DEC'], unit=u.deg)
            print(f"{colored(f'Find sources close to {name} with XMM catalog', 'blue')}")
            for number in tqdm(range(len(small_table))):
                if f.ang_separation(object_position, src_position[number]) < radius:
                    nearby_src_table.add_row(small_table[number])
            nearby_src_position = SkyCoord(ra=nearby_src_table['SC_RA'], dec=nearby_src_table['SC_DEC'], unit=u.deg)
                
        try:
            if len(nearby_src_table) != 0:
                print((f"We have detected {len(nearby_src_table)} sources close to {dictionary['object_name']}.\n"))
                return nearby_src_table, nearby_src_position
            else:
                print(f"No sources detected close to {dictionary['object_name']}.")
                sys.exit()
        except Exception as error:
            print(f"An error occured : {error}")
            
       
    def optimization_phoindex(self, number: int) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[float, float]]:
        """
        Optimizes and returns the parameters of both power-law and absorbed power-law models.

        This method fits the observed flux data to both power-law and absorbed power-law models using
        a non-linear least-squares minimization. The optimized parameters for both models along with
        the photon indices are returned.

        Parameters:
            number (int): The index of the source in the nearby_sources_table.

        Returns:
            optimization_parameters (tuple): A tuple containing energy bands, observed flux,
                                            observed flux errors, and predicted fluxes for both models.
            photon_index (tuple): A tuple containing photon indices for both models.
        """    
        
        def absorbed_power_law(x, constant, gamma):
            sigma = np.array([1e-20, 5e-21, 1e-22, 1e-23, 1e-24], dtype=float)
            return (constant * x ** (-gamma)) * (np.exp(-sigma * 3e20))
        
        
        key = 'XMM'
        energy_band = dict_cat.dictionary_catalog[key]["energy_band_center"]
        energy_band_half_width = dict_cat.dictionary_catalog[key]["energy_band_half_width"]
        tab_width = 2 * energy_band_half_width
        
        band_flux_obs_name = dict_cat.dictionary_catalog[key]["band_flux_obs"]
        band_flux_obs_err_name = dict_cat.dictionary_catalog[key]["band_flux_obs_err"]
        
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
        Visualizes the interpolation of photon index for the sources.

        This method plots the observed flux along with the predicted flux from both power-law
        and absorbed power-law models. It creates a grid of subplots showing each source's data.

        Parameters:
            optimization_parameters (list of tuples): A list containing tuples of energy bands,
                                                    observed flux, observed flux errors, and
                                                    predicted fluxes for both models for each source.
            photon_index (list of tuples): A list containing tuples of photon indices for both models
                                        for each source.
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
            # axes.step(energy_band, flux_obs, where='pre', label=f"$\Gamma$ = {absorb_pho_index:.8f}")
    
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
        Computes and visualizes the photon index and hydrogen column density (Nh) for each source.

        This method searches for each source in the xmm_dr11_catalog and x2a_catalog, computes the
        hydrogen column density (Nh), and optimizes the photon index using the optimization_phoindex method.
        The results are then visualized using the visualization_interp method and added to the nearby_sources_table.

        Returns:
            nearby_sources_table (astropy.table.Table): The updated nearby sources table with added columns
                                                        for Photon Index and Nh.
            index_table (astropy.table.Table): A table containing indices of sources in the various catalogs.
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
    
    
    def variability_table(self, dictionary: dict) -> Table:
        """
        Generates a table containing information about the variability of nearby sources.

        Parameters:
        - dictionary (dict): A dictionary containing object information.
            - "object_name" (str): Name of the object of interest.

        Returns:
        - variability_table (astropy.table.Table): A table containing information about the nearby sources,
        such as index, name (IAUNAME), right ascension (SC_RA), declination (SC_DEC), fractional variability (SC_FVAR),
        and a boolean indicating whether the source is present in Xmm2Athena catalog (IN_X2A).

        The function also prints the number of variable sources detected close to the object and the number
        of those sources that are present in Xmm2Athena catalog.
        """
        name = dictionary["object_name"]

        index_array, iauname_array, sc_ra_array = np.array([], dtype=int), np.array([], dtype=str), np.array([], dtype=float)
        sc_dec_array, sc_fvar_array, in_x2a_array = np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=bool)

        for number in range(len(self.nearby_sources_table)):
            if not np.isnan(self.xmm_dr11_catalog["SC_FVAR"][self.index_table["Index in xmm_dr11"][number]]):

                index_array = np.append(index_array, self.index_table["Index in nearby_sources_table"][number])
                iauname_array = np.append(iauname_array, self.nearby_sources_table["IAUNAME"][number])
                sc_ra_array = np.append(sc_ra_array, self.nearby_sources_table["SC_RA"][number])
                sc_dec_array = np.append(sc_dec_array, self.nearby_sources_table["SC_DEC"][number])
                sc_fvar_array = np.append(sc_fvar_array, self.nearby_sources_table["SC_FVAR"][number])

                if self.index_table["Index in x2a"][number] != "No data found":
                    in_x2a_array = np.append(in_x2a_array, True)
                else:
                    in_x2a_array = np.append(in_x2a_array, False)

        column_names = ["INDEX", "IAUNAME", "SC_RA", "SC_DEC", "SC_FVAR", "IN_X2A"]
        data_array = [index_array, iauname_array, sc_ra_array, sc_dec_array, sc_fvar_array, in_x2a_array]
        variability_table = Table()

        for data, name in zip(data_array, column_names):
            variability_table[name] = data

        message_xmm = f"Among {len(self.nearby_sources_table)} sources detected close to {name}, {len(index_array)} of them are variable. Using DR13 Catalog."
        print(message_xmm)
        message_xmm2ath = f"Among {len(index_array)} variable sources, {list(variability_table['IN_X2A']).count(True)} are in Xmm2Athena and {list(variability_table['IN_X2A']).count(False)} are not in Xmm2Athena. "    
        print(message_xmm2ath)

        return variability_table
    
    
    def neighbourhood_of_object(self, radius: Quantity, dictionary: dict, os_dictionary: dict) -> None:
        """
        Visualizes the neighborhood of the given object.

        Parameters:
        - radius (float): The radius within which to search for neighboring sources.
        - dictionary (dict): A dictionary containing object information.
            - "object_name" (str): Name of the object of interest.
            - "object_position" (SkyCoord): The sky coordinates of the object.

        Returns:
        None. The function generates plots depicting the neighboring sources and their variability.

        The function queries catalogs to get sources in the vicinity of the input object and then
        plots two graphs: one showing all sources close to the object and another differentiating 
        between variable and invariable sources.
        """
        print("\n")
        name = dictionary['object_name']
        obj_ra, obj_dec = dictionary['object_position'].ra, dictionary['object_position'].dec
        try:
            result = ESASky.query_object_catalogs(position=name, catalogs="XMM-EPIC")
            xmm_epic = Table(result[0])
            xmm_obs_id = list(xmm_epic["observation_id"])
            result_fits_images = ESASky.get_images(observation_ids=xmm_obs_id[0], radius=radius, missions="XMM")
        except Exception as error:
            print(f"{colored('An error occured : ', 'red')} {error}")
            result_fits_images = {}
            
        
        ra_in_x2a = [ra for index, ra in enumerate(self.variability_table['SC_RA']) if self.variability_table['IN_X2A'][index] == True]
        dec_in_x2a = [dec for index, dec in enumerate(self.variability_table['SC_DEC']) if self.variability_table['IN_X2A'][index] == True]
        ra_in_dr11 = [ra for index, ra in enumerate(self.variability_table['SC_RA']) if self.variability_table['IN_X2A'][index] == False]
        dec_in_dr11 = [dec for index, dec in enumerate(self.variability_table['SC_DEC']) if self.variability_table['IN_X2A'][index] == False]
        invar_ra = [ra for ra in self.nearby_sources_table["SC_RA"] if ra not in self.variability_table["SC_RA"]]
        invar_dec = [dec for dec in self.nearby_sources_table["SC_DEC"] if dec not in self.variability_table["SC_DEC"]]
    
        figure = plt.figure(figsize=(17, 8))
        figure.suptitle(f"Neighbourhood of {name}", fontsize=20)
        
        if result_fits_images == {}:
            figure.text(0.5, 0.04, 'Right Ascension [deg]', ha='center', va='center')
            figure.text(0.04, 0.5, 'Declination [deg]', ha='center', va='center', rotation='vertical')
            
            axes_0 = figure.add_subplot(121)
            axes_0.invert_xaxis()
            axes_0.scatter(list(self.nearby_sources_table['SC_RA']), list(self.nearby_sources_table['SC_DEC']), s=20, color='darkorange', label=f"Sources : {len(self.nearby_sources_table)}")
            axes_0.scatter(obj_ra, obj_dec, s=100, color='red', marker="*", label=f"{name}")
            axes_0.legend(loc='upper right', ncol=2, fontsize=7)
            axes_0.set_title(f"Sources close to {name}")
            
            axes_1 = figure.add_subplot(122)
            axes_1.invert_xaxis()
            axes_1.scatter(invar_ra, invar_dec, color='black', s=20, label=f"Invariant sources, {len(invar_ra)}")
            axes_1.scatter(obj_ra, obj_dec, color='red', marker='x', s=100, label=f"Position of {name}")
            axes_1.scatter(ra_in_x2a, dec_in_x2a, color='darkorange', marker="*", label=f"Variable sources in X2A, {len(ra_in_x2a)}")
            axes_1.scatter(ra_in_dr11, dec_in_dr11, color='royalblue', marker="*", label=f"Variable sources not in X2A, {len(ra_in_dr11)}")
            axes_1.legend(loc="lower right", ncol=2)
            axes_1.set_title(f"Variable and invariable sources close to {name} ")
            
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
            axes_0.scatter(list(self.nearby_sources_table["SC_RA"]), list(self.nearby_sources_table["SC_DEC"]), s=30, transform=axes_0.get_transform('fk5'), facecolors='none', edgecolors='orange', label=f"Sources : {len(self.nearby_sources_table)}")
            axes_0.scatter(obj_ra, obj_dec, s=100, color='red', marker="*", transform=axes_0.get_transform('fk5'), facecolors='none', edgecolors='red', label=f"{name}")
            axes_0.legend(loc='upper right', ncol=2, fontsize=7)
            xlim, ylim = plt.xlim(), plt.ylim()
            value_x, value_y = 180, 180
            axes_0.set_xlim(xmin=xlim[0]+value_x, xmax=xlim[1]-value_x)
            axes_0.set_ylim(ymin=ylim[0]+value_y, ymax=ylim[1]-value_y)
            axes_0.set_xlabel(" ")
            axes_0.set_ylabel(" ")
            axes_0.set_title(f"Sources close to {name}")
            
            axes_1 = figure.add_subplot(122, projection=_wcs_, sharex=axes_0, sharey=axes_0)
            axes_1.imshow(image, cmap='gray', origin='lower', norm=norm, interpolation='nearest', aspect='equal')
            axes_1.scatter(invar_ra, invar_dec, s=30, transform=axes_1.get_transform('fk5'), facecolors='none', edgecolors='orange', label=f"Invar src : {len(invar_ra)}")
            axes_1.scatter(ra_in_dr11, dec_in_dr11, s=30, transform=axes_1.get_transform('fk5'), facecolors='none', edgecolors='blue', label=f"Var src not in x2a : {len(ra_in_dr11)} sources")
            axes_1.scatter(ra_in_x2a, dec_in_x2a, s=30, transform=axes_1.get_transform('fk5'), facecolors='none', edgecolors='hotpink', label=f"Var src in x2a : {len(ra_in_x2a)} sources")
            axes_1.scatter(obj_ra, obj_dec, s=100, marker="*", transform=axes_1.get_transform('fk5'), facecolors='none', edgecolors='red', label=f"{name}")
            axes_1.legend(loc='upper right', ncol=2, fontsize=7)
            axes_1.set_xlabel(" ")
            axes_1.set_ylabel(" ")
            axes_1.set_title(f"Variable and invariable sources close to {name} ")
            
        plt.savefig(os.path.join(os_dictionary["img"], f"neighbourhood_of_{name}.png".replace(" ", "_")))
        plt.show()
        print("\n")


    def dictionary_model(self) -> Dict[str, Dict[str, Union[str, float]]]:
        """
        Generates a dictionary containing model information for each source in the nearby_sources_table.

        Returns:
        - model_dictionary (dict): A dictionary containing model information for each source. Each entry has the following format:
            - "src_i" (dict): Where i is the index of the source in the nearby_sources_table.
                - "model" (str): The model used (e.g., 'power', 'black_body', 'temp').
                - "model_value" (float): The value associated with the model (e.g., Photon Index).
                - "flux" (float): The flux of the source in the XMM-EPIC band.
                - "column_density" (float): The column density (Nh) of the source.

        The function currently supports only the 'power' model.
        """
        model_dictionary = {}
        number_source = len(self.nearby_sources_table)

        model = np.array([], dtype=str)
        model_value = np.array([], dtype=float)
        xmm_flux = np.array([self.nearby_sources_table["SC_EP_8_FLUX"][item] for item in range(number_source)], dtype=float)
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
    A class to perform various analyses and visualizations using the Chandra X-ray Observatory data.

    Attributes
    ----------
    chandra_catalog : astropy.table.Table
        The catalog of Chandra data loaded from the specified path.
    nearby_sources_table : astropy.table.Table
        Table of sources detected close to a given object.
    nearby_sources_position : astropy.coordinates.SkyCoord
        Sky coordinates of the nearby sources.
    cone_search_catalog : astropy.table.Table
        Catalog obtained by performing a cone search.
    cs_nearby_sources_position : astropy.coordinates.SkyCoord
        Sky coordinates of the sources obtained from cone search.
    cs_nearby_sources_table : astropy.table.Table
        Table of sources obtained from the cone search catalog.
    neighbourhood_of_object : method
        Performs and visualizes the neighborhood analysis of an object.
    photon_index : ndarray
        Array containing the photon index of the sources.
    model_dictionary : dict
        Dictionary containing models and parameters for each source.

    Methods
    -------
    open_catalog(catalog_path: str) -> astropy.table.Table
        Load and return the Chandra catalog from the given path.
    load_cs_catalog(radius: float, dictionary: dict) -> astropy.table.Table
        Load and return the cone search catalog for a given radius and dictionary.
    find_nearby_sources(radius: float, dictionary: dict, user_table: bool) -> Tuple[astropy.table.Table, astropy.coordinates.SkyCoord]
        Find and return nearby sources and their positions for the given radius and dictionary.
    cone_catalog() -> astropy.table.Table
        Generate and return a table of sources from the cone search catalog.
    neighbourhood_of_object(radius: float, dictionary: dict) -> None
        Visualize the neighborhood of an object using the provided radius and dictionary.
    get_pho_index(number: int) -> float
        Retrieve and return the photon index of a source given its number in the catalog.
    visualization_interp(optimization_parameters: list, photon_index: float) -> None
        Visualize interpolations based on the optimization parameters and photon index.
    power_law_pho_index() -> ndarray
        Calculate and return the photon index based on a power-law model.
    dictionary_model() -> dict
        Generate and return a dictionary containing models and parameters for each source.
    """


    def __init__(self, catalog_path: str, radius: Quantity, dictionary: dict, user_table: Table, os_dictionary: dict) -> None:
        """
        Initialize the Chandra class by loading catalogs, finding nearby sources, 
        and performing various analyses and visualizations.

        Parameters
        ----------
        catalog_path : str
            The path to the Chandra catalog file.
        radius : float
            The radius to use for searches and analyses.
        dictionary : dict
            A dictionary containing parameters or configurations for searches and analyses.
        user_table : bool
            A flag indicating whether to use a user-defined table in searches and analyses.

        Attributes
        ----------
        chandra_catalog : astropy.table.Table
            The catalog of Chandra data loaded from the specified path.
        nearby_sources_table : astropy.table.Table
            Table of sources detected close to a given object.
        nearby_sources_position : astropy.coordinates.SkyCoord
            Sky coordinates of the nearby sources.
        cone_search_catalog : astropy.table.Table
            Catalog obtained by performing a cone search.
        cs_nearby_sources_position : astropy.coordinates.SkyCoord
            Sky coordinates of the sources obtained from cone search.
        cs_nearby_sources_table : astropy.table.Table
            Table of sources obtained from the cone search catalog.
        photon_index : ndarray
            Array containing the photon index of the sources.
        model_dictionary : dict
            Dictionary containing models and parameters for each source.
        """
        self.chandra_catalog = self.open_catalog(catalog_path=catalog_path)
        self.cone_search_catalog = self.load_cs_catalog(radius=radius, dictionary=dictionary)
        self.cs_nearby_sources_position = SkyCoord(ra=list(self.cone_search_catalog['ra']), dec=list(self.cone_search_catalog['dec']), unit=u.deg)
        self.cone_search_catalog = self.variability_column()
        self.cone_search_catalog = self.threshold(self.cone_search_catalog)
        
        self.nearby_sources_table, self.nearby_sources_position = self.find_nearby_sources(radius=radius, dictionary=dictionary, user_table=user_table)
        # self.corrected_mean_flux()
        self.neighbourhood_of_object(radius=radius, dictionary=dictionary, os_dictionary=os_dictionary)
        self.cs_photon_index, self.photon_index = self.get_phoindex_nh()
        self.cs_model_dictionary, self.model_dictionary = self.dictionary_model()
        

    def open_catalog(self, catalog_path: str) -> Table:
        """
        Open and read the Chandra catalog file from the given path.

        Parameters
        ----------
        catalog_path : str
            The path to the Chandra catalog file.

        Returns
        -------
        result_table : astropy.table.Table
            The table containing data read from the catalog file.
        """
        with fits.open(catalog_path, memmap=True) as data:
            result_table = Table(data[1].data)
            data.close()
            return result_table
        
        
    def load_cs_catalog(self, radius: Quantity, dictionary: dict) -> Table:
        """
        Load the cone search catalog for a given object and search radius.

        Parameters
        ----------
        radius : float
            The search radius in degrees.
        dictionary : dict
            Dictionary containing parameters, including 'object_name' to search for.

        Returns
        -------
        cone_search_catalog : astropy.table.Table
            The cone search catalog containing data for the specified object.
        """
        cone = vo.dal.SCSService('http://cda.cfa.harvard.edu/csc2scs/coneSearch') 
        name = SkyCoord.from_name(dictionary['object_name'])
        cone_search_catalog = cone.search(pos=name, radius=radius, verbosity=3)
        return cone_search_catalog


    def find_nearby_sources(self, radius: Quantity, dictionary: dict, user_table: Table) -> Tuple[Table, SkyCoord]:
        """
        Find and return sources close to a given object from the Chandra catalog.

        Parameters
        ----------
        radius : float
            The search radius in degrees.
        dictionary : dict
            Dictionary containing parameters, including 'object_position' for the search.
        user_table : bool
            A flag indicating whether to use a user-defined table.

        Returns
        -------
        nearby_src_table : astropy.table.Table
            Table of sources detected close to the given object.
        nearby_src_position : astropy.coordinates.SkyCoord
            Sky coordinates of the nearby sources.

        Raises
        ------
        Exception
            Raises an exception if an error occurs during the process.
        """
        field_of_view = radius + 5*u.arcmin
        name = dictionary["object_name"]
        object_position = dictionary['object_position']

        min_ra, max_ra = object_position.ra - field_of_view, object_position.ra + field_of_view
        min_dec, max_dec = object_position.dec - field_of_view, object_position.dec + field_of_view

        small_table = Table(names= self.chandra_catalog.colnames,
                            dtype= self.chandra_catalog.dtype)

        nearby_sources_table = Table(names= self.chandra_catalog.colnames,
                                     dtype= self.chandra_catalog.dtype)
        
        print(fr"{colored('Reducing Chandra catalog...', 'yellow')} to fov of {radius.value + 5 } arcmin")
        for number in tqdm(range(len(self.chandra_catalog))):
            if min_ra/u.deg < self.chandra_catalog["RA"][number] < max_ra/u.deg and min_dec/u.deg < self.chandra_catalog["DEC"][number] < max_dec/u.deg:
                small_table.add_row(self.chandra_catalog[number])
                
        src_position = SkyCoord(ra=small_table['RA'], dec=small_table['DEC'], unit=u.deg)

        print(f"{colored(f'Find sources close to {name} with Chandra catalog', 'blue')}")
        for number in tqdm(range(len(small_table))):
            if f.ang_separation(object_position, src_position[number]) < radius:
                nearby_sources_table.add_row(small_table[number])
                
        column_name = {"source_name": "Chandra_IAUNAME", 
                       "right_ascension": "RA",
                       "declination": "DEC",
                       "catalog_name": "Chandra"}
            
        if len(nearby_sources_table) != 0:
            unique_table = f.create_unique_sources_catalog(nearby_sources_table=nearby_sources_table, column_name=column_name)
        else:
            print("Nearby sources table from Chandra catalog is empty.")
            sys.exit()
            
        nearby_src_position = SkyCoord(ra=unique_table['RA'], dec=unique_table['DEC'], unit=u.deg)
                
        try :
            if len(unique_table) != 0:
                print((f"We have detected {len(unique_table)} sources close to {dictionary['object_name']}"))
                return unique_table, nearby_src_position
            else:
                print(f"No sources detected close to {dictionary['object_name']}.")
        except Exception as error:
            print(f"An error occured : {error}")
            
    
    def variability_column(self) -> Table:
        """
        Create a catalog table by combining and processing data from a cone search catalog.

        Returns
        -------
        self.cs_catalog : astropy.table.Table
            Table containing 'IAUNAME', 'RA', 'DEC', and 'VAR' columns derived from the cone search catalog.
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
                
        # data_list = [cone_catalog['name'], cone_catalog['ra'], cone_catalog['dec'], var_column]

        # self.cs_catalog = Table(names=['IAUNAME', 'RA', 'DEC', 'VAR'],
        #                         data=data_list)
        
        return cone_catalog
    
    
    def neighbourhood_of_object(self, radius: Quantity, dictionary: dict, os_dictionary: dict) -> None:
        """
        Plot a visualization of the neighborhood of a given astronomical object based on data from the Chandra and cone search catalogs.

        Parameters
        ----------
        radius : float
            The search radius in degrees.
        dictionary : dict
            Dictionary containing parameters, including 'object_name' and 'object_position' for the search.

        Returns
        -------
        None
            This function plots a graph and does not return any value.
        """
        name = dictionary["object_name"]
        cs_csc_ra = np.array(list(self.cone_search_catalog['ra']), dtype=float)
        cs_csc_dec = np.array(list(self.cone_search_catalog['dec']), dtype=float)
        
        csc_ra = np.array(self.nearby_sources_table['RA'], dtype=float)
        csc_dec = np.array(self.nearby_sources_table['DEC'], dtype=float)
        csc_ra = list(set(csc_ra))
        csc_dec = list(set(csc_dec))
        
        figure_1, axes = plt.subplots(2, 2, figsize=(15, 8), sharex=True, sharey=True)
        figure_1.suptitle(f"Neighbourhood of {dictionary['object_name']}, radius = {radius}", fontsize=20)
        figure_1.text(0.5, 0.04, 'Right Ascension [deg]', ha='center', va='center', fontsize=16)
        figure_1.text(0.04, 0.5, 'Declination [deg]', ha='center', va='center', rotation='vertical', fontsize=16)

        
        ax00 = axes[0][0]
        ax00.scatter(csc_ra, csc_dec, s=10, c='black', marker="*", label=f"Sources close to {dictionary['object_name']} : {len(csc_ra)}")
        ax00.scatter(dictionary['object_position'].ra, dictionary['object_position'].dec, marker='x', c='red', label=f"{dictionary['object_name']}")
        ax00.legend(loc='upper right')
        ax00.set_title("With chandra.fits")
        
        ax01 = axes[0][1]
        ax01.scatter(cs_csc_ra, cs_csc_dec, s=10, c='black', marker="*", label=f"Sources close to {dictionary['object_name']} : {len(cs_csc_ra)}")
        ax01.scatter(dictionary['object_position'].ra, dictionary['object_position'].dec, marker='x', c='red', label=f"{dictionary['object_name']}")
        ax01.legend(loc='upper right')
        ax01.set_title("With cone search")
        
        ax10 = axes[1][0]

        cs_ra_var = [ra for index, ra in enumerate(list(self.cone_search_catalog['ra'])) if self.cone_search_catalog['Variability'][index] != 0.0]
        cs_ra_invar = [ra for index, ra in enumerate(list(self.cone_search_catalog['ra'])) if self.cone_search_catalog['Variability'][index] == 0.0]

        cs_dec_var = [dec for index, dec in enumerate(list(self.cone_search_catalog['dec'])) if self.cone_search_catalog['Variability'][index] != 0.0]
        cs_dec_invar = [dec for index, dec in enumerate(list(self.cone_search_catalog['dec'])) if self.cone_search_catalog['Variability'][index] == 0.0]

        ax11 = axes[1][1]
        ax11.scatter(cs_ra_var, cs_dec_var, s=10, c='darkorange', marker='*', label=f"Var src : {len(cs_ra_var)} sources")
        ax11.scatter(cs_ra_invar, cs_dec_invar, s=10, c='blue', marker='*', label=f"Invar src : {len(cs_ra_invar)} sources")
        ax11.scatter(dictionary['object_position'].ra, dictionary['object_position'].dec, marker='+', s=50, c='red', label=f"{dictionary['object_name']}")
        ax11.legend(loc="upper right", ncol=2)
        
        plt.savefig(os.path.join(os_dictionary["img"], f"neighbourhood_of_{name}.png".replace(" ", "_")))
        plt.show()


    def threshold(self, cone_search_catalog):
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
        Visualize the interpolation of photon indices.

        Parameters
        ----------
        optimization_parameters : List[Tuple[np.ndarray]]
            List of tuples containing arrays of energy bands, observed fluxes, non-absorbed power law values, and absorbed power law values.
        photon_index : List[Tuple[float, float]]
            List of tuples containing the non-absorbed and absorbed photon indices respectively.

        Returns
        -------
        None
            This function plots a graph and does not return any value.
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
            # axes.step(energy_band, flux_obs, where='pre', label=f"$\Gamma$ = {absorb_pho_index:.8f}")
    
        axes.legend(loc="upper left", ncol=4, fontsize=6)
        axes.loglog()
        
        plt.show()


    def get_photon_index(self, key, table, index):
        
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
        Create a dictionary of models and their parameters for each source in the cone search catalog.

        Returns
        -------
        model_dictionary : Dict[str, Dict[str, Union[str, float]]]
            Dictionary where each key is a source identifier and the corresponding value is another dictionary containing model details.
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
    

    def corrected_mean_flux(self):
        key = "Chandra"
        flux_obs = dict_cat.dictionary_catalog[key]["flux_obs"]

        data = []
        for item in range(len(self.nearby_sources_table)):
            if not isinstance(self.nearby_sources_table[flux_obs][item], np.nan):
                data.append(self.nearby_sources_table[flux_obs][item])
        min_value = np.min(data)
        flux = list(self.nearby_sources_table[flux])
        corrected_flux = np.nan_to_num(flux, nan=min_value)

        self.nearby_sources_table[flux_obs] = corrected_flux


class Swift:
    """
    This class represents a catalog of astronomical sources observed by the Swift satellite. It provides methods to find
    nearby sources, visualize the neighborhood of a target object, and construct a dictionary model.

    Attributes:
    - swi_catalog (Table): The catalog of Swift sources.
    - nearby_sources_table (Table): A table of sources located nearby the target object.
    - nearby_sources_position (SkyCoord): Sky coordinates of the nearby sources.
    - model_dictionary (dict): A dictionary model containing information about the sources.
    """


    def __init__(self, catalog_path: str, radius: Quantity, simulation_data: dict, user_table: Table) -> None:
        """
        Initializes the Swift class with a catalog path, search radius, dictionary of target object information, 
        and a user-defined table.

        Parameters:
        - catalog_path (str): Path to the Swift catalog file.
        - radius (float): Search radius (in degrees) for finding nearby sources.
        - dictionary (dict): A dictionary containing information about the target object.
        - user_table (Table): A user-defined table of sources.
        """
        self.swi_catalog = self.open_catalog(catalog_path)
        self.nearby_sources_table, self.nearby_sources_position = self.find_nearby_sources(radius=radius, object_data=simulation_data["object_data"])
        
        self.neighbourhood_of_object(radius=radius, simulation_data=simulation_data)
        self.photon_index = self.get_phoindex_nh()
        self.model_dictionary = self.dictionary_model()


    def open_catalog(self, catalog_path: str)-> Table:
        """
        Opens and reads a FITS catalog file and returns it as an Astropy Table.

        Parameters:
        - catalog_path (str): Path to the Swift catalog file.

        Returns:
        - Table: The catalog of Swift sources.
        """
        with fits.open(catalog_path, memmap=True) as data:
            result_table = Table(data[1].data)
            data.close()
            return result_table
        
        
    def find_nearby_sources(self, radius: Quantity, object_data: dict) -> Tuple[Table, SkyCoord]:
        """
        Finds sources in the Swift catalog that are within a certain radius from the target object.

        Parameters:
        - radius (float): Search radius (in degrees) for finding nearby sources.
        - dictionary (dict): A dictionary containing information about the target object.

        Returns:
        - Tuple[Table, SkyCoord]: A table of nearby sources and their sky coordinates.
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
            if min_ra/u.deg < self.swi_catalog["RA"][number] < max_ra/u.deg and min_dec/u.deg < self.swi_catalog["DEC"][number] < max_dec/u.deg:
                small_table.add_row(self.swi_catalog[number])
                
        src_position = SkyCoord(ra=small_table['RA'], dec=small_table['DEC'], unit=u.deg)
                
        print(f"{colored(f'Find sources close to {name} with Swift catalog', 'blue')}")
        for number in tqdm(range(len(small_table))):
            if f.ang_separation(object_position, src_position[number]) < radius:
                nearby_sources_table.add_row(small_table[number])

        column_name = {"source_name": "Swift_IAUNAME", 
                       "right_ascension": "RA",
                       "declination": "DEC",
                       "catalog_name": "Swift"}
        
        if len(nearby_sources_table) != 0:
            try:
                unique_table = f.create_unique_sources_catalog(nearby_sources_table=nearby_sources_table, column_name=column_name)
            except Exception as error:
                print(f"{colored('An error occured : ', 'red')} {error}")
        else:
            print("Nearby sources table from Swift catalog is empty.")
            sys.exit()
            
        nearby_src_position = SkyCoord(ra=unique_table['RA'], dec=unique_table['DEC'], unit=u.deg)
                
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
        Visualizes the neighborhood of the target object and nearby sources.

        Parameters:
        - radius (float): Search radius (in degrees) for finding nearby sources.
        - dictionary (dict): A dictionary containing information about the target object.
        """
        object_data = simulation_data["object_data"]
        object_position = object_data['object_position']
        os_dictionary = simulation_data["os_dictionary"]
    
        swi_ra = self.nearby_sources_table['RA']
        swi_dec = self.nearby_sources_table['DEC']
        
        corrected_swi_ra = list(set(swi_ra))
        corrected_swi_dec = list(set(swi_dec))
        
        figure_1, axes = plt.subplots(1, 1, figsize=(12, 8))
        figure_1.suptitle(f"Neighbourhood of {object_data['object_name']}, radius = {radius}", fontsize=20)
        
        axes.scatter(corrected_swi_ra, corrected_swi_dec, s=30, facecolors='none', edgecolors='black', label=f"Sources : {len(self.nearby_sources_table)}")
        axes.scatter(object_position.ra, object_position.dec, c='red', s=100, marker='*', label=f"{object_data['object_name']}")
        axes.set_xlabel('Right Ascension')
        axes.set_ylabel('Declination')
        axes.legend(loc='upper right')
        
        img = os_dictionary["img"]
        img_path = os.path.join(img, f"neighbourhood_of_{object_data['object_name']}.png".replace(" ", "_"))
        plt.savefig(img_path)
        plt.show()
    
    
    def visualization_inter(self, optimization_parameters, photon_index, key):
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
            # axes.step(energy_band, flux_obs, where='pre', label=f"$\Gamma$ = {absorb_pho_index:.8f}")

        axes.legend(loc="upper left", ncol=4, fontsize=6)
        axes.loglog()
        
        plt.show()
        

    def get_photon_index(self, table, key, index):
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


    def get_phoindex_nh(self):
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
    This class represents a catalog of astronomical sources observed by the eRosita satellite. It provides methods to 
    find nearby sources, visualize the neighborhood of a target object, and construct a dictionary model.

    Attributes:
    - eRo_catalog (Table): The catalog of eRosita sources.
    - nearby_sources_table (Table): A table of sources located nearby the target object.
    - nearby_sources_position (SkyCoord): Sky coordinates of the nearby sources.
    - model_dictionary (dict): A dictionary model containing information about the sources.
    """


    def __init__(self, catalog_path: str, radius: Quantity, simulation_data: dict, user_table: Table) -> None:
        """
        Initializes the eRosita class with a catalog path, search radius, dictionary of target object information,
        and a user-defined table.

        Parameters:
        - catalog_path (str): Path to the eRosita catalog file.
        - radius (float): Search radius (in degrees) for finding nearby sources.
        - dictionary (dict): A dictionary containing information about the target object.
        - user_table (Table): A user-defined table of sources.
        """
        self.eRo_catalog = self.open_catalog(catalog_path)
        self.nearby_sources_table, self.nearby_sources_position = self.find_nearby_sources(radius=radius, simulation_data=simulation_data)
        
        self.neighbourhood_of_object(radius=radius, simulation_data=simulation_data)
        self.photon_index = self.get_phoindex_nh()
        self.model_dictionary = self.dictionary_model()
        
        
    def open_catalog(self, catalog_path: str) -> Table:
        """
        Opens and reads a FITS catalog file and returns it as an Astropy Table.

        Parameters:
        - catalog_path (str): Path to the eRosita catalog file.

        Returns:
        - Table: The catalog of eRosita sources.
        """
        with fits.open(catalog_path, memmap=True) as data:
            result_table = Table(data[1].data)
            data.close()
            return result_table
        
        
    def find_nearby_sources(self, radius: Quantity, simulation_data: dict) -> Tuple[Table, SkyCoord]:
        """
        Finds sources in the eRosita catalog that are within a certain radius from the target object.

        Parameters:
        - radius (float): Search radius (in degrees) for finding nearby sources.
        - dictionary (dict): A dictionary containing information about the target object.

        Returns:
        - Tuple[Table, SkyCoord]: A table of nearby sources and their sky coordinates.
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
            if min_ra/u.deg < self.eRo_catalog["RA"][number] < max_ra/u.deg and min_dec/u.deg < self.eRo_catalog["DEC"][number] < max_dec/u.deg:
                small_table.add_row(self.eRo_catalog[number])
                
        src_position = SkyCoord(ra=small_table['RA'], dec=small_table['DEC'], unit=u.deg)
                
        print(f"{colored(f'Find sources close to {object_name} with eRosita catalog', 'blue')}")
        for number in tqdm(range(len(small_table))):
            if f.ang_separation(object_position, src_position[number]) < radius:
                nearby_sources_table.add_row(small_table[number])
        
        column_name = {"source_name": "eRosita_IAUNAME",
                       "right_ascension": "RA",
                       "declination": "DEC",
                       "catalog_name": "eRosita"}
        
        if len(nearby_sources_table) != 0:
            try:
                unique_table = f.create_unique_sources_catalog(nearby_sources_table=nearby_sources_table, column_name=column_name)
            except Exception as error:
                print(f"{colored('An error occured : ', 'red')} {error}")
        else:
            print("Nearby sources table from Swift catalog is empty.")
            sys.exit()
        
           
        nearby_src_position = SkyCoord(ra=unique_table['RA'], dec=unique_table['DEC'], unit=u.deg)
                
        try :
            if len(unique_table) != 0:
                print((f"We have detected {len(unique_table)} sources close to {object_data['object_name']}"))
                return unique_table, nearby_src_position
            else:
                print(f"No sources detected close to {object_data['object_name']}.")
                sys.exit()
        except Exception as error:
            print(f"An error occured : {error}")
    
    
    def neighbourhood_of_object(self, dictionary: dict, radius: Quantity) -> None:
        """
        Visualizes the neighborhood of the target object and nearby sources.

        Parameters:
        - dictionary (dict): A dictionary containing information about the target object.
        - radius (float): Search radius (in degrees) for finding nearby sources.
        """
        object_position = dictionary['object_position']
    
        ero_ra = self.nearby_sources_table['RA']
        ero_dec = self.nearby_sources_table['DEC']
        
        figure_1, axes = plt.subplots(1, 1, figsize=(12, 8))
        figure_1.suptitle(f"Neighbourhood of {dictionary['object_name']}, radius = {radius}", fontsize=20)
        
        axes.scatter(ero_ra, ero_dec, c='black', s=1, marker='*', label=f"Sources close to {dictionary['object_name']}, nbr_src : {len(ero_ra)}")
        axes.scatter(object_position.ra, object_position.dec, c='red', s=100, marker='+', label=f"{dictionary['object_name']}")
        axes.set_xlabel('Right Ascension')
        axes.set_ylabel('Declination')
        axes.legend(loc='upper right')
        
        plt.show()


    def visualization_inter(self, optimization_parameters, photon_index, key):
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
            # axes.step(energy_band, flux_obs, where='pre', label=f"$\Gamma$ = {absorb_pho_index:.8f}")

        axes.legend(loc="upper left", ncol=4, fontsize=6)
        axes.loglog()
        
        plt.show()
        

    def get_photon_index(self, table, key, index):
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


    def get_phoindex_nh(self):
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
    
    def __init__(self, catalog_path: List, radius, simulation_data: dict, exp_time: int) -> None:
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
        
        self.total_spectra_1, self.total_spectra_2, self.total_var_spectra_1, self.total_var_spectra_2, self.instrument = self.modeling_source_spectra(simulation_data=simulation_data, exp_time=exp_time, key=(key_1, key_2))
        self.data_1, self.data_2 = self.total_spectra_plot(simulation_data=simulation_data, radius=radius.value, key=(key_1, key_2))
        
        self.write_txt_file(simulation_data=simulation_data, data_1=self.data_1, data_2=self.data_2, key=(key_1, key_2))
        
    
    def open_catalog(self, key: Tuple, path:Tuple, radius, object_data: Dict) -> Tuple[Table, Table]:
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
            # axes.step(energy_band, flux_obs, where='pre', label=f"$\Gamma$ = {absorb_pho_index:.8f}")
    
        axes.legend(loc="upper left", ncol=4, fontsize=6)
        axes.loglog()
        
        plt.show()
    
    
    def photon_index_nh_for_xmm(self, os_dictionary: Dict, xmm_index: int) -> Tuple[Table, List]:
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
        

    def threshold(self, cone_search_catalog) -> Table:
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
        
        
    def photon_index_nh_for_other_catalog(self, key, table) -> Table:
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
        figure_1.text(0.7, 0.04, 'Right Ascension [deg]', ha='center', va='center', fontsize=16)
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
        
    #TODO test with Linux
    def count_rate(self) -> Tuple[List, List]:
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
        
        active_workflow = simulation_data["os_dictionary"]["active_workflow"]
        object_data = simulation_data["object_data"]
        excel_data_path = os.path.join(active_workflow, "excel_data").replace("\\", "/")

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
    
    
    def modeling_source_spectra(self, simulation_data: Dict, exp_time: int, key: Tuple[str, str]) -> Tuple[List, List]:
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
    
    def __init__(self, flux, flux_err) -> None:
        self.flux = flux
        self.flux_err = flux_err


class SwiftData:
    
    def __init__(self, stacked_flux, stacked_flux_err, stacked_times):
        self.stacked_flux = stacked_flux
        self.stacked_flux_err = stacked_flux_err
        self.stacked_times = stacked_times
        
    # ------------------------------------------- #

    # --------------- Source Class --------------- #


class Source:
    
    def __init__(self, catalog, iau_name, flux, flux_err, time_steps, band_flux_data, **kwargs) -> None:
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


class MasterSource():
    
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
    A class designed to perform catalog matching for astronomical datasets, 
    particularly focused on X-ray astronomy sources. The class provides functionalities 
    to load, analyze, and match data from different astronomical catalogs, such as XMM-Newton 
    and Chandra. It also calculates various properties of the matched sources, including 
    photon index, count rates, vignetting factors, and more. 

    Attributes:
        nearby_sources_table_1 (Table): Table of nearby sources from the first catalog.
        nearby_sources_table_2 (Table): Table of nearby sources from the second catalog.
        mixed_index (List): List of indices of mixed sources from both catalogs.
        coordinates (List): Coordinates of the matched sources.
        photon_index_list (List[float]): List of photon indices for the sources.
        flag (List[Tuple]): Flags indicating the catalog(s) of origin for each source.
        nh_list (List[float]): List of column density values for the sources.
        model_dictionary (Dict): Dictionary containing model parameters for the sources.
        nearby_sources_table (Table): Combined table of nearby sources from both catalogs.
        nearby_sources_position (SkyCoord): Sky coordinates of the nearby sources.
        count_rate (List[float]): List of count rates for the sources.
        vignetting_factor (List[float]): List of vignetting factors for the sources.
        OptimalPointingIdx (int): Index of the optimal pointing for observation.
        vector_dictionary (Dict): Dictionary containing vector data related to the sources.
    
    Methods:
        load_catalog(catalog_name: Tuple[str, str], os_dictionary: Dict) -> Tuple[Table, Table]:
            Loads catalogs from specified paths and returns two tables of catalog data.

        load_cs_catalog(radius: float, object_data: Dict) -> Table:
            Loads a cone search catalog based on given radius and object data.

        find_nearby_sources(os_dictionary: Dict) -> Tuple[Dict, Dict]:
            Finds and returns nearby sources based on the operational system dictionary.

        get_mixed_coordinate(catalog_key: Tuple[str, str], table: Tuple[Table, Table]) -> Tuple[List, List]:
            Computes mixed coordinates from two given tables and catalog keys.

        neighbourhood_of_object(simulation_data: Dict, radius: float) -> Tuple[List, List]:
            Determines the neighborhood of an object based on simulation data and radius.

        get_photon_index(catalog_key: Tuple[str, str], table: Table, index: int, os_dictionary: Dict) -> float:
            Retrieves the photon index for a specific source in a catalog.

        get_mixed_photon_index(catalog_key: Tuple[str, str], table: Tuple[Table, Table], mixed_index: List[Tuple], row: int, os_dictionary: Dict) -> float:
            Calculates the photon index for a source present in both catalogs.

        get_total_photon_nh_list(os_dictionary: Dict) -> Tuple[List[float], List[Tuple], List[float]]:
            Compiles a complete list of photon indices, flags, and column densities for all sources.

        model_dictionary() -> Dict:
            Creates a dictionary containing the modeling data for each source.

        create_nearby_sources_table() -> Table:
            Generates a table containing data of all nearby sources.

        get_sources_position() -> SkyCoord:
            Returns the sky coordinates of all nearby sources.

        count_rate_SNR_map(simulation_data: Dict, radius: float) -> List[float]:
            Calculates the count rate and signal-to-noise ratio map for the sources.

        vignetting_factor(OptimalPointingIdx: int, vector_dictionary: Dict, simulation_data: Dict) -> List[float]:
            Computes the vignetting factor for each source based on optimal pointing data.

        write_fits_table(os_dictionary: Dict) -> None:
            Writes the nearby sources data to a FITS table and opens it using TOPCAT.

    """
    def __init__(self, catalog_name: Tuple[str, str], radius, simulation_data: Dict) -> None:
        table_1, table_2 = self.load_catalog(catalog_name=catalog_name, os_dictionary=simulation_data["os_dictionary"])
        
        # if "Chandra" in catalog_name:
        #     self.cone_search_catalog = self.load_cs_catalog(radius=radius, object_data=simulation_data["object_data"])

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
        cone = vo.dal.SCSService('http://cda.cfa.harvard.edu/csc2scs/coneSearch') 
        name = SkyCoord.from_name(object_data['object_name'])
        cone_search_catalog = cone.search(pos=name, radius=radius, verbosity=3)
        print(f"{colored('Cone search catalog are loaded !', 'green')}")
        return cone_search_catalog.to_table()

    
    def unique_sources_table(self, nearby_sources_table, column_name) -> Table:
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
        ra_value = list(self.nearby_sources_table["RA"])
        dec_value = list(self.nearby_sources_table["DEC"])
        
        return SkyCoord(ra=ra_value, dec=dec_value, unit=u.deg)
    
    
    def count_rate_SNR_map(self, simulation_data: Dict, radius: float) -> List[float]:
        telescop_data = simulation_data["telescop_data"]
        object_data = simulation_data["object_data"]
        active_workflow = simulation_data["os_dictionary"]["active_workflow"]
        excel_data_path = os.path.join(active_workflow, 'excel_data').replace("\\", "/")
        
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
            f.master_source_plot(master_sources=master_sources, object_data=object_data, number_graph=1)
        except Exception as error :
            print(f"{colored('An error occured : ', 'red')} {error}")
            
        path = os.path.join(output_name, "Master_source_cone.fits").replace("\\", "/")
        
        command = f"java -jar {topcat_software_path} {path}"
        subprocess.run(command)
        
        with fits.open(path, memmap=True) as data:
            master_source_cone = Table(data[1].data)
            
        return master_source_cone
    

    def cross_table_index(self) -> List:
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
