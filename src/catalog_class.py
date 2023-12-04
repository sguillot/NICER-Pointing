# --------------- Packages --------------- #

from astropy.table import Table
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

import function as f
import catalog_information as dict_cat
import sys
import numpy as np
import matplotlib.pyplot as plt
import pyvo as vo
import subprocess
import os

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
        object_position = dictionary['object_position']
        min_ra, max_ra = object_position.ra - pointing_area, object_position.ra + pointing_area
        min_dec, max_dec = object_position.dec - pointing_area, object_position.dec + pointing_area
        
        small_table = Table(names=self.xmm_catalog.colnames,
                            dtype=self.xmm_catalog.dtype)
        nearby_src_table = Table(names=self.xmm_catalog.colnames,
                                 dtype=self.xmm_catalog.dtype)
        
        for number in range(len(self.xmm_catalog)):
            if min_ra/u.deg < self.xmm_catalog['SC_RA'][number] < max_ra/u.deg and min_dec/u.deg < self.xmm_catalog['SC_DEC'][number] < max_dec/u.deg:
                small_table.add_row(self.xmm_catalog[number])
                
        if len(user_table) == 0:
            src_position = SkyCoord(ra=small_table['SC_RA'], dec=small_table['SC_DEC'], unit=u.deg)
            for number in tqdm(range(len(small_table))):
                if f.ang_separation(object_position, src_position[number]) < radius:
                    nearby_src_table.add_row(small_table[number])
            nearby_src_position = SkyCoord(ra=nearby_src_table['SC_RA'], dec=nearby_src_table['SC_DEC'], unit=u.deg)

        else:
            for number in range(len(user_table)):
                small_table.add_row(user_table[number])
                
            src_position = SkyCoord(ra=small_table['SC_RA'], dec=small_table['SC_DEC'], unit=u.deg)
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
        
        energy_band = np.array([0.35, 0.75, 1.5, 3.25, 8.25], dtype=float)
        flux_obs = np.array([self.nearby_sources_table[f"SC_EP_{item + 1}_FLUX"][number] for item in range(5)], dtype=float)
        flux_obs_err = np.array([self.nearby_sources_table[f"SC_EP_{item + 1}_FLUX_ERR"][number] for item in range(5)], dtype=float)
        
        popt, pcov = curve_fit(absorbed_power_law, energy_band, flux_obs, sigma=flux_obs_err)
        constant, absorb_pho_index = popt

        optimization_parameters = (energy_band, flux_obs, flux_obs_err, absorbed_power_law(energy_band, *popt))
        
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
        # number_interp = len(optimization_parameters)
        # number_column = 4
        # number_row = number_interp/number_column
        # if number_row < 1:
        #     number_row = 1
        # elif number_row %1 == 0:
        #     number_row= int(number_interp/4)
        # else:
        #     number_row = int(number_interp/4) + 1
        # index_figure, axes = plt.subplots(nrows=number_row, ncols=number_column, figsize=(17, 8), sharex=True)
        # index_figure.subplots_adjust(wspace=0.5, hspace=1.5)
        # index_figure.suptitle("Interpolation Photon Index", fontsize=20)
        # index_figure.text(0.5, 0.04, 'Energy [keV]', ha='center', va='center')
        # index_figure.text(0.04, 0.5, 'Flux [erg/cm^2/s]', ha='center', va='center', rotation='vertical')

        # count = 0
        # for row in range(number_row):
        #     for column in range(number_column):
        #         if count < number_interp:
        #             energy_band = optimization_parameters[count][0]
        #             flux_obs = optimization_parameters[count][1]
        #             flux_obs_err = optimization_parameters[count][2]
        #             absorbed_power_law = optimization_parameters[count][3]
        #             absorb_pho_index = photon_index[count]
        #             axes[row][column].errorbar(energy_band, flux_obs, flux_obs_err, fmt='*', color='red', ecolor='black')
        #             axes[row][column].plot(energy_band, absorbed_power_law, linestyle='dashdot', color="navy", label="Absorb")
        #             axes[row][column].set_title(f"Absorb $\Gamma$ = {absorb_pho_index:.8f}", fontsize=7)
        #             axes[row][column].legend(loc="upper left", ncol=2, fontsize=6)
        #         count += 1
                
        # plt.show()
        
        energy_band = dict_cat.dictionary_catalog["XMM"]["energy_band_center"]
        
        fig, axes = plt.subplots(1, 1, figsize=(15, 8))
        fig.suptitle("Interpolation Photon Index plot", fontsize=20)
        fig.text(0.5, 0.04, 'Energy [keV]', ha='center', va='center')
        fig.text(0.04, 0.5, 'Flux [erg/cm^2/s]', ha='center', va='center', rotation='vertical')
        
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
        result = ESASky.query_object_catalogs(position=name, catalogs="XMM-EPIC")
        xmm_epic = Table(result[0])
        xmm_obs_id = list(xmm_epic["observation_id"])
        result_fits_images = ESASky.get_images(observation_ids=xmm_obs_id[0], radius=radius, missions="XMM")
        
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
        self.corrected_mean_flux()
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
        radius = 5 * u.arcmin
        field_of_view = radius + 5*u.arcmin

        object_position = dictionary['object_position']

        min_ra, max_ra = object_position.ra - field_of_view, object_position.ra + field_of_view
        min_dec, max_dec = object_position.dec - field_of_view, object_position.dec + field_of_view

        small_table = Table(names= self.chandra_catalog.colnames,
                            dtype= self.chandra_catalog.dtype)

        nearby_sources_table = Table(names= self.chandra_catalog.colnames,
                                     dtype= self.chandra_catalog.dtype)

        for number in range(len(self.chandra_catalog)):
            if min_ra/u.deg < self.chandra_catalog["RA"][number] < max_ra/u.deg and min_dec/u.deg < self.chandra_catalog["DEC"][number] < max_dec/u.deg:
                small_table.add_row(self.chandra_catalog[number])
                
        src_position = SkyCoord(ra=small_table['RA'], dec=small_table['DEC'], unit=u.deg)

        for number in range(len(small_table)):
            if f.ang_separation(object_position, src_position[number]) < radius:
                nearby_sources_table.add_row(small_table[number])
                
        column_name = {"source_name": "Chandra_IAUNAME", 
                       "right_ascension": "RA",
                       "declination": "DEC",
                       "catalog_name": "Chandra"}

        unique_table = f.create_unique_sources_catalog(nearby_sources_table=nearby_sources_table, column_name=column_name)

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
            

    def visualization_interp(self, optimization_parameters, photon_index) -> None:
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
        number_interp = len(optimization_parameters)
        number_column = 4
        number_row = number_interp/number_column
        if number_row < 1 :
            number_row = 1
        elif number_row % 1 == 0:
            number_row = int(number_interp/number_column)
        else:
            number_row = int(number_interp/number_column) + 1 

        figure_interp, axes = plt.subplots(nrows=number_row, ncols=number_column, figsize=(15, 15), sharex=True)
        figure_interp.subplots_adjust(wspace=0.5, hspace=1.2)
        figure_interp.suptitle("Interpolation Photon Index", fontsize=20)
        figure_interp.text(0.5, 0.04, 'Energy [keV]', ha='center', va='center')
        figure_interp.text(0.04, 0.5, 'Flux [erg/cm^2/s]', ha='center', va='center', rotation='vertical')
        
        count = 0
        for row in range(number_row):
            for column in range(number_column):
                if count < number_interp:
                    energy_band = optimization_parameters[count][0]
                    flux_obs = optimization_parameters[count][1]
                    absorb_power_law = optimization_parameters[count][2]
                    absorb_pho_index = photon_index[count]

                    axes[row][column].errorbar(energy_band, flux_obs, fmt='*', color='red', ecolor='black')
                    axes[row][column].plot(energy_band, absorb_power_law, linestyle='dashdot', color="darkorange")
                    axes[row][column].set_title(f"Absorb $\Gamma$ = {absorb_pho_index}", fontsize=7)
                
                count += 1
                
        plt.show()


    def get_photon_index(self, key, table, index):
        
        if key == "Chandra":
            interp_data = {"band_flux_obs": dict_cat.dictionary_catalog[key]["band_flux_obs"],
                           "band_flux_obs_err": dict_cat.dictionary_catalog[key]["band_flux_obs_err"],
                           "energy_band_center": dict_cat.dictionary_catalog[key]["energy_band_center"]}
        
        if key == "CS_Chandra":
            interp_data = {"band_flux_obs": dict_cat.dictionary_catalog[key]["band_flux_obs"],
                           "band_flux_obs_err": dict_cat.dictionary_catalog[key]["band_flux_obs_err"],
                           "energy_band_center": dict_cat.dictionary_catalog[key]["energy_band_center"]}
            
        def model(energy_band, constant, gamma):
            sigma = np.array([1e-20, 1e-22, 1e-24], dtype=float)
            return (constant * energy_band **(-gamma)) * (np.exp(-sigma*3e20))
        
        flux_obs = [table[band_flux][index] for band_flux in interp_data["band_flux_obs"]]
        
        flux_err_obs = [[table[err_0][index] for err_0 in interp_data["band_flux_obs_err"][0]],
                        [table[err_1][index] for err_1 in interp_data["band_flux_obs_err"][1]]]
        err_neg = flux_err_obs[0]
        err_pos = flux_err_obs[1]
        
        mean_error = [np.mean([err_0, err_1]) for (err_0, err_1) in zip(err_neg, err_pos)]
        
        try:
            popt, pcov = curve_fit(model, interp_data["energy_band_center"], flux_obs, sigma=mean_error)
            constant, photon_index = popt
        except Exception as error:
            photon_index = 1.7
            
        params = (flux_obs, flux_err_obs, interp_data["energy_band_center"])
            
        return photon_index, params 
        

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


    def __init__(self, catalog_path: str, radius: Quantity, dictionary: dict, user_table: Table) -> None:
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
        self.nearby_sources_table, self.nearby_sources_position = self.find_nearby_sources(radius=radius, dictionary=dictionary)
        
        self.neighbourhood_of_object(radius=radius, dictionary=dictionary)
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
        
        
    def find_nearby_sources(self, radius: Quantity, dictionary: dict) -> Tuple[Table, SkyCoord]:
        """
        Finds sources in the Swift catalog that are within a certain radius from the target object.

        Parameters:
        - radius (float): Search radius (in degrees) for finding nearby sources.
        - dictionary (dict): A dictionary containing information about the target object.

        Returns:
        - Tuple[Table, SkyCoord]: A table of nearby sources and their sky coordinates.
        """
        field_of_view = radius + 5*u.arcmin

        object_position = dictionary['object_position']

        min_ra, max_ra = object_position.ra - field_of_view, object_position.ra + field_of_view
        min_dec, max_dec = object_position.dec - field_of_view, object_position.dec + field_of_view

        small_table = Table(names=self.swi_catalog.colnames,
                            dtype=self.swi_catalog.dtype)

        nearby_src_table = Table(names=self.swi_catalog.colnames,
                                 dtype=self.swi_catalog.dtype)

        for number in range(len(self.swi_catalog)):
            if min_ra/u.deg < self.swi_catalog["RA"][number] < max_ra/u.deg and min_dec/u.deg < self.swi_catalog["DEC"][number] < max_dec/u.deg:
                small_table.add_row(self.swi_catalog[number])
                
        src_position = SkyCoord(ra=small_table['RA'], dec=small_table['DEC'], unit=u.deg)
                
        for number in range(len(small_table)):
            if f.ang_separation(object_position, src_position[number]) < radius:
                nearby_src_table.add_row(small_table[number])
                
        column_names = ['Swift_IAUNAME', 'RA', 'DEC', 'Flux', 'Flux1', 'Flux2', 'Flux3']    
        nearby_src_table = f.sources_to_unique_sources(result_table=nearby_src_table, column_names=column_names)
                
        nearby_src_position = SkyCoord(ra=nearby_src_table['RA'], dec=nearby_src_table['DEC'], unit=u.deg)
                
        try :
            if len(nearby_src_table) != 0:
                print((f"We have detected {len(nearby_src_table)} sources close to {dictionary['object_name']}"))
                return nearby_src_table, nearby_src_position
            else:
                print(f"No sources detected close to {dictionary['object_name']}.")
                sys.exit()
        except Exception as error:
            print(f"An error occured : {error}")


    def neighbourhood_of_object(self, radius: Quantity, dictionary: dict) -> None:
        """
        Visualizes the neighborhood of the target object and nearby sources.

        Parameters:
        - radius (float): Search radius (in degrees) for finding nearby sources.
        - dictionary (dict): A dictionary containing information about the target object.
        """

        object_position = dictionary['object_position']
    
        swi_ra = self.nearby_sources_table['RA']
        swi_dec = self.nearby_sources_table['DEC']
        
        corrected_swi_ra = list(set(swi_ra))
        corrected_swi_dec = list(set(swi_dec))
        
        figure_1, axes = plt.subplots(1, 1, figsize=(12, 8))
        figure_1.suptitle(f"Neighbourhood of {dictionary['object_name']}, radius = {radius}", fontsize=20)
        
        axes.scatter(corrected_swi_ra, corrected_swi_dec, c='black', s=10, marker='*', label=f"Sources close to {dictionary['object_name']}, nbr_src : {len(corrected_swi_ra)}")
        axes.scatter(object_position.ra, object_position.dec, c='red', s=100, marker='+', label=f"{dictionary['object_name']}")
        axes.set_xlabel('Right Ascension')
        axes.set_ylabel('Declination')
        axes.legend(loc='upper right')
        
        plt.show()
    
    
    def dictionary_model(self):
        pass


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


    def __init__(self, catalog_path: str, radius: Quantity, dictionary: dict, user_table: Table) -> None:
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
        self.nearby_sources_table, self.nearby_sources_position = self.find_nearby_sources(radius=radius, dictionary=dictionary)
        
        self.neighbourhood_of_object(radius=radius, dictionary=dictionary)
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
        
        
    def find_nearby_sources(self, radius: Quantity, dictionary: dict) -> Tuple[Table, SkyCoord]:
        """
        Finds sources in the eRosita catalog that are within a certain radius from the target object.

        Parameters:
        - radius (float): Search radius (in degrees) for finding nearby sources.
        - dictionary (dict): A dictionary containing information about the target object.

        Returns:
        - Tuple[Table, SkyCoord]: A table of nearby sources and their sky coordinates.
        """
        field_of_view = radius + 5*u.arcmin

        object_position = dictionary['object_position']

        min_ra, max_ra = object_position.ra - field_of_view, object_position.ra + field_of_view
        min_dec, max_dec = object_position.dec - field_of_view, object_position.dec + field_of_view

        small_table = Table(names=self.eRo_catalog.colnames,
                            dtype=self.eRo_catalog.dtype)

        nearby_src_table = Table(names=self.eRo_catalog.colnames,
                                 dtype=self.eRo_catalog.dtype)

        for number in range(len(self.eRo_catalog)):
            if min_ra/u.deg < self.eRo_catalog["RA"][number] < max_ra/u.deg and min_dec/u.deg < self.eRo_catalog["DEC"][number] < max_dec/u.deg:
                small_table.add_row(self.eRo_catalog[number])
                
        src_position = SkyCoord(ra=small_table['RA'], dec=small_table['DEC'], unit=u.deg)
                
        for number in range(len(small_table)):
            if f.ang_separation(object_position, src_position[number]) < radius:
                nearby_src_table.add_row(small_table[number])
                
        nearby_src_position = SkyCoord(ra=nearby_src_table['RA'], dec=nearby_src_table['DEC'], unit=u.deg)
                
        try :
            if len(nearby_src_table) != 0:
                print((f"We have detected {len(nearby_src_table)} sources close to {dictionary['object_name']}"))
                return nearby_src_table, nearby_src_position
            else:
                print(f"No sources detected close to {dictionary['object_name']}.")
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


    def dictionary_model(self):
        pass


class CompareCatalog:
    """
    A class used to compare two astronomical catalogs for common and nearby sources.
    
    Attributes
    ----------
    catalog_1 : Table
        First catalog to be compared.
    catalog_2 : Table
        Second catalog to be compared.
    catalog_1_name : str
        Name of the first catalog.
    catalog_2_name : str
        Name of the second catalog.
    nearby_src_table_1 : Table
        Table of nearby sources from the first catalog.
    nearby_src_table_2 : Table
        Table of nearby sources from the second catalog.
    nearby_src_position_1 : SkyCoord
        Sky coordinates of the nearby sources from the first catalog.
    nearby_src_position_2 : SkyCoord
        Sky coordinates of the nearby sources from the second catalog.
    model_dictionary_1 : Dict[str, Dict[str, Union[str, float]]]
        Dictionary containing the model details for each source in the first catalog.
    model_dictionary_2 : Dict[str, Dict[str, Union[str, float]]]
        Dictionary containing the model details for each source in the second catalog.
    count_rate_1 : np.ndarray
        Array of count rates for each source in the first catalog.
    count_rate_2 : np.ndarray
        Array of count rates for each source in the second catalog.
    vector_dictionary : Dict
        (not described, as the method `opti_point_calcul` is not fully provided)
    OptimalPointingIdx_1 : (type not provided)
        Index of the optimal pointing for the first catalog.
    OptimalPointingIdx_2 : (type not provided)
        Index of the optimal pointing for the second catalog.
    
    Methods
    -------
    __init__(self, catalog_path, radius, dictionary, user_table) -> None:
        Initializes the CompareCatalog object by loading the catalogs, finding nearby sources, and performing various calculations.
    open_catalog(self, catalogs_path, radius, dictionary) -> Tuple[Table, Table, str, str]:
        Opens and processes the catalogs from the given paths.
    optimization(self, number, table) -> Tuple[float, float]:
        Performs an optimization calculation based on the provided number and table.
    add_photon_nh_and_gamma_xmm(self, table) -> Tuple[Table, Table]:
        Adds photon, NH, and gamma columns to the XMM catalog.
    add_pho_csc(self, table) -> Table:
        Adds photon information to the CSC catalog.
    var_function(self, dictionary) -> Tuple[Table, Table]:
        Applies a variable function based on the provided dictionary.
    find_nearby_sources(self, radius, dictionary) -> Tuple[Table, Table, SkyCoord, SkyCoord]:
        Finds sources in both catalogs that are within a certain radius.
    neighbourhood_of_object(self, radius, dictionary) -> None:
        Defines the neighbourhood of an object based on radius and dictionary.
    model_dictionary(self) -> Tuple[Dict[str, Dict[str, Union[str, float]]], Dict[str, Dict[str, Union[str, float]]]]:
        Generates dictionaries with model details for each source in both catalogs.
    count_rate(self) -> Tuple[np.ndarray, np.ndarray]:
        Calculates the count rate for each source in both catalogs.
    opti_point_calcul(self, simulation_data) -> None:
        Calculates the optimal pointing based on simulation data (method not fully provided).
    """

 
    def __init__(self, catalog_path: str, radius: Quantity, dictionary: dict, user_table: Table) -> None:
        self.catalog_1, self.catalog_2, self.catalog_1_name, self.catalog_2_name = self.open_catalog(catalogs_path=catalog_path, radius=radius, dictionary=dictionary)
        self.nearby_src_table_1, self.nearby_src_table_2, self.nearby_src_position_1, self.nearby_src_position_2 = self.find_nearby_sources(radius=radius, dictionary=dictionary)
        self.neighbourhood_of_object(radius=radius, dictionary=dictionary)
        
        self.model_dictionary_1, self.model_dictionary_2 = self.model_dictionary()
        self.count_rate_1, self.count_rate_2 = self.count_rate()

    
    def open_catalog(self, catalogs_path: str, radius: Quantity, dictionary: dict) -> Tuple[Table, Table, str, str]:
        if "catalog_data/Chandra.fits" not in catalogs_path:
            with fits.open(catalogs_path[0], memmap=True) as data1, fits.open(catalogs_path[1], memmap=True) as data2:
                result_1, result_2 = Table(data1[1].data), Table(data2[1].data)
                data1.close()
                data2.close()
                return result_1, result_2, catalogs_path[2], catalogs_path[3]
        else:
            index = catalogs_path.index("catalog_data/Chandra.fits")
            if index == 0 :
                with fits.open(catalogs_path[1], memmap=True) as data:
                    result_2 = Table(data[1].data)
                    data.close()
                cone = vo.dal.SCSService('http://cda.cfa.harvard.edu/csc2scs/coneSearch') 
                name = SkyCoord.from_name(dictionary['object_name'])
                self.cone_search_catalog = cone.search(pos=name, radius=radius, verbosity=3)
                return self.cone_search_catalog.to_table(), result_2, catalogs_path[2], catalogs_path[3]
            
            elif index == 1 :
                with fits.open(catalogs_path[0], memmap=True) as data:
                    result_1 = Table(data[1].data)
                    data.close()
                cone = vo.dal.SCSService('http://cda.cfa.harvard.edu/csc2scs/coneSearch') 
                name = SkyCoord.from_name(dictionary['object_name'])
                self.cone_search_catalog = cone.search(pos=name, radius=radius, verbosity=3)
                return result_1, self.cone_search_catalog.to_table(), catalogs_path[2], catalogs_path[3]

         
    def optimization(self, number: int, table: Table) -> Tuple[float, float]:
        
        def absorbed_power_law(x, constant, gamma):
            sigma = np.array([1e-20, 5e-21, 1e-22, 1e-23, 1e-24], dtype=float)
            return (constant * x ** (-gamma)) * (np.exp(-sigma * 3e20))
        
        energy_band = np.array([0.35, 0.75, 1.5, 3.25, 8.25], dtype=float)
        flux_obs = np.array([table[f"SC_EP_{item + 1}_FLUX"][number] for item in range(5)], dtype=float)
        flux_obs_err = np.array([table[f"SC_EP_{item + 1}_FLUX_ERR"][number] for item in range(5)], dtype=float)
            
        try:
            popt, pcov = curve_fit(absorbed_power_law, energy_band, flux_obs, sigma=flux_obs_err)
            constant, photon_index = popt
            return constant, photon_index
        except Exception as error:
            constant, photon_index = 2e-15, 1.0
            return constant, photon_index
         
         
    def add_photon_nh_and_gamma_xmm(self, table: Table) -> Tuple[Table, Table]:
        nbr_src = len(table)
        name_list = table['IAUNAME']
        xmm_dr11_table = Table(names=self.xmm_catalog.colnames,
                               dtype=self.xmm_catalog.dtype)
        index_in_xd11 = []
        for name in name_list:
            if name in self.xmm_catalog['IAUNAME']:
                index = list(self.xmm_catalog['IAUNAME']).index(name)
                index_in_xd11.append(index)
                xmm_dr11_table.add_row(self.xmm_catalog[index])
            else:
                print(f"{name} is missing")
        index_in_x2a = []
        message = "No data founded"
        for det_id in xmm_dr11_table["DETID"]:
            if det_id in self.x2a_catalog["DETID"]:
                index = list(self.x2a_catalog["DETID"]).index(det_id)
                index_in_x2a.append(index)
            else:
                index_in_x2a.append(message)
        print("Finish message put in list")
        col_name = ["Index in nearby_src_table", "Index in XmmDR11", "Index in Xmm2Athena"]
        data = [[n for n in range(nbr_src)], index_in_xd11, index_in_x2a]
        index_table = Table(data=data,
                            names=col_name)
        log_nh, col_photon_index = [], []
        for number in range(nbr_src):
            if index_table["Index in Xmm2Athena"][number] != message:
                log_nh.append(self.x2a_catalog["logNH_med"][number])
                col_photon_index.append(self.x2a_catalog["PhoIndex_med"][number])
            else:
                log_nh.append(0.0)
                constant, photon_index = self.optimization(number, table)
                col_photon_index.append(photon_index)
        print("Finish optimization")
        col_nh = [np.exp(value * np.log(10)) if value != 0.0 else 3e20 for value in log_nh]
        col_names = ["PhoIndex", "Nh"]
        col_data = [col_photon_index, col_nh]
        for name, data in zip(col_names, col_data):
            table[name] = data
            
        return table, index_table   
         
           
    def add_pho_csc(self, table: Table) -> Table:
        
        def get_gamma (index, table):
            
            flux = np.array([], dtype=float)
            for value in ['s', 'm', 'h']:
                flux = np.append(flux, table[f"flux_aper_{value}"][index])
            
            energy_range = np.array([0.85, 1.6, 4.5], dtype=float)
            
            def power_law(x, constant, gamma):
                return constant * (x ** (-gamma))
            
            try:
                popt, pcov = curve_fit(power_law, energy_range, flux)
                constant, photon_index = popt
                return photon_index
            except Exception:
                constant, photon_index = 2e-15, 1.0
                return photon_index
        
        pho_index = np.array([], dtype=float)
        
        for index, pho in enumerate(table["powlaw_gamma"]):
            if not isinstance(pho, np.ma.core.MaskedConstant):
                pho_index= np.append(pho_index, pho)
            elif isinstance(pho, np.ma.core.MaskedConstant):
                pho_index = np.append(pho_index, get_gamma(index, table))

        table['PhoIndex_csc'] = pho_index
        
        return table
    
    
    def var_function(self, dictionary: dict) -> Tuple[Table, Table]:
        if self.catalog_1_name == "CSC_2.0" or self.catalog_2_name == "Xmm_DR13" and self.catalog_1_name =="Xmm_DR13" or self.catalog_2_name =="CSC_2.0":
            x2a_path = "catalog_data/xmm2athena_D6.1_V3.fits"
            xmm_path = "catalog_data/4XMM_DR11cat_v1.0.fits"
            with fits.open(x2a_path) as data_x2a, fits.open(xmm_path) as data_xmm:
                self.x2a_catalog = Table(data_x2a[1].data)
                self.xmm_catalog = Table(data_xmm[1].data)
                data_xmm.close()
                data_x2a.close()

        
        if self.catalog_1_name == "CSC_2.0" and self.catalog_2_name == "Xmm_DR13":
            
            # -------------------- Cone search -------------------- #
            
            inter, intra = self.catalog_1['var_inter_prob_b'], self.catalog_1['var_intra_prob_b']
            var_column_cs = np.array([])
            
            self.catalog_1 = self.add_pho_csc(self.catalog_1)
            
            for inter_value, intra_value in zip(inter, intra):
                if isinstance(inter_value, np.ma.core.MaskedConstant) and isinstance(intra_value, np.ma.core.MaskedConstant):
                    var_column_cs = np.append(var_column_cs, 0.0)
                elif isinstance(inter_value, np.ma.core.MaskedConstant) or isinstance(intra_value, np.ma.core.MaskedConstant):
                    if isinstance(inter_value, np.ma.core.MaskedConstant):
                        var_column_cs = np.append(var_column_cs, intra_value)
                    else:
                        var_column_cs = np.append(var_column_cs, inter_value)
                else:
                    mean_value = np.mean([inter_value, intra_value])
                    var_column_cs = np.append(var_column_cs, mean_value)  
            self.catalog_1['Variability'] = var_column_cs
            
            mean_value, new_flux_aper_b = [], []
            for flux in self.catalog_1["flux_aper_b"]:
                if not isinstance(flux, np.ma.core.MaskedConstant) and flux < 2e-15:
                    mean_value.append(flux)
            mean_flux = np.mean(mean_value)
            
            for flux in self.catalog_1['flux_aper_b']:
                if not isinstance(flux, np.ma.core.MaskedConstant):
                    new_flux_aper_b.append(flux)
                else:
                    new_flux_aper_b.append(mean_flux)
            self.catalog_1['flux_aper_b'] = new_flux_aper_b
            
            # -------------------- Xmm_catalog -------------------- #

            self.nearby_src_table_2, index_table = self.add_photon_nh_and_gamma_xmm(self.nearby_src_table_2)
            
            nbr_src = len(self.nearby_src_table_2)
            message = "No data founded"
            name = dictionary["object_name"]

            index_array, iauname_array, sc_ra_array = np.array([], dtype=int), np.array([], dtype=str), np.array([], dtype=float)
            sc_dec_array, sc_fvar_array, in_x2a_array = np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=bool)

            for number in range(nbr_src):
                if not np.isnan(self.xmm_catalog["SC_FVAR"][index_table["Index in XmmDR11"][number]]):

                    index_array = np.append(index_array, index_table["Index in nearby_src_table"][number])
                    iauname_array = np.append(iauname_array, self.nearby_src_table_2["IAUNAME"][number])
                    sc_ra_array = np.append(sc_ra_array, self.nearby_src_table_2["SC_RA"][number])
                    sc_dec_array = np.append(sc_dec_array, self.nearby_src_table_2["SC_DEC"][number])
                    sc_fvar_array = np.append(sc_fvar_array, self.nearby_src_table_2["SC_FVAR"][number])

                    if index_table["Index in Xmm2Athena"][number] != message:
                        in_x2a_array = np.append(in_x2a_array, True)
                    else:
                        in_x2a_array = np.append(in_x2a_array, False)

            column_names = ["INDEX", "IAUNAME", "SC_RA", "SC_DEC", "SC_FVAR", "IN_X2A"]
            data_array = [index_array, iauname_array, sc_ra_array, sc_dec_array, sc_fvar_array, in_x2a_array]
            self.variability_table = Table()

            for data, name in zip(data_array, column_names):
                self.variability_table[name] = data

            message_xmm = f"Among {len(self.nearby_src_table_2)} sources detected close to {name}, {len(index_array)} of them are variable. Using DR13 Catalog."
            print(message_xmm)
            message_xmm2ath = f"Among {len(index_array)} variable sources, {list(self.variability_table['IN_X2A']).count(True)} are in Xmm2Athena and {list(self.variability_table['IN_X2A']).count(False)} are not in Xmm2Athena. "    
            print(message_xmm2ath)

            return self.catalog_1, self.nearby_src_table_2
                    
        elif self.catalog_1_name == "Xmm_DR13" and self.catalog_2_name == "CSC_2.0":
            
            # -------------------- Cone search -------------------- #
            
            inter, intra = self.catalog_2['var_inter_prob_b'], self.catalog_2['var_intra_prob_b']
            var_column_cs = np.array([])
            
            self.catalog_2 = self.add_pho_csc(self.catalog_2)
            
            for inter_value, intra_value in zip(inter, intra):
                if isinstance(inter_value, np.ma.core.MaskedConstant) and isinstance(intra_value, np.ma.core.MaskedConstant):
                    var_column_cs = np.append(var_column_cs, 0.0)
                elif isinstance(inter_value, np.ma.core.MaskedConstant) or isinstance(intra_value, np.ma.core.MaskedConstant):
                    if isinstance(inter_value, np.ma.core.MaskedConstant):
                        var_column_cs = np.append(var_column_cs, intra_value)
                    else:
                        var_column_cs = np.append(var_column_cs, inter_value)
                else:
                    mean_value = np.mean([inter_value, intra_value])
                    var_column_cs = np.append(var_column_cs, mean_value)
            self.catalog_2['Variability'] = var_column_cs
            
            mean_value, new_flux_aper_b = [], []
            for flux in self.catalog_2["flux_aper_b"]:
                if not isinstance(flux, np.ma.core.MaskedConstant) and flux < 2e-15:
                    mean_value.append(flux)
            mean_flux = np.mean(mean_value)    
        
            for flux in self.catalog_2['flux_aper_b']:
                if not isinstance(flux, np.ma.core.MaskedConstant):
                    new_flux_aper_b.append(flux)
                else:
                    new_flux_aper_b.append(mean_flux)
            self.catalog_2['flux_aper_b'] = new_flux_aper_b
            
            # -------------------- Xmm_catalog -------------------- #
                    
            self.nearby_src_table_1, index_table = self.add_photon_nh_and_gamma_xmm(self.nearby_src_table_1)
            
            nbr_src = len(self.nearby_src_table_1)
            message = "No data founded"
            name = dictionary["object_name"]

            index_array, iauname_array, sc_ra_array = np.array([], dtype=int), np.array([], dtype=str), np.array([], dtype=float)
            sc_dec_array, sc_fvar_array, in_x2a_array = np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=bool)

            for number in range(nbr_src):
                if not np.isnan(self.xmm_catalog["SC_FVAR"][index_table["Index in XmmDR11"][number]]):

                    index_array = np.append(index_array, index_table["Index in nearby_src_table"][number])
                    iauname_array = np.append(iauname_array, self.nearby_src_table_1["IAUNAME"][number])
                    sc_ra_array = np.append(sc_ra_array, self.nearby_src_table_1["SC_RA"][number])
                    sc_dec_array = np.append(sc_dec_array, self.nearby_src_table_1["SC_DEC"][number])
                    sc_fvar_array = np.append(sc_fvar_array, self.nearby_src_table_1["SC_FVAR"][number])

                    if index_table["Index in Xmm2Athena"][number] != message:
                        in_x2a_array = np.append(in_x2a_array, True)
                    else:
                        in_x2a_array = np.append(in_x2a_array, False)

            column_names = ["INDEX", "IAUNAME", "SC_RA", "SC_DEC", "SC_FVAR", "IN_X2A"]
            data_array = [index_array, iauname_array, sc_ra_array, sc_dec_array, sc_fvar_array, in_x2a_array]
            self.variability_table = Table()

            for data, name in zip(data_array, column_names):
                self.variability_table[name] = data

            message_xmm = f"Among {len(self.nearby_src_table_1)} sources detected close to {name}, {len(index_array)} of them are variable. Using DR13 Catalog."
            print(message_xmm)
            message_xmm2ath = f"Among {len(index_array)} variable sources, {list(self.variability_table['IN_X2A']).count(True)} are in Xmm2Athena and {list(self.variability_table['IN_X2A']).count(False)} are not in Xmm2Athena. "    
            print(message_xmm2ath)        
            
            return self.nearby_src_table_1, self.catalog_2
            
            
    def find_nearby_sources(self, radius: Quantity, dictionary: dict) -> Tuple[Table, Table, SkyCoord, SkyCoord]:
        
        field_of_view = radius + 5*u.arcmin
        object_position = dictionary['object_position']
        min_ra, max_ra = object_position.ra - field_of_view, object_position.ra + field_of_view
        min_dec, max_dec = object_position.dec - field_of_view, object_position.dec + field_of_view
        
        if self.catalog_1_name == "Xmm_DR13" and self.catalog_2_name =="CSC_2.0":
            ra_1, dec_1 = "SC_RA", "SC_DEC"
            ra_2, dec_2 = 'ra', 'dec'
            
            nbr_src_1 = len(self.catalog_1)
            
            table_1 = Table(names=self.catalog_1.colnames,
                            dtype=self.catalog_1.dtype)
            
            self.nearby_src_table_1 = Table(names=self.catalog_1.colnames,
                                            dtype=self.catalog_1.dtype)
            
            for n_1 in range(nbr_src_1):
                if min_ra/u.deg < self.catalog_1[ra_1][n_1] < max_ra/u.deg and min_dec/u.deg < self.catalog_1[dec_1][n_1] < max_dec/u.deg:
                    table_1.add_row(self.catalog_1[n_1])
            
            nbr_nearby_src_1 = len(table_1)
            src_position = SkyCoord(ra=table_1[ra_1], dec=table_1[dec_1], unit=u.deg)
            for n_1 in range(nbr_nearby_src_1):
                if f.ang_separation(object_position, src_position[n_1]) < radius:
                    self.nearby_src_table_1.add_row(table_1[n_1])
                    
            self.nearby_src_position_1 = SkyCoord(ra=list(self.nearby_src_table_1[ra_1]), dec=list(self.nearby_src_table_1[dec_1]), unit=u.deg)
            self.nearby_src_position_2 = SkyCoord(ra=list(self.catalog_2[ra_2]), dec=list(self.catalog_2[dec_2]), unit=u.deg)
            self.nearby_src_table_1, self.catalog_2 = self.var_function(dictionary=dictionary)
            
            try:
                if len(self.nearby_src_table_1) != 0 or len(self.catalog_2) != 0:
                    print(f"We have detected {len(self.nearby_src_table_1)} sources in {self.catalog_1_name} and {len(self.catalog_2)} sources in {self.catalog_2_name} close to {dictionary['object_name']}")
                    return self.nearby_src_table_1, self.catalog_2, self.nearby_src_position_1, self.nearby_src_position_2
                else:
                    print(f"We can't compare catalogs because no sources was detected in one of the catalog")
                    sys.exit()
            except Exception as error:
                print(f"An error occured : {error}")
        
        elif self.catalog_1_name == "CSC_2.0" and self.catalog_2_name =="Xmm_DR13":
            ra_1, dec_1 = "ra", "dec"
            ra_2, dec_2 = "SC_RA", "SC_DEC"
            
            nbr_src_2 = len(self.catalog_2)
            
            table_2 = Table(names=self.catalog_2.colnames,
                            dtype=self.catalog_2.dtype)
            
            self.nearby_src_table_2 = Table(names=self.catalog_2.colnames,
                                            dtype=self.catalog_2.dtype)
            
            for n_2 in range(nbr_src_2):
                if min_ra/u.deg < self.catalog_2[ra_2][n_2] < max_ra/u.deg and min_dec/u.deg < self.catalog_2[dec_2][n_2] < max_dec/u.deg:
                    table_2.add_row(self.catalog_2[n_2])
            
            nbr_nearby_src_2 = len(table_2)
            src_position = SkyCoord(ra=table_2[ra_2], dec=table_2[dec_2], unit=u.deg)
            for n_2 in range(nbr_nearby_src_2):
                if f.ang_separation(object_position, src_position[n_2]) < radius:
                    self.nearby_src_table_2.add_row(table_2[n_2])
                    
            self.nearby_src_position_1 = SkyCoord(ra=list(self.catalog_1[ra_1]), dec=list(self.catalog_1[dec_1]), unit=u.deg)
            self.nearby_src_position_2 = SkyCoord(ra=list(self.nearby_src_table_2[ra_2]), dec=list(self.nearby_src_table_2[dec_2]), unit=u.deg)
            self.catalog_1, self.nearby_src_table_2 = self.var_function(dictionary=dictionary)
            
            try:
                if len(self.catalog_2) != 0 or len(self.nearby_src_table_2) != 0:
                    print(f"We have detected {len(self.catalog_1)} sources in {self.catalog_1_name} and {len(self.nearby_src_table_2)} sources in {self.catalog_2_name} close to {dictionary['object_name']}")
                    return self.catalog_1, self.nearby_src_table_2, self.nearby_src_position_1, self.nearby_src_position_2
                else:
                    print(f"We can't compare catalogs because no sources was detected in one of the catalog")
                    sys.exit()
            except Exception as error:
                print(f"An error occured : {error}")


    def neighbourhood_of_object(self, radius: Quantity, dictionary: dict) -> None:
        
        name = dictionary["object_name"]
        obj_ra, obj_dec = dictionary["object_position"].ra, dictionary["object_position"].dec
        ra_in_x2a = [ra for index, ra in enumerate(self.variability_table['SC_RA']) if self.variability_table['IN_X2A'][index] == True]
        dec_in_x2a = [dec for index, dec in enumerate(self.variability_table['SC_DEC']) if self.variability_table['IN_X2A'][index] == True]
        ra_in_dr11 = [ra for index, ra in enumerate(self.variability_table['SC_RA']) if self.variability_table['IN_X2A'][index] == False]
        dec_in_dr11 = [dec for index, dec in enumerate(self.variability_table['SC_DEC']) if self.variability_table['IN_X2A'][index] == False]
        
        figure = plt.figure(figsize=(17, 9))
        figure.suptitle(f"Neighbourhood of {name}", fontsize=20)

        result = ESASky.query_object_catalogs(position=name, catalogs="XMM-EPIC")
        xmm_epic = Table(result[0])
        xmm_obs_id = list(xmm_epic["observation_id"])
        result_fits_images = ESASky.get_images(observation_ids=xmm_obs_id[0], radius=radius, missions=["XMM", "Chandra"])
        
        if len(result_fits_images) != 0:
            image = result_fits_images["XMM"][0][0].data[0, :, :]
            wcs = WCS(result_fits_images["XMM"][0][0].header)
            _wcs_ = wcs.dropaxis(2)
            norm = ImageNormalize(image, interval=PercentileInterval(98.0), stretch=LinearStretch())
            
            if self.catalog_1_name == "Xmm_DR13" and self.catalog_2_name == "CSC_2.0":
                title_1, title_2 = "Xmm_DR13", "CSC_2.0"
                axes_0 = figure.add_subplot(221, projection=_wcs_)
                axes_1 = figure.add_subplot(223, projection=_wcs_, sharex=axes_0, sharey=axes_0)
                axes_2 = figure.add_subplot(222)
                axes_3 = figure.add_subplot(224)
                axes_2.tick_params(axis='x', which='both', labelbottom=False)
                axes_0.set_title(f"{title_1}")
                axes_2.set_title(f"{title_2}")
                
                xmm_ra = list(self.nearby_src_table_1['SC_RA'])
                xmm_dec = list(self.nearby_src_table_1['SC_DEC'])
                csc_ra = list(self.nearby_src_table_2["ra"])
                csc_dec = list(self.nearby_src_table_2["dec"])
                cs_ra_var = [ra for index, ra in enumerate(list(self.nearby_src_table_2['ra'])) if self.nearby_src_table_2['Variability'][index] != 0.0]
                cs_ra_invar = [ra for index, ra in enumerate(list(self.nearby_src_table_2['ra'])) if self.nearby_src_table_2['Variability'][index] == 0.0]
                cs_dec_var = [dec for index, dec in enumerate(list(self.nearby_src_table_2['dec'])) if self.nearby_src_table_2['Variability'][index] != 0.0]
                cs_dec_invar = [dec for index, dec in enumerate(list(self.nearby_src_table_2['dec'])) if self.nearby_src_table_2['Variability'][index] == 0.0]
                
            elif self.catalog_1_name == "CSC_2.0" and self.catalog_2_name == "Xmm_DR13":
                title_1, title_2 = "CSC_2.0", "Xmm_DR13"
                axes_0 = figure.add_subplot(222, projection=_wcs_)
                axes_1 = figure.add_subplot(224, projection=_wcs_, sharex=axes_0, sharey=axes_0)
                axes_2 = figure.add_subplot(221)
                axes_3 = figure.add_subplot(223)
                axes_2.tick_params(axis='x', which='both', labelbottom=False)
                axes_0.set_title(f"{title_2}")
                axes_2.set_title(f"{title_1}")

                xmm_ra = list(self.nearby_src_table_2['SC_RA'])
                xmm_dec = list(self.nearby_src_table_2['SC_DEC'])
                csc_ra = list(self.nearby_src_table_1["ra"])
                csc_dec = list(self.nearby_src_table_1["dec"])
                cs_ra_var = [ra for index, ra in enumerate(list(self.nearby_src_table_1['ra'])) if self.nearby_src_table_1['Variability'][index] != 0.0]
                cs_ra_invar = [ra for index, ra in enumerate(list(self.nearby_src_table_1['ra'])) if self.nearby_src_table_1['Variability'][index] == 0.0]
                cs_dec_var = [dec for index, dec in enumerate(list(self.nearby_src_table_1['dec'])) if self.nearby_src_table_1['Variability'][index] != 0.0]
                cs_dec_invar = [dec for index, dec in enumerate(list(self.nearby_src_table_1['dec'])) if self.nearby_src_table_1['Variability'][index] == 0.0]

            axes_0.coords[0].set_format_unit(u.hourangle)
            axes_0.coords[1].set_format_unit(u.deg) 
            axes_0.imshow(image, cmap='gray', origin='lower', norm=norm, interpolation='nearest', aspect='equal')
            axes_0.scatter(xmm_ra, xmm_dec, s=30, transform=axes_0.get_transform('fk5'), facecolors='none', edgecolors='orange', label=f"Sources : {len(xmm_ra)}")
            axes_0.scatter(obj_ra, obj_dec, s=100, marker="*", transform=axes_0.get_transform('fk5'), facecolors='none', edgecolors='red', label=f"{name}")
            axes_0.set_xlabel(" ")
            axes_0.set_ylabel(" ")
            # xlim, ylim = plt.xlim(), plt.ylim()
            # value_x, value_y = 180, 180
            # axes_0.set_xlim(xmin=xlim[0]+value_x, xmax=xlim[1]-value_x)
            # axes_0.set_ylim(ymin=ylim[0]+value_y, ymax=ylim[1]-value_y)

            axes_1.imshow(image, cmap='gray', origin='lower', norm=norm, interpolation='nearest', aspect='equal')
            axes_1.scatter(xmm_ra, xmm_dec, s=30, transform=axes_1.get_transform('fk5'), facecolors='none', edgecolors='orange', label=f"Sources : {len(xmm_ra)}")
            axes_1.scatter(ra_in_dr11, dec_in_dr11, s=30, transform=axes_1.get_transform('fk5'), facecolors='none', edgecolors='blue', label=f"Var src not in x2a : {len(ra_in_dr11)} sources")
            axes_1.scatter(ra_in_x2a, dec_in_x2a, s=30, transform=axes_1.get_transform('fk5'), facecolors='none', edgecolors='hotpink', label=f"Var src in x2a : {len(ra_in_x2a)} sources")
            axes_1.scatter(obj_ra, obj_dec, s=100, marker="*", transform=axes_1.get_transform('fk5'), facecolors='none', edgecolors='red', label=f"{name}")
            axes_1.set_xlabel(" ")
            axes_1.set_ylabel(" ")
            
        elif len(result_fits_images) == 0:
            figure.text(0.5, 0.04, 'Right Ascension [deg]', ha='center', va='center', fontsize=16)
            figure.text(0.04, 0.5, 'Declination [deg]', ha='center', va='center', rotation='vertical', fontsize=16)
            
            if self.catalog_1_name == "Xmm_DR13" and self.catalog_2_name == "CSC_2.0":
                title_1, title_2 = "Xmm_DR13", "CSC_2.0"
                axes_0 = figure.add_subplot(221)
                axes_1 = figure.add_subplot(223)
                axes_2 = figure.add_subplot(222)
                axes_3 = figure.add_subplot(224)
                axes_0.tick_params(axis='x', which='both', labelbottom=False)
                axes_2.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
                axes_3.tick_params(axis='y', which='both', labelleft=False)
                axes_0.set_title(f"{title_1}")
                axes_2.set_title(f"{title_2}")
                
                xmm_ra = list(self.nearby_src_table_1['SC_RA'])
                xmm_dec = list(self.nearby_src_table_1['SC_DEC'])
                csc_ra = list(self.nearby_src_table_2["ra"])
                csc_dec = list(self.nearby_src_table_2["dec"])
                cs_ra_var = [ra for index, ra in enumerate(list(self.nearby_src_table_2['ra'])) if self.nearby_src_table_2['Variability'][index] != 0.0]
                cs_ra_invar = [ra for index, ra in enumerate(list(self.nearby_src_table_2['ra'])) if self.nearby_src_table_2['Variability'][index] == 0.0]
                cs_dec_var = [dec for index, dec in enumerate(list(self.nearby_src_table_2['dec'])) if self.nearby_src_table_2['Variability'][index] != 0.0]
                cs_dec_invar = [dec for index, dec in enumerate(list(self.nearby_src_table_2['dec'])) if self.nearby_src_table_2['Variability'][index] == 0.0]
                
            elif self.catalog_1_name == "CSC_2.0" and self.catalog_2_name == "Xmm_DR13":
                title_1, title_2 = "CSC_2.0", "Xmm_DR13"
                axes_0 = figure.add_subplot(222)
                axes_1 = figure.add_subplot(224)
                axes_2 = figure.add_subplot(221)
                axes_3 = figure.add_subplot(223)
                axes_0.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
                axes_2.tick_params(axis='x', which='both', labelbottom=False)
                axes_1.tick_params(axis='y', which='both', labelleft=False)
                axes_0.set_title(f"{title_2}")
                axes_2.set_title(f"{title_1}")
                
                xmm_ra = list(self.nearby_src_table_2['SC_RA'])
                xmm_dec = list(self.nearby_src_table_2['SC_DEC'])
                csc_ra = list(self.nearby_src_table_1["ra"])
                csc_dec = list(self.nearby_src_table_1["dec"])
                cs_ra_var = [ra for index, ra in enumerate(list(self.nearby_src_table_1['ra'])) if self.nearby_src_table_1['Variability'][index] != 0.0]
                cs_ra_invar = [ra for index, ra in enumerate(list(self.nearby_src_table_1['ra'])) if self.nearby_src_table_1['Variability'][index] == 0.0]
                cs_dec_var = [dec for index, dec in enumerate(list(self.nearby_src_table_1['dec'])) if self.nearby_src_table_1['Variability'][index] != 0.0]
                cs_dec_invar = [dec for index, dec in enumerate(list(self.nearby_src_table_1['dec'])) if self.nearby_src_table_1['Variability'][index] == 0.0]
                
            axes_0.scatter(xmm_ra, xmm_dec, s=30, facecolors='none', edgecolors='black', label=f"Sources : {len(xmm_ra)}")
            axes_0.scatter(obj_ra, obj_dec, s=100, marker="*", facecolors='none', edgecolors='red', label=f"{name}")
            
            axes_1.scatter(xmm_ra, xmm_dec, s=30, facecolors='none', edgecolors='orange', label=f"Sources : {len(xmm_ra)}")
            axes_1.scatter(ra_in_dr11, dec_in_dr11, s=30, facecolors='none', edgecolors='blue', label=f"Var src not in x2a : {len(ra_in_dr11)} sources")
            axes_1.scatter(ra_in_x2a, dec_in_x2a, s=30, facecolors='none', edgecolors='hotpink', label=f"Var src in x2a : {len(ra_in_x2a)} sources")
            axes_1.scatter(obj_ra, obj_dec, s=100, marker="*", facecolors='none', edgecolors='red', label=f"{name}")
        
        axes_0.legend(loc="upper right", fontsize=8)
        
        axes_1.legend(loc='upper right', ncol=2, fontsize=8)
        
        axes_2.scatter(csc_ra, csc_dec, facecolors='none', edgecolors='black', s=30, label=f"Nearby sources : {len(csc_ra)}")
        axes_2.scatter(dictionary["object_position"].ra, dictionary["object_position"].dec, s=100, marker="*", facecolors='none', edgecolors='red', label=f"{dictionary['object_name']}")
        axes_2.legend(loc="upper right", fontsize=8)
        
        axes_3.scatter(cs_ra_var, cs_dec_var, s=30, facecolors='none', edgecolors='darkorange', label=f"Var src : {len(cs_ra_var)} sources")
        axes_3.scatter(cs_ra_invar, cs_dec_invar, s=30, facecolors='none', edgecolors='blue', label=f"Invar src : {len(cs_ra_invar)} sources")
        axes_3.scatter(obj_ra, obj_dec, marker='*', s=100, facecolors='none', edgecolors='red', label=f"{name}")
        axes_3.legend(loc="upper right", ncol=2, fontsize=8)

        plt.show()
        
        
    def model_dictionary(self) -> Tuple[Dict[str, Dict[str, Union[str, float]]], Dict[str, Dict[str, Union[str, float]]]]:
        
        model_dictionary_1 = {}
        model_dictionary_2 = {}
        
        if self.catalog_1_name == "Xmm_DR13" and self.catalog_2_name == "CSC_2.0":
            flux_value_1, flux_value_2 = "SC_EP_8_FLUX", "flux_aper_b"
            m_1, m_2 = "PhoIndex", "PhoIndex_csc"
            nh_value_1, nh_value_2 = "Nh", "nh_gal"
        elif self.catalog_1_name == "CSC_2.0" and self.catalog_2_name == "Xmm_DR13":
            flux_value_1, flux_value_2 = "flux_aper_b", "SC_EP_8_FLUX"
            m_1, m_2 = "PhoIndex_csc", "PhoIndex"
            nh_value_1, nh_value_2 = "nh_gal", "Nh"
        
        nbr_src_1, nbr_src_2 = len(self.nearby_src_table_1), len(self.nearby_src_table_2)
        self.model_1, self.model_2 = {}, {}
        
        model_1, model_2 = np.array(['power' for item in range(nbr_src_1)], dtype=str), np.array(['power' for item in range(nbr_src_2)], dtype=str)
        
        model_value_1 = np.array([self.nearby_src_table_1[m_1][n_1] for n_1 in range(nbr_src_1)], dtype=float)
        model_value_2 = np.array([self.nearby_src_table_2[m_2][n_2]for n_2 in range(nbr_src_2)], dtype=float)
        
        flux_1 = np.array([self.nearby_src_table_1[flux_value_1][n_1] for n_1 in range(nbr_src_1)], dtype=float)
        flux_2 = np.array([self.nearby_src_table_2[flux_value_2][n_2] for n_2 in range(nbr_src_2)], dtype=float)
        
        nh_1 = np.array([self.nearby_src_table_1[nh_value_1][n_1] for n_1 in range(nbr_src_1)], dtype=float)
        nh_2 = np.array([self.nearby_src_table_2[nh_value_2][n_2] for n_2 in range(nbr_src_2)], dtype=float)

        for item in range(nbr_src_1):

            dictionary = {
                "model": model_1[item],
                "model_value": model_value_1[item],
                "flux": flux_1[item],
                "column_dentsity": nh_1[item]
            }

            model_dictionary_1[f"src_{item}"] = dictionary
            
        for item in range(nbr_src_2):

            dictionary = {
                "model": model_2[item],
                "model_value": model_value_2[item],
                "flux": flux_2[item],
                "column_dentsity": nh_2[item]
            }

            model_dictionary_2[f"src_{item}"] = dictionary
            
        return model_dictionary_1, model_dictionary_2


    def count_rate(self) -> Tuple[np.ndarray, np.ndarray]:
        nbr_src_1 = len(self.nearby_src_table_1)
        nbr_src_2 = len(self.nearby_src_table_2)
        
        ct_rates_1, ct_rates_2 = np.array([], dtype=float), np.array([], dtype=float)

        for item in range(nbr_src_1):
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
            
        for item in range(nbr_src_2):
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
        
        return ct_rates_1, ct_rates_2
    
    
    def opti_point_calcul(self, simulation_data: dict) -> None:
        object_data = simulation_data["object_data"]
        telescop_data = simulation_data["telescop_data"]
            
        Delta_RA, Delta_DEC = Angle(np.arange(-3.0, 3.1, 0.1), unit=u.deg)/60, Angle(np.arange(-3.0, 3.1, 0.1), unit=u.deg)/60
        
        Sample_RA, Sample_DEC, PSRrates =  np.zeros((3, len(Delta_RA) * len(Delta_DEC)))
        
        SNR_1, SRCrates_1 = np.zeros((2, len(Delta_RA) * len(Delta_DEC)))
        SNR_2, SRCrates_2 = np.zeros((2, len(Delta_RA) * len(Delta_DEC)))
        
        PSRcountrates = object_data['count_rate']
        
        count = 0
        for i in Delta_RA:
            for j in Delta_DEC:
                NICERpointing = SkyCoord(ra=object_data["object_position"].ra + i, dec=object_data["object_position"].dec + j)
                PSRseparation = f.ang_separation(object_data["object_position"], NICERpointing)
                
                SRCseparation_1 = f.ang_separation(self.nearby_src_position_1, NICERpointing)
                SRCseparation_2 = f.ang_separation(self.nearby_src_position_2, NICERpointing)
                
                PSRcountrateScaled = f.scaled_ct_rate(PSRseparation.arcmin, PSRcountrates, telescop_data["EffArea"], telescop_data["OffAxisAngle"])
                
                SRCcountrateScaled_1 = f.scaled_ct_rate(SRCseparation_1.arcmin, self.count_rate_1, telescop_data["EffArea"], telescop_data["OffAxisAngle"])
                SRCcountrateScaled_2 = f.scaled_ct_rate(SRCseparation_2.arcmin, self.count_rate_2, telescop_data["EffArea"], telescop_data["OffAxisAngle"])
                
                Sample_RA[count] = NICERpointing.ra.deg
                Sample_DEC[count] = NICERpointing.dec.deg

                PSRrates[count] = PSRcountrateScaled
                
                SRCrates_1[count] = np.sum(SRCcountrateScaled_1)
                SRCrates_2[count] = np.sum(SRCcountrateScaled_2)

                SNR_1[count] = f.signal_to_noise(PSRcountrateScaled, SRCcountrateScaled_1, simulation_data["INSTbkgd"], simulation_data["EXPtime"])
                SNR_2[count] = f.signal_to_noise(PSRcountrateScaled, SRCcountrateScaled_2, simulation_data["INSTbkgd"], simulation_data["EXPtime"])
                count +=  1
                
        self.OptimalPointingIdx_1 = np.where(SNR_1==max(SNR_1))[0][0]
        SRCoptimalSEPAR_1 = f.ang_separation(self.nearby_src_position_1, SkyCoord(ra=Sample_RA[self.OptimalPointingIdx_1]*u.degree, dec=Sample_DEC[self.OptimalPointingIdx_1]*u.degree)).arcmin
        SRCoptimalRATES_1 = f.scaled_ct_rate(SRCoptimalSEPAR_1, self.count_rate_1, telescop_data["EffArea"], telescop_data["OffAxisAngle"])
        
        self.OptimalPointingIdx_2 = np.where(SNR_2==max(SNR_2))[0][0]
        SRCoptimalSEPAR_2 = f.ang_separation(self.nearby_src_position_2, SkyCoord(ra=Sample_RA[self.OptimalPointingIdx_1]*u.degree, dec=Sample_DEC[self.OptimalPointingIdx_2]*u.degree)).arcmin
        SRCoptimalRATES_2 = f.scaled_ct_rate(SRCoptimalSEPAR_2, self.count_rate_2, telescop_data["EffArea"], telescop_data["OffAxisAngle"])     

        self.vector_dictionary = {'Sample_RA': Sample_RA,
                                  'Sample_DEC': Sample_DEC,
                                  'PSRrates': PSRrates,
                                  'vecteur_1':{'SRCrates_1': SRCrates_1,
                                               'SRCoptimalRATES_1':SRCoptimalRATES_1,
                                               'SNR_1': SNR_1
                                               },
                                  'vecteur_2':{'SRCrates_1': SRCrates_1,
                                               'SRCoptimalRATES_2': SRCoptimalRATES_2,
                                               'SNR_2': SNR_2
                                               }
                                  }

        if self.catalog_1_name == "Xmm_DR13" and self.catalog_2_name == "CSC_2.0":
            title_0, title_1 = f"Xmm_DR13\nopti_point : {self.vector_dictionary['Sample_RA'][self.OptimalPointingIdx_1]}, {self.vector_dictionary['Sample_DEC'][self.OptimalPointingIdx_1]}", f"CSC_2.0\nopti_point: {self.vector_dictionary['Sample_RA'][self.OptimalPointingIdx_2]}, {self.vector_dictionary['Sample_DEC'][self.OptimalPointingIdx_2]}"
        elif self.catalog_1_name == "CSC_2.0" and self.catalog_2_name == "Xmm_DR13":
            title_0, title_1 = f"CSC_2.0\nopti_point : {self.vector_dictionary['Sample_RA'][self.OptimalPointingIdx_1]}, {self.vector_dictionary['Sample_DEC'][self.OptimalPointingIdx_1]}", f"Xmm_DR13\nopti_point: {self.vector_dictionary['Sample_RA'][self.OptimalPointingIdx_2]}, {self.vector_dictionary['Sample_DEC'][self.OptimalPointingIdx_2]}"
        
        figure_map, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
        figure_map.suptitle(f"S/N map for {object_data['object_name']}")
        figure_map.text(0.5, 0.04, 'Right Ascension [deg]', ha='center', va='center', fontsize=16)
        figure_map.text(0.04, 0.5, 'Declination [deg]', ha='center', va='center', rotation='vertical', fontsize=16)

        ax0 = axes[0]
        ax0.plot(self.nearby_src_position_1.ra, self.nearby_src_position_1.dec, marker='.', color='black', linestyle='')
        ax0.plot(object_data["object_position"].ra, object_data["object_position"].dec, marker='*', color="green", linestyle='')
        ax0.plot(self.vector_dictionary['Sample_RA'][self.OptimalPointingIdx_1], self.vector_dictionary['Sample_DEC'][self.OptimalPointingIdx_1], marker="+", color='red', linestyle='')
        ax0.scatter(self.vector_dictionary['Sample_RA'], self.vector_dictionary['Sample_DEC'], c=self.vector_dictionary["vecteur_1"]["SNR_1"], s=10, edgecolor='face')
        ax0.set_title(title_0)

        ax1 = axes[1]
        ax1.plot(self.nearby_src_position_2.ra, self.nearby_src_position_2.dec, marker='.', color='black', linestyle='')
        ax1.plot(object_data["object_position"].ra, object_data["object_position"].dec, marker='*', color="green", linestyle='')
        ax1.plot(self.vector_dictionary['Sample_RA'][self.OptimalPointingIdx_2], self.vector_dictionary['Sample_DEC'][self.OptimalPointingIdx_2], marker="+", color='red', linestyle='')
        ax1.scatter(self.vector_dictionary['Sample_RA'], self.vector_dictionary['Sample_DEC'], c=self.vector_dictionary["vecteur_2"]["SNR_2"], s=10, edgecolor='face')
        ax1.set_title(title_1)
 
        norm = plt.Normalize(vmin=min(self.vector_dictionary["vecteur_1"]["SNR_1"] + self.vector_dictionary["vecteur_2"]["SNR_2"]), vmax=max(self.vector_dictionary["vecteur_1"]["SNR_1"] + self.vector_dictionary["vecteur_2"]["SNR_2"]))
        sm = ScalarMappable(cmap='viridis', norm=norm)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        color_bar = figure_map.colorbar(sm, cax=cax)
        color_bar.set_label("S/N")
        
        plt.show()
        
        
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