# --------------- Packages --------------- #

from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from scipy.optimize import curve_fit
from astroquery.esasky import ESASky
from astropy.visualization import PercentileInterval, ImageNormalize, LinearStretch
from astropy.wcs import WCS
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
from typing import Dict, Tuple, Union, List

import function as f
import sys
import numpy as np
import matplotlib.pyplot as plt
import pyvo as vo
import subprocess

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


    def __init__(self, catalog_path, radius, dictionary, user_table) -> None:
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
        
        test_dr11_path = "data/4XMM_DR11cat_v1.0.fits"
        test_x2a_path = "data/xmm2athena_D6.1_V3.fits"
        xmm_dr11_path = f.get_valid_file_path(test_dr11_path)
        x2a_path = f.get_valid_file_path(test_x2a_path)
        
        self.xmm_dr11_catalog = self.open_catalog(catalog_path=xmm_dr11_path)
        self.x2a_catalog = self.open_catalog(catalog_path=x2a_path)
        self.nearby_sources_table, self.index_table = self.get_phoindex_nh()
        self.variability_table = self.variability_table(dictionary)
        self.neighbourhood_of_object(radius=radius, dictionary=dictionary)
        self.model_dictionary = self.dictionary_model()
        
    
    def open_catalog(self, catalog_path) -> Table:
        """
        Opens and reads the catalog data from the specified file path.

        This method opens a FITS file from the given path, reads the data from the first HDU extension,
        converts it into an Astropy Table, and returns the resulting table.

        Parameters:
            catalog_path (str): The file path to the catalog to be opened and read.

        Returns:
            result_table (astropy.table.Table): The catalog data converted into an Astropy Table.
        """
        with fits.open(catalog_path) as data:
            result_table = Table(data[1].data)
            data.close()
            return result_table
        
    
    def find_nearby_sources(self, radius, dictionary, user_table) -> Tuple[Table, SkyCoord]:
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
                
        src_position = SkyCoord(ra=small_table['SC_RA'], dec=small_table['SC_DEC'], unit=u.deg)
                
        if len(user_table) == 0:
            for number in range(len(small_table)):
                if f.ang_separation(object_position, src_position[number]) < radius:
                    nearby_src_table.add_row(small_table[number])
            nearby_src_position = SkyCoord(ra=nearby_src_table['SC_RA'], dec=nearby_src_table['SC_DEC'], unit=u.deg)

        else:
            for number in range(len(user_table)):
                small_table.add_row(user_table[number])
                
            for number in range(len(small_table)):
                if f.ang_separation(object_position, src_position[number]) < radius:
                    nearby_src_table.add_row(small_table[number])
            nearby_src_position = SkyCoord(ra=nearby_src_table['SC_RA'], dec=nearby_src_table['SC_DEC'], unit=u.deg)
                
        try:
            if len(nearby_src_table) != 0:
                print((f"We have detected {len(nearby_src_table)} sources close to {dictionary['object_name']}"))
                return nearby_src_table, nearby_src_position
            else:
                print(f"No sources detected close to {dictionary['object_name']}.")
                sys.exit()
        except Exception as error:
            print(f"An error occured : {error}")
            
       
    def optimization_phoindex(self, number) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[float, float]]:
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
        def power_law(x, constant, gamma):
            return constant * x ** (-gamma)
        
        def absorbed_power_law(x, sigma, constant, gamma):
            return (constant * x ** (-gamma)) * (np.exp(-sigma * 3e20))
        
        energy_band = np.array([0.35, 0.75, 1.5, 3.25, 8.25], dtype=float)
        sigma = np.array([1e-20, 5e-21, 1e-22, 1e-23, 1e-24])
        flux_obs = np.array([self.nearby_sources_table[f"SC_EP_{item + 1}_FLUX"][number] for item in range(5)], dtype=float)
        flux_obs_err = np.array([self.nearby_sources_table[f"SC_EP_{item + 1}_FLUX_ERR"][number] for item in range(5)], dtype=float)
        
        popt_1, pcov_1 = curve_fit(lambda x, constant, gamma: power_law(x, constant, gamma), energy_band, flux_obs, sigma=flux_obs_err)
        popt_2, pcov_2 = curve_fit(lambda x, constant, gamma: absorbed_power_law(x, sigma, constant, gamma), energy_band, flux_obs, sigma=flux_obs_err)
        constant_1, pho_index = popt_1
        constant_2, absorb_pho_index = popt_2
        
        photon_index = (pho_index, absorb_pho_index)
        optimization_parameters = (energy_band, flux_obs, flux_obs_err, power_law(energy_band, *popt_1), absorbed_power_law(energy_band, sigma, *popt_2))
        
        return optimization_parameters, photon_index


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
        number_interp = len(optimization_parameters)
        number_column = 4
        number_row = number_interp/number_column
        
        if number_row < 1:
            number_row = 1
        elif number_row %1 == 0:
            number_row= int(number_interp/4)
        else:
            number_row = int(number_interp/4) + 1
            
        index_figure, axes = plt.subplots(nrows=number_row, ncols=number_column, figsize=(17, 8), sharex=True)
        index_figure.subplots_adjust(wspace=0.5, hspace=1.5)
        index_figure.suptitle("Interpolation Photon Index", fontsize=20)
        index_figure.text(0.5, 0.04, 'Energy [keV]', ha='center', va='center')
        index_figure.text(0.04, 0.5, 'Flux [erg/cm^2/s]', ha='center', va='center', rotation='vertical')

        count = 0
        for row in range(number_row):
            for column in range(number_column):
                if count < number_interp:
                    energy_band = optimization_parameters[count][0]
                    flux_obs = optimization_parameters[count][1]
                    flux_obs_err = optimization_parameters[count][2]
                    power_law = optimization_parameters[count][3]
                    absorbed_power_law = optimization_parameters[count][4]
                    pho_index = photon_index[count][0]
                    absorb_pho_index = photon_index[count][1]
                    
                    axes[row][column].errorbar(energy_band, flux_obs, flux_obs_err, fmt='*', color='red', ecolor='black')
                    axes[row][column].plot(energy_band, power_law, linestyle='dashdot', color="navy", label="Non Absorb")
                    axes[row][column].plot(energy_band, absorbed_power_law, linestyle='dashdot', color="darkorange", label="Absorb")
                    axes[row][column].set_title(f"Non absorb $\Gamma$ = {pho_index:.8f}\nAbsorb $\Gamma$ = {absorb_pho_index:.8f}", fontsize=7)
                    axes[row][column].legend(loc="upper left", ncol=2, fontsize=6)
                count += 1
                
        plt.show()


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
                column_nh = np.append(column_nh, np.exp(nh_value) * np.log(10))
                column_phoindex = np.append(column_phoindex, self.x2a_catalog['PhoIndex_med'][number])
            else:
                column_nh = np.append(column_nh, 3e20)
                parameters, pho_value = self.optimization_phoindex(number)
                optimization_parameters.append(parameters)
                photon_index.append(pho_value)
                column_phoindex = np.append(column_phoindex, pho_value[1])

        self.visualization_interp(optimization_parameters=optimization_parameters, photon_index=photon_index)
        
        col_names = ["Photon Index", "Nh"]
        col_data = [column_phoindex, column_nh]
        
        for name, data in zip(col_names, col_data):
            self.nearby_sources_table[name] = data
        
        return self.nearby_sources_table, index_table
    
    
    def variability_table(self, dictionary) -> Table:
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
    
    
    def neighbourhood_of_object(self, radius, dictionary) -> None:
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
            
            plt.show()
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
            
            plt.show()


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

    def __init__(self, catalog_path, radius, dictionary, user_table) -> None:
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
        self.nearby_soucres_table, self.nearby_sources_position = self.find_nearby_sources(radius=radius, dictionary=dictionary, user_table=user_table)

        self.cone_search_catalog = self.load_cs_catalog(radius=radius, dictionary=dictionary)
        self.cs_nearby_sources_position = SkyCoord(ra=list(self.cone_search_catalog['ra']), dec=list(self.cone_search_catalog['dec']), unit=u.deg)
        self.cs_nearby_soucres_table = self.cone_catalog()
        
        self.neighbourhood_of_object(radius=radius, dictionary=dictionary)
        self.photon_index = self.power_law_pho_index()
        self.model_dictionary = self.dictionary_model()
        

    def open_catalog(self, catalog_path) -> Table:
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
        with fits.open(catalog_path) as data:
            result_table = Table(data[1].data)
            data.close()
            return result_table
        
        
    def load_cs_catalog(self, radius, dictionary) -> Table:
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


    def find_nearby_sources(self, radius, dictionary, user_table) -> Tuple[Table, SkyCoord]:
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

        object_position = dictionary['object_position']

        min_ra, max_ra = object_position.ra - field_of_view, object_position.ra + field_of_view
        min_dec, max_dec = object_position.dec - field_of_view, object_position.dec + field_of_view

        small_table = Table(names=self.chandra_catalog.colnames,
                            dtype=self.chandra_catalog.dtype)

        nearby_src_table = Table(names=self.chandra_catalog.colnames,
                                 dtype=self.chandra_catalog.dtype)

        for number in range(len(self.chandra_catalog)):
            if min_ra/u.deg < self.chandra_catalog["RA"][number] < max_ra/u.deg and min_dec/u.deg < self.chandra_catalog["DEC"][number] < max_dec/u.deg:
                small_table.add_row(self.chandra_catalog[number])
                
        src_position = SkyCoord(ra=small_table['RA'], dec=small_table['DEC'], unit=u.deg)
                
        for number in range(len(small_table)):
            if f.ang_separation(object_position, src_position[number]) < radius:
                nearby_src_table.add_row(small_table[number])
                   
        column_names = list(self.chandra_catalog.colnames )  
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
            
    
    def cone_catalog(self) -> Table:
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
                
        data_list = [cone_catalog['name'], cone_catalog['ra'], cone_catalog['dec'], var_column]

        self.cs_catalog = Table(names=['IAUNAME', 'RA', 'DEC', 'VAR'],
                                data=data_list)
        
        return self.cs_catalog
    
    
    def neighbourhood_of_object(self, radius, dictionary) -> None:
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
        cs_csc_ra = np.array(list(self.cone_search_catalog['ra']), dtype=float)
        cs_csc_dec = np.array(list(self.cone_search_catalog['dec']), dtype=float)
        
        csc_ra = np.array(self.nearby_soucres_table['RA'], dtype=float)
        csc_dec = np.array(self.nearby_soucres_table['DEC'], dtype=float)
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

        cs_ra_var = [ra for index, ra in enumerate(list(self.cs_catalog['RA'])) if self.cs_catalog['VAR'][index] != 0.0]
        cs_ra_invar = [ra for index, ra in enumerate(list(self.cs_catalog['RA'])) if self.cs_catalog['VAR'][index] == 0.0]

        cs_dec_var = [dec for index, dec in enumerate(list(self.cs_catalog['DEC'])) if self.cs_catalog['VAR'][index] != 0.0]
        cs_dec_invar = [dec for index, dec in enumerate(list(self.cs_catalog['DEC'])) if self.cs_catalog['VAR'][index] == 0.0]

        ax11 = axes[1][1]
        ax11.scatter(cs_ra_var, cs_dec_var, s=10, c='darkorange', marker='*', label=f"Var src : {len(cs_ra_var)} sources")
        ax11.scatter(cs_ra_invar, cs_dec_invar, s=10, c='blue', marker='*', label=f"Invar src : {len(cs_ra_invar)} sources")
        ax11.scatter(dictionary['object_position'].ra, dictionary['object_position'].dec, marker='+', s=50, c='red', label=f"{dictionary['object_name']}")
        ax11.legend(loc="upper right", ncol=2)
        
        plt.show()


    def get_pho_index(self, number) -> Tuple[Tuple[float, float], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Calculate the photon index for an astronomical object using the Harvard catalog data.

        Parameters
        ----------
        number : int
            The index of the astronomical object in the catalog.

        Returns
        -------
        photon_index : Tuple[float, float]
            A tuple containing the non-absorbed and absorbed photon indices respectively.
        optimization_parameters : Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            A tuple containing arrays of energy bands, observed fluxes, non-absorbed power law values, and absorbed power law values.
        
        Raises
        ------
        RuntimeError
            Raises a runtime error if curve fitting fails.
        """
        # with Harvard catalog
        column_name = np.array(["flux_aper_s", "flux_aper_m", "flux_aper_h"], dtype=str)
        energy_band = np.array([0.7, 1.6, 4.5], dtype=float)
        sigma = np.array([1e-20, 1e-22, 1e-24], dtype=float)
        flux_obs = np.array([self.cone_search_catalog[flux][number] for flux in column_name])
        flux_obs = np.nan_to_num(flux_obs, nan=0.0)
        
        def power_law(x, constant, gamma):
            return constant * x **(-gamma)
        
        def absorbed_power_law(x, sigma, constant, gamma):
            return (constant * x **(-gamma)) * (np.exp(-sigma*3e20))
        
        try:
            popt_1, pcov_1 = curve_fit(power_law, energy_band, flux_obs)
            constant_1, photon_index_non_absorb = popt_1
            powerlaw_value_non_absorb = power_law(energy_band, *popt_1)
            
            popt_2, pcov_2 = curve_fit(lambda x, constant, gamma: absorbed_power_law(x, sigma, constant, gamma), energy_band, flux_obs)
            constant_2, photon_index_absorb = popt_2
            powerlaw_value_absorb = absorbed_power_law(energy_band, sigma, *popt_2)
            
        except RuntimeError as error:
            photon_index_non_absorb = 1.7
            powerlaw_value_non_absorb = [0.0, 0.0, 0.0]
            
            photon_index_absorb = 1.7
            powerlaw_value_absorb = [0.0, 0.0, 0.0]
            
        photon_index = (photon_index_non_absorb, photon_index_absorb)
        optimization_parameters = (energy_band, flux_obs, powerlaw_value_non_absorb, powerlaw_value_absorb)
        return photon_index, optimization_parameters


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
                    power_law = optimization_parameters[count][2]
                    absorb_power_law = optimization_parameters[count][3]
                    
                    pho_index = photon_index[count][0]
                    absorb_pho_index = photon_index[count][1]
                    
                    axes[row][column].errorbar(energy_band, flux_obs, fmt='*', color='red', ecolor='black')
                    axes[row][column].plot(energy_band, power_law, linestyle='dashdot', color="navy")
                    axes[row][column].plot(energy_band, absorb_power_law, linestyle='dashdot', color="darkorange")
                    axes[row][column].set_title(f"Non absorb $\Gamma$ = {pho_index:.8f}\nAbsorb $\Gamma$ = {absorb_pho_index:.8f}", fontsize=7)
                
                count += 1
                
        plt.show()


    def power_law_pho_index(self) -> List[tuple[float, float]]:
        """
        Calculate and visualize the power law photon indices for all sources in the cone search catalog.

        Returns
        -------
        photon_index : List[Tuple[float, float]]
            List of tuples containing the non-absorbed and absorbed photon indices respectively for each source.
        """
        photon_index = []
        optimization_parameters = []
        
        for index, item in enumerate(self.cone_search_catalog["powlaw_gamma"]):
            if item != 0:
                photon_index.append((item, item))
            else:
                pho_value, params_value = self.get_pho_index(index)
                photon_index.append(pho_value)
                optimization_parameters.append(params_value)
                
        self.visualization_interp(optimization_parameters=optimization_parameters, photon_index=photon_index)
        
        return photon_index
    
    
    def dictionary_model(self) -> Dict[str, Dict[str, Union[str, float]]]:
        """
        Create a dictionary of models and their parameters for each source in the cone search catalog.

        Returns
        -------
        model_dictionary : Dict[str, Dict[str, Union[str, float]]]
            Dictionary where each key is a source identifier and the corresponding value is another dictionary containing model details.
        """
        model_dictionary = {}
        nbr_src = len(self.cone_search_catalog)
        
        model = np.array(['power' for item in range(nbr_src)], dtype=str)
        model_value = []
        csc_flux = np.array(list(self.cone_search_catalog['flux_aper_b']))
        csc_flux = np.nan_to_num(csc_flux, nan=0.0)
        nh_value = np.array(list(self.cone_search_catalog['nh_gal']*1e20))
        
        for item in range(nbr_src):
            if model[item] == 'power':
                model_value.append(self.photon_index[item][1])
        
        for item in range(nbr_src):

            dictionary = {
                "model": model[item],
                "model_value": model_value[item],
                "flux": csc_flux[item],
                "column_dentsity": nh_value[item]
            }

            model_dictionary[f"src_{item}"] = dictionary

        return model_dictionary


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
    
    def __init__(self, catalog_path, radius, dictionary, user_table) -> None:
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
        self.nearby_soucres_table, self.nearby_sources_position = self.find_nearby_sources(radius=radius, dictionary=dictionary)
        
        self.neighbourhood_of_object(radius=radius, dictionary=dictionary)
        self.model_dictionary = self.dictionary_model()
        
    
    def open_catalog(self, catalog_path)-> Table:
        """
        Opens and reads a FITS catalog file and returns it as an Astropy Table.

        Parameters:
        - catalog_path (str): Path to the Swift catalog file.

        Returns:
        - Table: The catalog of Swift sources.
        """
        with fits.open(catalog_path) as data:
            result_table = Table(data[1].data)
            data.close()
            return result_table
        
        
    def find_nearby_sources(self, radius, dictionary) -> Tuple[Table, SkyCoord]:
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


    def neighbourhood_of_object(self, radius, dictionary) -> None:
        """
        Visualizes the neighborhood of the target object and nearby sources.

        Parameters:
        - radius (float): Search radius (in degrees) for finding nearby sources.
        - dictionary (dict): A dictionary containing information about the target object.
        """

        object_position = dictionary['object_position']
    
        swi_ra = self.nearby_soucres_table['RA']
        swi_dec = self.nearby_soucres_table['DEC']
        
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
    
    def __init__(self, catalog_path, radius, dictionary, user_table) -> None:
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
        self.nearby_soucres_table, self.nearby_sources_position = self.find_nearby_sources(radius=radius, dictionary=dictionary)
        
        self.neighbourhood_of_object(radius=radius, dictionary=dictionary)
        self.model_dictionary = self.dictionary_model()
        
        
    def open_catalog(self, catalog_path) -> Table:
        """
        Opens and reads a FITS catalog file and returns it as an Astropy Table.

        Parameters:
        - catalog_path (str): Path to the eRosita catalog file.

        Returns:
        - Table: The catalog of eRosita sources.
        """
        with fits.open(catalog_path) as data:
            result_table = Table(data[1].data)
            data.close()
            return result_table
        
        
    def find_nearby_sources(self, radius, dictionary) -> Tuple[Table, SkyCoord]:
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
    
    
    def neighbourhood_of_object(self, dictionary, radius) -> None:
        """
        Visualizes the neighborhood of the target object and nearby sources.

        Parameters:
        - dictionary (dict): A dictionary containing information about the target object.
        - radius (float): Search radius (in degrees) for finding nearby sources.
        """
        object_position = dictionary['object_position']
    
        ero_ra = self.nearby_soucres_table['RA']
        ero_dec = self.nearby_soucres_table['DEC']
        
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
    def __init__(self, catalog_path, radius, dictionary, user_table) -> None:
        self.catalog_1, self.catalog_2, self.catalog_1_name, self.catalog_2_name = self.open_catalog(catalogs_path=catalog_path, radius=radius, dictionary=dictionary)
        self.nearby_src_table_1, self.nearby_src_table_2, self.nearby_src_position_1, self.nearby_src_position_2 = self.find_nearby_sources(radius=radius, dictionary=dictionary)
        self.neighbourhood_of_object(radius=radius, dictionary=dictionary)
        
        self.model_dictionary_1, self.model_dictionary_2 = self.model_dictionary()
        self.count_rate_1, self.count_rate_2 = self.count_rate()
        self.vector_dictionary, self.OptimalPointingIdx_1, self.OptimalPointingIdx_2 = self.opti_point_calcul()

    
    def open_catalog(self, catalogs_path, radius, dictionary) -> Tuple[Table, Table, str, str]:
        if "data/Chandra.fits" not in catalogs_path:
            with fits.open(catalogs_path[0]) as data1, fits.open(catalogs_path[1]) as data2:
                result_1, result_2 = Table(data1[1].data), Table(data2[1].data)
                data1.close()
                data2.close()
                return result_1, result_2, catalogs_path[2], catalogs_path[3]
        else:
            index = catalogs_path.index("data/Chandra.fits")
            if index == 0 :
                with fits.open(catalogs_path[1]) as data:
                    result_2 = Table(data[1].data)
                    data.close()
                cone = vo.dal.SCSService('http://cda.cfa.harvard.edu/csc2scs/coneSearch') 
                name = SkyCoord.from_name(dictionary['object_name'])
                self.cone_search_catalog = cone.search(pos=name, radius=radius, verbosity=3)
                return self.cone_search_catalog.to_table(), result_2, catalogs_path[2], catalogs_path[3]
            
            elif index == 1 :
                with fits.open(catalogs_path[0]) as data:
                    result_1 = Table(data[1].data)
                    data.close()
                cone = vo.dal.SCSService('http://cda.cfa.harvard.edu/csc2scs/coneSearch') 
                name = SkyCoord.from_name(dictionary['object_name'])
                self.cone_search_catalog = cone.search(pos=name, radius=radius, verbosity=3)
                return result_1, self.cone_search_catalog.to_table(), catalogs_path[2], catalogs_path[3]

         
    def optimization(self, number, table) -> Tuple[float, float]:
        
        def power_law(x, constant, gamma):
            return constant * (x ** (-gamma))
        
        energy_band = np.array([0.35, 0.75, 1.5, 3.25, 8.25], dtype=float)
        
        flux = np.array([], dtype=float)
        for item in range(5):
            flux = np.append(flux, table[f"SC_EP_{item + 1}_FLUX"][number])
            
        try:
            popt, pcov = curve_fit(power_law, energy_band, flux)
            constant, photon_index = popt
            return constant, photon_index
        except Exception as error:
            constant, photon_index = 2e-15, 1.0
            return constant, photon_index
         
         
    def add_photon_nh_and_gamma_xmm(self, table) -> Tuple[Table, Table]:
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
         
           
    def add_pho_csc(self, table) -> Table:
        
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
    
    
    def var_function(self, dictionary) -> Tuple[Table, Table]:
        if self.catalog_1_name == "CSC_2.0" or self.catalog_2_name == "Xmm_DR13" and self.catalog_1_name =="Xmm_DR13" or self.catalog_2_name =="CSC_2.0":
            x2a_path = "data/xmm2athena_D6.1_V3.fits"
            xmm_path = "data/4XMM_DR11cat_v1.0.fits"
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
            
            
    def find_nearby_sources(self, radius, dictionary) -> Tuple[Table, Table, SkyCoord, SkyCoord]:
        
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


    def neighbourhood_of_object(self, radius, dictionary) -> None:
        
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
                axes_0 = figure.add_subplot(221, projection=_wcs_)
                axes_1 = figure.add_subplot(223, projection=_wcs_, sharex=axes_0, sharey=axes_0)
                axes_2 = figure.add_subplot(222)
                axes_3 = figure.add_subplot(224)
                axes_2.tick_params(axis='x', which='both', labelbottom=False)
                
                xmm_ra = list(self.nearby_src_table_1['SC_RA'])
                xmm_dec = list(self.nearby_src_table_1['SC_DEC'])
                csc_ra = list(self.nearby_src_table_2["ra"])
                csc_dec = list(self.nearby_src_table_2["dec"])
                cs_ra_var = [ra for index, ra in enumerate(list(self.nearby_src_table_2['ra'])) if self.nearby_src_table_2['Variability'][index] != 0.0]
                cs_ra_invar = [ra for index, ra in enumerate(list(self.nearby_src_table_2['ra'])) if self.nearby_src_table_2['Variability'][index] == 0.0]
                cs_dec_var = [dec for index, dec in enumerate(list(self.nearby_src_table_2['dec'])) if self.nearby_src_table_2['Variability'][index] != 0.0]
                cs_dec_invar = [dec for index, dec in enumerate(list(self.nearby_src_table_2['dec'])) if self.nearby_src_table_2['Variability'][index] == 0.0]
                
            elif self.catalog_1_name == "CSC_2.0" and self.catalog_2_name == "Xmm_DR13":
                axes_0 = figure.add_subplot(222, projection=_wcs_)
                axes_1 = figure.add_subplot(224, projection=_wcs_, sharex=axes_0, sharey=axes_0)
                axes_2 = figure.add_subplot(221)
                axes_3 = figure.add_subplot(223)
                axes_2.tick_params(axis='x', which='both', labelbottom=False)
                
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
                axes_0 = figure.add_subplot(221)
                axes_1 = figure.add_subplot(223)
                axes_2 = figure.add_subplot(222)
                axes_3 = figure.add_subplot(224)
                axes_0.tick_params(axis='x', which='both', labelbottom=False)
                axes_2.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
                axes_3.tick_params(axis='y', which='both', labelleft=False)
                
                xmm_ra = list(self.nearby_src_table_1['SC_RA'])
                xmm_dec = list(self.nearby_src_table_1['SC_DEC'])
                csc_ra = list(self.nearby_src_table_2["ra"])
                csc_dec = list(self.nearby_src_table_2["dec"])
                cs_ra_var = [ra for index, ra in enumerate(list(self.nearby_src_table_2['ra'])) if self.nearby_src_table_2['Variability'][index] != 0.0]
                cs_ra_invar = [ra for index, ra in enumerate(list(self.nearby_src_table_2['ra'])) if self.nearby_src_table_2['Variability'][index] == 0.0]
                cs_dec_var = [dec for index, dec in enumerate(list(self.nearby_src_table_2['dec'])) if self.nearby_src_table_2['Variability'][index] != 0.0]
                cs_dec_invar = [dec for index, dec in enumerate(list(self.nearby_src_table_2['dec'])) if self.nearby_src_table_2['Variability'][index] == 0.0]
                
            elif self.catalog_1_name == "CSC_2.0" and self.catalog_2_name == "Xmm_DR13":
                axes_0 = figure.add_subplot(222)
                axes_1 = figure.add_subplot(224)
                axes_2 = figure.add_subplot(221)
                axes_3 = figure.add_subplot(223)
                axes_0.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
                axes_2.tick_params(axis='x', which='both', labelbottom=False)
                axes_1.tick_params(axis='y', which='both', labelleft=False)
                
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
        axes_0.set_title(f"{self.catalog_1_name}")
        
        axes_1.legend(loc='upper right', ncol=2, fontsize=8)
        
        axes_2.scatter(csc_ra, csc_dec, facecolors='none', edgecolors='black', s=30, label=f"Nearby sources : {len(csc_ra)}")
        axes_2.scatter(dictionary["object_position"].ra, dictionary["object_position"].dec, s=100, marker="*", facecolors='none', edgecolors='red', label=f"{dictionary['object_name']}")
        axes_2.legend(loc="upper right", fontsize=8)
        axes_2.set_title(f"{self.catalog_2_name}")
        
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
    
    
    def opti_point_calcul(self, simulation_data) -> None:
        object_data = simulation_data["object_data"]
        telescop_data = simulation_data["telescop_data"]
            
        Delta_RA, Delta_DEC = Angle(np.arange(-3.0, 3.1, 0.1), unit=u.deg)/60, Angle(np.arange(-3.0, 3.1, 0.1), unit=u.deg)/60
        
        Sample_RA, Sample_DEC, PSRrates =  np.zeros((3, len(Delta_RA) * len(Delta_DEC)))
        
        SNR_1, SRCrates_1 = np.zeros((2, len(Delta_RA) * len(Delta_DEC)))
        SNR_2, SRCrates_2 = np.zeros((2, len(Delta_RA) * len(Delta_DEC)))
        
        PSRcountrates = object_data['CountRate']
        
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
        