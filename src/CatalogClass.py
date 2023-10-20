import numpy as np
import sys
from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import Function as F
from scipy.optimize import curve_fit
import pyvo as vo


class XmmCatalog:
    """
    A class for processing and analyzing XMM-Newton (XMM) DR13 catalog data and related NICER parameters.

    This class provides methods for opening the XMM-Newton DR13 catalog, transforming source coordinates to SkyCoord
    objects, loading NICER parameters from a file, creating a table of nearby sources, plotting a neighborhood map,
    and calculating count rates for each source in the catalog.

    Parameters:
    - XMM_DR13_PATH (str): The path to the XMM DR13 catalog file.
    - NICER_PARAMETERS_PATH (str): The path to the file containing NICER parameters.

    Attributes:
    - catalog (astropy.table.Table): The XMM-Newton DR13 catalog data.
    - NICER_PARAMETERS (tuple): NICER parameters (EffArea and OffAxisAngle) loaded from a file.
    - NearbySources_Table (astropy.table.Table): Table of nearby sources.

    Methods:
    - open_catalog(XMM_DR13_path): Open and load the XMM-Newton DR13 catalog as an astropy Table.
    - NICER_PARAMETERS(NICER_PARAMETERS_PATH): Load NICER parameters (EffArea and OffAxisAngle) from a file.
    - nearby_sources_table(object_data, user_table, catalog): Create a table of nearby sources within a specified field of view around the object's position.
    - neighbourhood_of_object(nearby_src_table, object_data, var_src_table): Plot a neighborhood map for the object.

    Returns:
    Various data types, depending on the specific method's purpose and return values.
    """

    def __init__(self, catalog_path, radius, dictionary, user_table):
        """
        Constructor for XmmCatalog class.

        Initializes an instance of the XmmCatalog class by opening the XMM DR13 catalog file,
        transforming source coordinates to SkyCoord objects, and loading the NICER parameters
        from the specified files.

        :param XMM_DR13_PATH: The path to the XMM DR13 catalog file.
        :type XMM_DR13_PATH: str
        :param NICER_PARAMETERS_PATH: The path to the file containing NICER parameters.
        :type NICER_PARAMETERS_PATH: str
        """
        self.catalog = self.open_catalog(catalog_path)
        self.nearby_src_table, self.nearby_src_position, self.nbr_var_src = self.find_nearby_src(radius=radius, dictionary=dictionary, user_table=user_table)


    def open_catalog(self, catalog_path):
        """
        Opens an XMM DR13 catalog file and returns it as an astropy Table.

        :param XMM_DR13_PATH: The path to the XMM DR13 catalog file.
        :type XMM_DR13_PATH: str
        :return: The catalog Table or None if an error occurs.
        :rtype: astropy.table.Table
        """
        with fits.open(catalog_path) as data:
            self.result_table = Table(data[1].data)
            data.close()
        
        return self.result_table
            

    def find_nearby_src(self, radius, dictionary, user_table):

        field_of_view = radius + 5*u.arcmin

        object_position = dictionary['object_position']

        min_ra, max_ra = object_position.ra - field_of_view, object_position.ra + field_of_view
        min_dec, max_dec = object_position.dec - field_of_view, object_position.dec + field_of_view
        
        table = Table(names=self.catalog.colnames,
                      dtype=self.catalog.dtype)

        self.nearby_src_table = Table(names=self.catalog.colnames,
                                      dtype=self.catalog.dtype)

        for number in range(len(self.catalog)):
            if min_ra/u.deg < self.catalog["SC_RA"][number] < max_ra/u.deg and min_dec/u.deg < self.catalog["SC_DEC"][number] < max_dec/u.deg:
                table.add_row(self.catalog[number])

        nbr_source = len(table)
        SRCposition = SkyCoord(ra=table['SC_RA'], dec=table['SC_DEC'], unit=u.deg)
        
        self.nbr_var_src = 0
        if len(user_table) != 0:
            for number in range(nbr_source):
                if F.ang_separation(dictionary["object_position"], SRCposition[number]) < 5*u.arcmin:
                    self.nearby_src_table.add_row(table[number])
                    
            User_position = SkyCoord(ra=user_table["Right Ascension"], dec=user_table["Declination"], unit=u.deg)
            
            for number in range(len(user_table)):
                if F.ang_separation(dictionary["object_position"], User_position[number]) < 5*u.arcmin:
                    new_row = [0, user_table["Name"][number], user_table["Right Ascension"][number], user_table["Declination"][number]] + [0] * 28 + [user_table["Var Value"][number]] + [0]*11 + ['no website URL']
                    self.nearby_src_table.add_row(new_row)
                    self.nbr_var_src += 1

        else:
            src_position = SkyCoord(ra=table['SC_RA'], dec=table['SC_DEC'], unit=u.deg)
            
            for number in range(len(table)):
                if F.ang_separation(object_position, src_position[number]) < radius:
                    self.nearby_src_table.add_row(table[number])
        
        self.nearby_src_position = SkyCoord(ra=self.nearby_src_table['SC_RA'], dec=self.nearby_src_table['SC_DEC'], unit=u.deg)
        
        try :
            if len(self.nearby_src_table) != 0:
                print((f"We have detected {len(self.nearby_src_table)} sources close to {dictionary['object_name']}"))
                return self.nearby_src_table, self.nearby_src_position, self.nbr_var_src
            else:
                print(f"No sources detected close to {dictionary['object_name']}.")
                sys.exit()
        except Exception as error:
            print(f"An error occured : {error}")

    
    def neighbourhood_of_object(self, nearby_src_table, variability_table, simulation_data):
        """
        Visualize the neighborhood of a celestial object, highlighting nearby sources and their variability.

        This method generates a pair of scatterplots to visualize the neighborhood of a celestial object. 
        It highlights the positions of nearby sources and their variability status. The first plot shows
        nearby sources and the celestial object's position, while the second plot distinguishes variable
        and invariable sources.

        Parameters:
        nearby_src_table (Table): Table containing information about nearby sources.
        variability_table (Table): Table containing information about source variability.
        simulation_data (dict): Data related to the celestial object, including its position.

        Returns:
        None
        """
        self.nearby_src_table = nearby_src_table
        
        nbr_src_var = len(variability_table)
        nbr_src = len(self.nearby_src_table)
        obj_ra = simulation_data["object_data"]["object_position"].ra/u.deg
        obj_dec = simulation_data["object_data"]["object_position"].dec/u.deg
            
        sc_ra_in_x2a, sc_dec_in_x2a = np.array([], dtype=float), np.array([], dtype=float)
        sc_ra_in_dr11, sc_dec_in_dr11 = np.array([], dtype=float), np.array([], dtype=float)
        
        for index in range(nbr_src_var):
            if variability_table["IN_X2A"][index] == True:
                sc_ra_in_x2a = np.append(sc_ra_in_x2a, variability_table["SC_RA"][index])
                sc_dec_in_x2a = np.append(sc_dec_in_x2a, variability_table["SC_DEC"][index])
            else:
                sc_ra_in_dr11 = np.append(sc_ra_in_dr11, variability_table["SC_RA"][index])
                sc_dec_in_dr11 = np.append(sc_dec_in_dr11, variability_table["SC_DEC"][index])
                
        invar_sc_ra, invar_sc_dec = np.array([], dtype=float), np.array([], dtype=float)
        for ra in self.nearby_src_table["SC_RA"]:
            if ra not in variability_table["SC_RA"]:
                invar_sc_ra = np.append(invar_sc_ra, ra)
                
        for dec in self.nearby_src_table["SC_DEC"]:
            if dec not in variability_table["SC_DEC"]:
                invar_sc_dec = np.append(invar_sc_dec, dec)  
        
        fig, axes = plt.subplots(1, 2, figsize=(17, 8))
        fig.suptitle(f"Neighbourhood of {simulation_data['object_data']['object_name']}", fontsize=20)
        
        ax0 = axes[0]
        ax0.invert_xaxis()
        ax0.scatter(list(self.nearby_src_table["SC_RA"]), list(self.nearby_src_table["SC_DEC"]), color='black', s=10, label="Nearby sources")
        ax0.scatter(obj_ra, obj_dec, color='red', marker='x', s=50, label=f"Position of {simulation_data['object_data']['object_name']}")
        ax0.legend(loc="lower right", ncol=2)
        ax0.set_xlabel("Right Ascension")
        ax0.set_ylabel("Declination")
        ax0.set_title(f"Sources close to {simulation_data['object_data']['object_name']}, {nbr_src} sources")
        
        ax1 = axes[1]
        ax1.invert_xaxis()
        ax1.scatter(invar_sc_ra, invar_sc_dec, color='black', s=10, label=f"Invariant sources, {len(invar_sc_ra)}")
        ax1.scatter(obj_ra, obj_dec, color='red', marker='x', s=50, label=f"Position of {simulation_data['object_data']['object_name']}")
        ax1.scatter(sc_ra_in_x2a, sc_dec_in_x2a, color='darkorange', marker="*", label=f"Variable sources in X2A, {len(sc_ra_in_x2a)}")
        ax1.scatter(sc_ra_in_dr11, sc_dec_in_dr11, color='royalblue', marker="+", label=f"Variable sources not in X2A, {len(sc_ra_in_dr11)}")
        ax1.legend(loc="lower right", ncol=2)
        ax1.set_xlabel("Right Ascension")
        ax1.set_ylabel("Declination")
        ax1.set_title(f"Variable and invariable sources close to {simulation_data['object_data']['object_name']} ")
        
        plt.show()
        
        
    def model_dictionary(self):
        """
            Create a dictionary of models and associated parameters for nearby sources.

            This function generates a dictionary that stores information about the models and their parameters for nearby sources.
            It extracts model information from the 'nearby_src_table', including the model type, model value, X-ray flux, and column density.

            Parameters:
                nearby_src_table (Table): A table containing data on nearby sources, including model information, X-ray flux, and column density.

            Returns:
                dict: A dictionary where each key represents a nearby source (e.g., "src_0") and maps to a sub-dictionary containing:
                    - "model": The type of the model used.
                    - "model_value": The value associated with the model.
                    - "flux": The X-ray flux of the source.
                    - "column_density": The column density value.

            Note:
            - The 'model' field indicates the type of model used, e.g., 'power,' 'black_body,' or 'temp' (the last model in PIMMS).
            - If the model is 'black_body' or 'temp,' there may not be a valid 'model_value' provided.
        """

        model_dictionary = {}
        nbr_src = len(self.nearby_src_table)

        model = np.array([], dtype=str)
        model_value = np.array([], dtype=float)
        xmm_flux = np.array([self.nearby_src_table["SC_EP_8_FLUX"][item] for item in range(nbr_src)], dtype=float)
        nh_value = np.array([self.nearby_src_table["Nh"][item] for item in range(nbr_src)], dtype=float)
        
        # Pour le moment seulement 'power' indiquant le modèle a utiliser pour la commande pimms
        for item in range(nbr_src):
            model = np.append(model, 'power')    
            
        for item in range(nbr_src):
            if model[item] == 'power':
                model_value = np.append(model_value, self.nearby_src_table["Photon Index"][item])
            elif model[item] == 'black_body':
                pass # Pas de valeur pour le moment...
            elif model[item] == 'temp':
                pass # Pas de valeur pour le moment... (dernier model pimms)

        for item in range(nbr_src):

            dictionary = {
                "model": model[item],
                "model_value": model_value[item],
                "flux": xmm_flux[item],
                "column_dentsity": nh_value[item]
            }

            model_dictionary[f"src_{item}"] = dictionary
            
        return model_dictionary 


class Xmm2Athena:
    """
    A class for performing various operations on XMM-Newton and Athena catalogs.

    This class provides methods to open XMM-Newton and Athena catalogs, perform
    optimizations on photon index, visualize interpolation results, and add
    columns for photon index and column density (Nh) to a nearby source table.

    Parameters:
        XMM_DR11_PATH (str): Path to the XMM-Newton DR11 catalog FITS file.
        XMM_2_ATHENA_PATH (str): Path to the XMM-Newton 2 Athena catalog FITS file.

    Attributes:
        nearby_src_table_dr11 (None or astropy.table.Table): Table for nearby sources from XMM-Newton DR11 catalog.
        nearby_src_table_x2a (None or astropy.table.Table): Table for nearby sources from XMM-Newton 2 Athena catalog.

    Methods:
        open_catalog(XMM_DR11_PATH, XMM_2_ATHENA_PATH):
            Opens and reads the XMM-Newton catalogs specified by the provided paths.

        optimization_photon_index(number, nearby_src_table):
            Performs optimization on the photon index for a given nearby source.

        visualization_interpolation(tup_data):
            Visualizes interpolation results for multiple nearby sources.

        add_nh_photon_index(nearby_src_table, user_table):
            Adds columns for photon index and column density (Nh) to a nearby source table.
    """
    
    def __init__(self, xmm_dr11_path, xmm_2_athena_path):
        """
        Initialize an instance of the Xmm2Athena class.

        This constructor opens and reads XMM-Newton catalogs specified by the provided
        file paths. It also initializes attributes for nearby source tables.

        Parameters:
            XMM_DR11_PATH (str): Path to the XMM-Newton DR11 catalog FITS file.
            XMM_2_ATHENA_PATH (str): Path to the XMM-Newton 2 Athena catalog FITS file.

        Attributes:
            XMM_DR11 (astropy.table.Table): XMM-Newton DR11 catalog table.
            XMM_2_ATHENA (astropy.table.Table): XMM-Newton 2 Athena catalog table.
            nearby_src_table_dr11 (None or astropy.table.Table): Table for nearby sources from XMM-Newton DR11 catalog.
            nearby_src_table_x2a (None or astropy.table.Table): Table for nearby sources from XMM-Newton 2 Athena catalog.
        """
        
        self.xmm_dr11, self.xmm_2_athena = self.open_catalog(xmm_dr11_path, xmm_2_athena_path)

        self.nearby_src_table_dr11 = None
        self.nearby_src_table_x2a = None


    def open_catalog(self, xmm_dr11_path, xmm_2_athena_path):
        """
        Open and read XMM-Newton catalogs.

        This method opens and reads the XMM-Newton catalogs specified by the provided paths,
        returning two astropy tables.

        Parameters:
            XMM_DR11_PATH (str): Path to the XMM-Newton DR11 catalog FITS file.
            XMM_2_ATHENA_PATH (str): Path to the XMM-Newton 2 Athena catalog FITS file.

        Returns:
            tuple: A tuple containing two astropy tables (XMM-Newton DR11 and XMM-Newton 2 Athena).

        Notes:
            The returned tables include data from the specified FITS files.
        """
        
        with fits.open(xmm_dr11_path) as data1, fits.open(xmm_2_athena_path) as data2:
            xmm_dr11 = Table(data1[1].data)
            xmm_2_athena = Table(data2[1].data)
            data1.close()
            data2.close()
            
        return xmm_dr11, xmm_2_athena


    def optimization_photon_index(self, number, nearby_src_table):
        """
        Perform optimization on the photon index for a nearby source.

        This method calculates the optimized photon index for a specific nearby source,
        using the provided source table.

        Parameters:
            number (int): Index of the nearby source in the source table.
            nearby_src_table (astropy.table.Table): Table containing nearby sources.

        Returns:
            tuple: A tuple containing the calculated photon index and related data.

        Notes:
            The returned tuple includes the photon index, energy ranges, flux values,
            fitted power-law curve, and optimization parameters.
        """
        
        def power_law(x, constant, gamma):
                return  constant * (x ** ( - gamma))

        col_names = ["SC_EP_1_FLUX", "SC_EP_2_FLUX", "SC_EP_3_FLUX", "SC_EP_4_FLUX", "SC_EP_5_FLUX"]
        col_err_names = ["SC_EP_1_FLUX_ERR", "SC_EP_2_FLUX_ERR", "SC_EP_3_FLUX_ERR", "SC_EP_4_FLUX_ERR", "SC_EP_5_FLUX_ERR"]

        flux_values = np.array([nearby_src_table[name][number] for name in col_names])
        flux_err_values = np.array([nearby_src_table[name][number] for name in col_err_names])
        energy_ranges = np.array([0.35, 0.75, 1.5, 3.25, 8.25])

        popt, pcov = curve_fit(power_law, energy_ranges, flux_values, sigma=flux_err_values)
        constant, photon_index = popt
        
        return photon_index, (energy_ranges, flux_values, flux_err_values, power_law(energy_ranges, *popt), popt)
    
    
    def visualization_interpolation(self, tup_data):
        """
        Visualize interpolation results for multiple nearby sources.

        This method generates plots to visualize interpolation results for multiple nearby
        sources based on the provided data.

        Parameters:
            tup_data (list of tuples): List of tuples containing interpolation data for
            multiple nearby sources.

        Notes:
            The method generates plots showing power-law interpolation and scatter plots
            for each nearby source's photon index.
        """
        nbr_of_interpolation = len(tup_data)
        n_col = 4
        n_row = nbr_of_interpolation/n_col

        if n_row < 1:
            n_row = 1
        elif n_row % 1 == 0:
            n_row = int(nbr_of_interpolation/4)
        else:
            n_row = int(nbr_of_interpolation/4) + 1
        
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(17, 8))
        fig.subplots_adjust(wspace=0.5, hspace=1.5)
        fig.suptitle("Interpolation Photon Index", fontsize=20)
        count = 0
        for row in range(n_row):
            for col in range(n_col):
                if count < nbr_of_interpolation:
                    energy_ranges = tup_data[count][0]
                    flux_values = tup_data[count][1]
                    flux_err_values = tup_data[count][2]
                    power_law = tup_data[count][3]
                    constant, photon_index = tup_data[count][4]

                    axes[row][col].errorbar(energy_ranges, flux_values, flux_err_values, fmt='*', color='red', ecolor='black')
                    axes[row][col].plot(energy_ranges, power_law, linestyle='dashdot', color="navy")
                    axes[row][col].set_xlabel('Energy [keV]', fontsize=7)
                    axes[row][col].set_ylabel('Flux [erg/cm^2/s]', fontsize=7)
                    axes[row][col].set_title(f"Photon_Index : {photon_index:.4f}", fontsize=7)
                    
                count += 1


    def add_nh_photon_index(self, nearby_src_table, user_table):
        """
        Add columns for photon index and column density (Nh) to a nearby source table.

        This method adds new columns for photon index and column density (Nh) to a nearby
        source table, while performing various data operations to link the tables together.

        Parameters:
            nearby_src_table (astropy.table.Table): Table containing nearby sources.
            user_table: Not specified in the method, it seems to be an output variable.

        Returns:
            tuple: A tuple containing the modified nearby source table and an index table.

        Notes:
            This method also generates an index table linking nearby sources to XMM-Newton
            DR11 and XMM-Newton 2 Athena catalogs. It visualizes interpolation results.
        """
        
        user_table = []
        nbr_src = len(nearby_src_table)
        
        name_list = nearby_src_table['IAUNAME']
        
        self.xmm_dr11_table = Table(names=self.xmm_dr11.colnames,
                                    dtype=self.xmm_dr11.dtype)
        
        index_in_xd11 = []
        for name in name_list:
            if name in self.xmm_dr11['IAUNAME']:
                index = list(self.xmm_dr11['IAUNAME']).index(name)
                index_in_xd11.append(index)
                self.xmm_dr11_table.add_row(self.xmm_dr11[index])
            else:
                print(f"{name} is missing")

        index_in_x2a = []
        message = "No data founded"
        for det_id in self.xmm_dr11_table["DETID"]:
            if det_id in self.xmm_2_athena["DETID"]:
                index = list(self.xmm_2_athena["DETID"]).index(det_id)
                index_in_x2a.append(index)
            else:
                index_in_x2a.append(message)

        col_name = ["Index in nearby_src_table", "Index in XmmDR11", "Index in Xmm2Athena"]
        data = [[n for n in range(nbr_src)], index_in_xd11, index_in_x2a]
        index_table = Table(data=data,
                            names=col_name)

        log_nh, col_photon_index, tup_data = [], [], []
        
        for number in range(nbr_src):
            if index_table["Index in Xmm2Athena"][number] != message:
                log_nh.append(self.xmm_2_athena["logNH_med"][number])
                col_photon_index.append(self.xmm_2_athena["PhoIndex_med"][number])
            else:
                log_nh.append(0.0)
                data, tuple_data = self.optimization_photon_index(number, nearby_src_table)
                tup_data.append(tuple_data)
                col_photon_index.append(data)

        col_nh = [np.exp(value * np.log(10)) if value != 0.0 else 3e20 for value in log_nh]

        col_names = ["Photon Index", "Nh"]
        col_data = [col_photon_index, col_nh]
        
        for name, data in zip(col_names, col_data):
            nearby_src_table[name] = data

        self.visualization_interpolation(tup_data)

        return nearby_src_table, index_table


class Chandra:
    """
    A class for working with Chandra X-ray Observatory data and analyzing sources.

    Attributes:
        catalog (astropy.table.Table): Catalog of X-ray sources from Chandra data.
        nearby_src_table (astropy.table.Table): Table of nearby sources around the target object.
        nearby_src_position (astropy.coordinates.SkyCoord): Sky coordinates of nearby sources.
        cone_search_catalog (vo.dal.SCSService): Cone search catalog from Chandra data.
        cone_search_src_position (astropy.coordinates.SkyCoord): Sky coordinates of sources in the cone search catalog.

    Methods:
        __init__(self, catalog_path, radius, dictionary):
            Initializes a Chandra object with the provided catalog, radius, and object information.

        open_catalog(self, catalog_path):
            Opens and returns an astropy table from the specified catalog path.

        cone_search_catalog(self, radius, dictionary):
            Performs a cone search to retrieve a catalog of sources around the target object.

        find_nearby_src(self, radius, dictionary):
            Finds nearby sources within a specified radius around the target object.

        neighbourhood_of_object(self, radius, dictionary):
            Visualizes the neighborhood of the target object, comparing Chandra data and cone search results.

        find_gamma(self, number):
            Determines the photon index for a specific source in multiple energy bands.

        optimization_visualization(self, opti_data):
            Visualizes the interpolation of photon indices for multiple sources.

        powlaw_gamma(self):
            Gets the photon indices for all sources and visualizes them.

        model_dictionary(self, gamma):
            Creates a dictionary of source models and associated parameters for all sources.
    """
    
    def __init__(self, catalog_path, radius, dictionary):
        """
        Initializes a Chandra object with the provided parameters.

        Parameters:
        - catalog_path (str): The file path to the Chandra X-ray source catalog.
        - radius (float): The search radius (in degrees) for nearby sources.
        - dictionary (dict): A dictionary containing information about the target object.

        The method performs the following actions:
        1. Opens and loads the Chandra catalog from the specified catalog path.
        2. Finds nearby sources within the specified radius around the target object.
        3. Performs a cone search to retrieve a catalog of sources within the same radius.
        4. Creates SkyCoord objects for the source positions in the cone search catalog.
        5. Visualizes the neighborhood of the target object.

        Returns:
        None
        """
        self.catalog = self.open_catalog(catalog_path=catalog_path)
        self.nearby_src_table, self.nearby_src_position = self.find_nearby_src(radius=radius, dictionary=dictionary)

        # -------------------------------------------------- #
        
                        # data with cone search
        self.cone_search_catalog = self.cone_search_catalog(radius=radius, dictionary=dictionary)
        self.cone_search_src_position = SkyCoord(ra=list(self.cone_search_catalog['ra']), dec=list(self.cone_search_catalog['dec']), unit=u.deg)
        
        # -------------------------------------------------- #
        
        self.neighbourhood_of_object(radius=radius, dictionary=dictionary)
        
        
    def open_catalog(self, catalog_path):
        """
        Opens and loads a Chandra catalog from the specified catalog file.

        Parameters:
        - catalog_path (str): The file path to the Chandra X-ray source catalog.

        Returns:
        astropy.table.Table: An astropy table containing the catalog data.
        """
        with fits.open(catalog_path) as data:
            self.result_table = Table(data[1].data)
            data.close()
            return self.result_table


    def cone_search_catalog(self, radius, dictionary):
        """
        Performs a cone search to retrieve a catalog of sources around the target object.

        Parameters:
        - radius (float): The search radius (in degrees) for the cone search.
        - dictionary (dict): A dictionary containing information about the target object, including its name.

        Returns:
        vo.dal.SCSResults: The cone search catalog containing sources within the specified radius.
        """
        self.cone = vo.dal.SCSService('http://cda.cfa.harvard.edu/csc2scs/coneSearch') 
        self.name = SkyCoord.from_name(dictionary['object_name'])
        self.cone_search_catalog = self.cone.search(pos=self.name, radius=radius, verbosity=3)
        return self.cone_search_catalog


    def find_nearby_src(self, radius, dictionary):
        """
        Finds nearby sources within the specified radius around the target object.

        Parameters:
        - radius (float): Search radius (in degrees) for finding nearby sources.
        - dictionary (dict): A dictionary containing information about the target object.

        Returns:
        - astropy.table.Table: Table of nearby sources.
        - astropy.coordinates.SkyCoord: Sky coordinates of nearby sources.

        This method searches for sources in the Chandra catalog that are located within
        a given radius of the target object's position. It calculates the minimum and
        maximum right ascension and declination values to define the search area. The
        sources within this area are added to a table, and their positions are stored as
        SkyCoord objects. The method also checks if any nearby sources are found and
        prints a message accordingly. The nearby sources table and their positions are
        returned.
        """
        field_of_view = radius + 5*u.arcmin

        object_position = dictionary['object_position']

        min_ra, max_ra = object_position.ra - field_of_view, object_position.ra + field_of_view
        min_dec, max_dec = object_position.dec - field_of_view, object_position.dec + field_of_view

        table = Table(names=self.catalog.colnames,
                      dtype=self.catalog.dtype)

        self.result_table = Table(names=self.catalog.colnames,
                                  dtype=self.catalog.dtype)

        for number in range(len(self.catalog)):
            if min_ra/u.deg < self.catalog["RA"][number] < max_ra/u.deg and min_dec/u.deg < self.catalog["DEC"][number] < max_dec/u.deg:
                table.add_row(self.catalog[number])
                
        src_position = SkyCoord(ra=table['RA'], dec=table['DEC'], unit=u.deg)
                
        for number in range(len(table)):
            if F.ang_separation(object_position, src_position[number]) < radius:
                self.result_table.add_row(table[number])
                
        column_names = ['Chandra_IAUNAME', 'RA', 'DEC', 'flux_aper_b', 'flux_aper_s', 'flux_aper_m', 'flux_aper_h']       
        self.nearby_src_table = F.sources_to_unique_sources(result_table=self.result_table, column_names=column_names)
                
        self.nearby_src_position = SkyCoord(ra=self.nearby_src_table['RA'], dec=self.nearby_src_table['DEC'], unit=u.deg)
                
        try :
            if len(self.nearby_src_table) != 0:
                print((f"We have detected {len(self.nearby_src_table)} sources close to {dictionary['object_name']}"))
                return self.nearby_src_table, self.nearby_src_position
            else:
                print(f"No sources detected close to {dictionary['object_name']}.")
                sys.exit()
        except Exception as error:
            print(f"An error occured : {error}")
            
    
    def neighbourhood_of_object(self, radius, dictionary):
        """
        Visualizes the neighborhood of the target object with sources from Chandra data and cone search.

        Parameters:
        - radius (float): The search radius (in degrees) for finding nearby sources.
        - dictionary (dict): A dictionary containing information about the target object.

        This method creates a visualization of the neighborhood of the target object by
        plotting sources from Chandra data (retrieved from the nearby source table) and
        sources obtained from a cone search. The positions of the sources are plotted
        on a 2D scatter plot, and the target object's position is marked with a red 'x.'
        The left subplot displays sources from Chandra data, while the right subplot shows
        sources from the cone search catalog. The number of sources found in each catalog
        is also displayed in the legend.

        Returns:
        None
        """
        cs_csc_ra = np.array(list(self.cone_search_catalog['ra']), dtype=float)
        cs_csc_dec = np.array(list(self.cone_search_catalog['dec']), dtype=float)
        
        csc_ra = np.array(self.nearby_src_table['RA'], dtype=float)
        csc_dec = np.array(self.nearby_src_table['DEC'], dtype=float)
        csc_ra = list(set(csc_ra))
        csc_dec = list(set(csc_dec))
        
        figure_1, axes = plt.subplots(1, 2, figsize=(17, 8))
        figure_1.suptitle(f"Neighbourhood of {dictionary['object_name']}, radius = {radius}", fontsize=20)
        
        ax0 = axes[0]
        ax0.scatter(csc_ra, csc_dec, s=10, c='black', marker="*", label=f"Sources close to {dictionary['object_name']} : {len(csc_ra)}")
        ax0.scatter(dictionary['object_position'].ra, dictionary['object_position'].dec, marker='x', c='red', label=f"{dictionary['object_name']}")
        ax0.legend(loc='upper right')
        ax0.set_xlabel("Right Ascension")
        ax0.set_ylabel("Declination")
        ax0.set_title("With chandra.fits")
        
        ax1 = axes[1]
        ax1.scatter(cs_csc_ra, cs_csc_dec, s=10, c='black', marker="*", label=f"Sources close to {dictionary['object_name']} : {len(cs_csc_ra)}")
        ax1.scatter(dictionary['object_position'].ra, dictionary['object_position'].dec, marker='x', c='red', label=f"{dictionary['object_name']}")
        ax1.legend(loc='upper right')
        ax1.set_xlabel("Right Ascension")
        ax1.set_ylabel("Declination")
        ax1.set_title("With cone search")
        

    def find_gamma(self, number):
        """
        Determine the photon index for a specific source in multiple energy bands.

        Parameters:
        - number (int): The index of the source in the result table.

        Returns:
        Tuple[float, Tuple[np.ndarray, np.ndarray, float, np.ndarray]]:
            - Photon index for the source.
            - Tuple containing energy band, flux values, photon index, and power law fit.

        This method calculates the photon index for a specific source in multiple energy bands
        by fitting a power law function to the energy band and flux data. If the fit is
        successful, it returns the photon index and a tuple containing the energy band,
        flux values, the photon index, and the power law fit. If the fit fails, it returns
        a default photon index of 2.0.

        """
        def power_law(x, constant, gamma):
            return constant * (x**(-gamma))
        
        column_names = np.array(['flux_aper_s', 'flux_aper_m', 'flux_aper_h'], dtype=str)
        energy_band = np.array([0.7, 1.6, 4.5], dtype=float)
        flux = np.array([self.cone_search_catalog[name][number] for name in column_names])
        flux = np.nan_to_num(flux, nan=0.0)
        try:
            popt, pcov = curve_fit(power_law, energy_band, flux.astype(float))
            constant, photon_index = popt
            function = power_law(energy_band, *popt)
        except RuntimeError as e:
            photon_index = 2.0
            function = [0.0, 0.0, 0.0]

        return photon_index, (energy_band, flux, photon_index, function)


    def optimization_visualization(self, opti_data):
        """
        Visualize interpolation of photon indices for multiple sources.

        Parameters:
        - opti_data (List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]]):
            List of tuples containing energy band, flux values, photon index, and power law fit for sources.

        Returns:
        None

        This method creates a visualization of the interpolation of photon indices for
        multiple sources. It plots energy bands, flux values, and power law fits for each
        source in a grid of subplots. The number of subplots depends on the number of sources
        provided in the opti_data list. Each subplot displays the data and a power law fit
        for one source.

        """
        nbr_of_interpolation = len(opti_data)
        n_col = 4
        n_row = nbr_of_interpolation/n_col

        if n_row < 1:
            n_row = 1
        elif n_row % 1 == 0:
            n_row = int(nbr_of_interpolation/4)
        else:
            n_row = int(nbr_of_interpolation/4) + 1

        fig_2, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(15, 15))
        fig_2.subplots_adjust(wspace=0.5, hspace=2.5)
        fig_2.suptitle("Interpolation Photon Index", fontsize=20)
        count = 0
        for row in range(n_row):
            for col in range(n_col):
                if count < nbr_of_interpolation:
                    energy_ranges = opti_data[count][0]
                    flux_values = opti_data[count][1]
                    photon_index = opti_data[count][2]
                    power_law = opti_data[count][3]

                    axes[row][col].errorbar(energy_ranges, flux_values, fmt='*', color='red', ecolor='black')
                    axes[row][col].plot(energy_ranges, power_law, linestyle='dashdot', color="navy")
                    axes[row][col].set_xlabel('Energy [keV]', fontsize=7)
                    axes[row][col].set_ylabel('Flux [erg/cm^2/s]', fontsize=7)
                    axes[row][col].set_title(f"Photon_Index : {photon_index:.4f}", fontsize=7)

                count += 1
                
        plt.show()


    def powlaw_gamma(self):
        """
        Get the photon indices for all sources and visualize them.

        Returns:
        List[float]: List of photon indices for all sources.

        This method calculates and collects the photon indices for all sources in the Chandra catalog.
        If the catalog already contains valid photon indices, those are used. Otherwise, it calculates
        the photon index using the `find_gamma` method and visualizes the interpolation using the
        `optimization_visualization` method. The method returns a list of photon indices for all sources.
        """
        nbr_src = len(self.cone_search_catalog)
        gamma = []
        opti_data = []

        for item, number in zip(self.cone_search_catalog['powlaw_gamma'], range(nbr_src)):
            if item != 0:
                gamma.append(item)
            else:
                photon_index, tup_data = self.find_gamma(number)
                gamma.append(photon_index)
                opti_data.append(tup_data)

        self.optimization_visualization(opti_data)

        return gamma
    
    
    def model_dictionary(self, gamma):
        """
        Create a dictionary of source models and associated parameters for all sources.

        Parameters:
        - gamma (List[float]): List of photon indices for sources.

        Returns:
        Dict[str, Dict[str, Union[str, float]]:
            Dictionary of source models and associated parameters for all sources.

        This method creates a dictionary that contains source models and their associated parameters
        for all sources in the Chandra catalog. It extracts information such as model type, model value,
        flux, and column density for each source. The provided `gamma` list is used to populate the
        photon indices in the dictionary. The resulting dictionary is returned.
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
                model_value.append(gamma[item])
        
        for item in range(nbr_src):

            dictionary = {
                "model": model[item],
                "model_value": model_value[item],
                "flux": csc_flux[item],
                "column_dentsity": nh_value[item]
            }

            model_dictionary[f"src_{item}"] = dictionary

        return model_dictionary


class Swift():
    """
    A class for working with Swift X-ray catalog data.

    This class provides methods for opening a catalog, finding nearby sources
    within a specified radius, and visualizing the neighborhood of a target object.

    Parameters:
    - catalog_path (str): Path to the Swift catalog FITS file.
    - radius (float): Search radius (in degrees) for finding nearby sources.
    - dictionary (dict): A dictionary containing information about the target object.

    Attributes:
    - catalog (astropy.table.Table): Table containing the Swift catalog data.
    - nearby_src_table (astropy.table.Table): Table containing nearby sources.
    - nearby_src_position (astropy.coordinates.SkyCoord): Sky coordinates of nearby sources.

    Methods:
    1. __init__(self, catalog_path, radius, dictionary)
       Initializes the Swift object by opening the catalog, finding nearby sources, and visualizing the neighborhood.

    2. open_catalog(self, catalog_path)
       Opens the Swift catalog from the specified file.

    3. find_nearby_src(self, radius, dictionary)
       Finds nearby sources within the specified radius around the target object.

    4. neighbourhood_of_object(self, dictionary, radius)
       Visualizes the neighborhood of the target object and nearby sources.

    """

    def __init__(self, catalog_path, radius, dictionary):
        """
        Initializes the Swift object by opening the catalog, finding nearby sources, and visualizing the neighborhood.

        Parameters:
        - catalog_path (str): Path to the Swift catalog FITS file.
        - radius (float): Search radius (in degrees) for finding nearby sources.
        - dictionary (dict): A dictionary containing information about the target object.
        """
        self.catalog = self.open_catalog(catalog_path)
        self.nearby_src_table, self.nearby_src_position = self.find_nearby_src(radius, dictionary)
        self.neighbourhood_of_object(dictionary=dictionary, radius=radius)
        
    
    def open_catalog(self, catalog_path):
        """
        Opens the Swift catalog from the specified file.

        Parameters:
        - catalog_path (str): Path to the Swift catalog FITS file.

        Returns:
        - astropy.table.Table: The opened catalog as an astropy table.
        """
        with fits.open(catalog_path) as data:
            self.result_table = Table(data[1].data)
            data.close()
            return self.result_table
        
        
    def find_nearby_src(self, radius, dictionary):
        """
        Finds nearby sources within the specified radius around the target object.

        Parameters:
        - radius (float): Search radius (in degrees) for finding nearby sources.
        - dictionary (dict): A dictionary containing information about the target object.

        Returns:
        - astropy.table.Table: Table of nearby sources.
        - astropy.coordinates.SkyCoord: Sky coordinates of nearby sources.

        This method searches for sources in the Chandra catalog that are located within
        a given radius of the target object's position. It calculates the minimum and
        maximum right ascension and declination values to define the search area. The
        sources within this area are added to a table, and their positions are stored as
        SkyCoord objects. The method also checks if any nearby sources are found and
        prints a message accordingly. The nearby sources table and their positions are
        returned.
        """
        field_of_view = radius + 5*u.arcmin

        object_position = dictionary['object_position']

        min_ra, max_ra = object_position.ra - field_of_view, object_position.ra + field_of_view
        min_dec, max_dec = object_position.dec - field_of_view, object_position.dec + field_of_view

        table = Table(names=self.catalog.colnames,
                      dtype=self.catalog.dtype)

        self.result_table = Table(names=self.catalog.colnames,
                                  dtype=self.catalog.dtype)

        for number in range(len(self.catalog)):
            if min_ra/u.deg < self.catalog["RA"][number] < max_ra/u.deg and min_dec/u.deg < self.catalog["DEC"][number] < max_dec/u.deg:
                table.add_row(self.catalog[number])
                
        src_position = SkyCoord(ra=table['RA'], dec=table['DEC'], unit=u.deg)
                
        for number in range(len(table)):
            if F.ang_separation(object_position, src_position[number]) < radius:
                self.result_table.add_row(table[number])
                
        column_names = ['Swift_IAUNAME', 'RA', 'DEC', 'Flux', 'Flux1', 'Flux2', 'Flux3']    
        self.nearby_src_table = F.sources_to_unique_sources(result_table=self.result_table, column_names=column_names)
                
        self.nearby_src_position = SkyCoord(ra=self.nearby_src_table['RA'], dec=self.nearby_src_table['DEC'], unit=u.deg)
                
        try :
            if len(self.nearby_src_table) != 0:
                print((f"We have detected {len(self.nearby_src_table)} sources close to {dictionary['object_name']}"))
                return self.nearby_src_table, self.nearby_src_position
            else:
                print(f"No sources detected close to {dictionary['object_name']}.")
                sys.exit()
        except Exception as error:
            print(f"An error occured : {error}")


    def neighbourhood_of_object(self, dictionary, radius):
        """
        Visualizes the neighborhood of the target object and nearby sources.

        Parameters:
        - dictionary (dict): A dictionary containing information about the target object.
        - radius (float): Search radius (in degrees) for finding nearby sources.
        """
        object_position = dictionary['object_position']
    
        swi_ra = self.nearby_src_table['RA']
        swi_dec = self.nearby_src_table['DEC']
        
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


class ERosita():
    """
    A class for working with ERosita X-ray catalog data.

    This class provides methods for opening a catalog, finding nearby sources
    within a specified radius, and visualizing the neighborhood of a target object.

    Parameters:
    - catalog_path (str): Path to the ERosita catalog FITS file.
    - radius (float): Search radius (in degrees) for finding nearby sources.
    - dictionary (dict): A dictionary containing information about the target object.

    Attributes:
    - catalog (astropy.table.Table): Table containing the ERosita catalog data.
    - nearby_src_table (astropy.table.Table): Table containing nearby sources.
    - nearby_src_position (astropy.coordinates.SkyCoord): Sky coordinates of nearby sources.

    Methods:
    1. __init__(self, catalog_path, radius, dictionary)
       Initializes the ERosita object by opening the catalog and finding nearby sources.

    2. open_catalog(self, catalog_path)
       Opens the ERosita catalog from the specified file.

    3. find_nearby_src(self, radius, dictionary)
       Finds nearby sources within the specified radius around the target object.

    4. neighbourhood_of_object(self, dictionary, radius)
       Visualizes the neighborhood of the target object and nearby sources.

    """
    def __init__(self, catalog_path, radius, dictionary):
        """
        Initializes the ERosita object by opening the catalog and finding nearby sources.

        Parameters:
        - catalog_path (str): Path to the ERosita catalog FITS file.
        - radius (float): Search radius (in degrees) for finding nearby sources.
        - dictionary (dict): A dictionary containing information about the target object.
        """
        self.catalog = self.open_catalog(catalog_path)
        self.nearby_src_table, self.nearby_src_position = self.find_nearby_src(radius=radius, dictionary=dictionary)
        
        
    def open_catalog(self, catalog_path):
        """
        Opens the ERosita catalog from the specified file.

        Parameters:
        - catalog_path (str): Path to the ERosita catalog FITS file.

        Returns:
        - astropy.table.Table: The opened catalog as an astropy table.
        """
        with fits.open(catalog_path) as data:
            self.result_table = Table(data[1].data)
            data.close()
            return self.result_table
        
        
    def find_nearby_src(self, radius, dictionary):
        """
        Finds nearby sources within the specified radius around the target object.

        Parameters:
        - radius (float): Search radius (in degrees) for finding nearby sources.
        - dictionary (dict): A dictionary containing information about the target object.

        Returns:
        - astropy.table.Table: Table of nearby sources.
        - astropy.coordinates.SkyCoord: Sky coordinates of nearby sources.
        """
        field_of_view = radius + 5*u.arcmin

        object_position = dictionary['object_position']

        min_ra, max_ra = object_position.ra - field_of_view, object_position.ra + field_of_view
        min_dec, max_dec = object_position.dec - field_of_view, object_position.dec + field_of_view

        table = Table(names=self.catalog.colnames,
                      dtype=self.catalog.dtype)

        self.nearby_src_table = Table(names=self.catalog.colnames,
                                      dtype=self.catalog.dtype)

        for number in range(len(self.catalog)):
            if min_ra/u.deg < self.catalog["RA"][number] < max_ra/u.deg and min_dec/u.deg < self.catalog["DEC"][number] < max_dec/u.deg:
                table.add_row(self.catalog[number])
                
        src_position = SkyCoord(ra=table['RA'], dec=table['DEC'], unit=u.deg)
                
        for number in range(len(table)):
            if F.ang_separation(object_position, src_position[number]) < radius:
                self.nearby_src_table.add_row(table[number])
                
        self.nearby_src_position = SkyCoord(ra=self.nearby_src_table['RA'], dec=self.nearby_src_table['DEC'], unit=u.deg)
                
        try :
            if len(self.nearby_src_table) != 0:
                print((f"We have detected {len(self.nearby_src_table)} sources close to {dictionary['object_name']}"))
                return self.nearby_src_table, self.nearby_src_position
            else:
                print(f"No sources detected close to {dictionary['object_name']}.")
                sys.exit()
        except Exception as error:
            print(f"An error occured : {error}")
    
    
    def neighbourhood_of_object(self, dictionary, radius):
        """
        Visualizes the neighborhood of the target object and nearby sources.

        Parameters:
        - dictionary (dict): A dictionary containing information about the target object.
        - radius (float): Search radius (in degrees) for finding nearby sources.
        """
        object_position = dictionary['object_position']
    
        ero_ra = self.nearby_src_table['RA']
        ero_dec = self.nearby_src_table['DEC']
        
        figure_1, axes = plt.subplots(1, 1, figsize=(12, 8))
        figure_1.suptitle(f"Neighbourhood of {dictionary['object_name']}, radius = {radius}", fontsize=20)
        
        axes.scatter(ero_ra, ero_dec, c='black', s=1, marker='*', label=f"Sources close to {dictionary['object_name']}, nbr_src : {len(ero_ra)}")
        axes.scatter(object_position.ra, object_position.dec, c='red', s=100, marker='+', label=f"{dictionary['object_name']}")
        axes.set_xlabel('Right Ascension')
        axes.set_ylabel('Declination')
        axes.legend(loc='upper right')
        
        plt.show()
            