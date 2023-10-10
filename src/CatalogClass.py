import numpy as np
from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import Function as F
from scipy.optimize import curve_fit

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

    def __init__(self, XMM_DR13_PATH, NICER_PARAMETERS_PATH):
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
        self.catalog = self.open_catalog(XMM_DR13_PATH)
        self.NICER_PARAMETERS = self.nicer_parameters(NICER_PARAMETERS_PATH)
        
        self.nearby_src_table = None
        self.modif_table = None


    def open_catalog(self, XMM_DR13_PATH):
        """
            Opens an XMM DR13 catalog file and returns it as an astropy Table.

            :param XMM_DR13_PATH: The path to the XMM DR13 catalog file.
            :type XMM_DR13_PATH: str
            :return: The catalog Table or None if an error occurs.
            :rtype: astropy.table.Table
        """
        with fits.open(XMM_DR13_PATH) as data:
            XMM_catalog = Table(data[1].data)
            data.close()
        
        return XMM_catalog


    def nicer_parameters(self, NICER_PARAMETERS_PATH):
        """
            Loads NICER parameters (EffArea and OffAxisAngle) from a file and returns them.

            This method reads NICER parameters from the specified file and returns them as NumPy arrays.

            :param NICER_PARAMETERS_PATH: The path to the file containing NICER parameters.
            :type NICER_PARAMETERS_PATH: str
            :return: EffArea, OffAxisAngle
            :rtype: numpy.ndarray, numpy.ndarray
        """
        while True: 
            with open(NICER_PARAMETERS_PATH, "r") as data:
                first_row = data.readline()
                column = first_row.split()
                ncol = len(column)
               
            try:
                if ncol == 2:
                    EffArea, OffAxisAngle = np.loadtxt(NICER_PARAMETERS_PATH, unpack=True, usecols=(0, 1))
                    return EffArea, OffAxisAngle
                else:
                    raise Exception(f"The file {NICER_PARAMETERS_PATH} doesn't have 2 columns")
            except Exception as error:
                print(f"An error occured {error}")
                NICER_PARAMETERS_PATH = str(input('Input another file path : \n'))
            

    def nearby_sources_table(self, object_data, user_table, catalog):
        """        
            Create a table of nearby sources within a specified field of view around the object's position.

            Parameters:
                object_data (dict): A dictionary containing object information, including position.
                user_table (Table): A table containing user-defined source information.
                catalog (Table): A catalog of sources to search for nearby sources.

            Returns:
                Tuple (Table, SkyCoord, int): A tuple containing the nearby_src_table, nearby_src_position, and the number of variable sources.

            This method generates a nearby_src_table, containing sources from the provided catalog that fall within a specified field of view
            centered around the object's position. It also includes nearby sources from the UserTable if available.

            Parameters:
                - object_data (dict): A dictionary containing object information, including position.
                - user_table (Table): A table containing user-defined source information.
                - catalog (Table): A catalog of sources to search for nearby sources.

            Returns:
                - Tuple (Table, SkyCoord, int): A tuple containing the nearby_src_table, nearby_src_position, and the number of variable sources.

            This method generates a nearby_src_table, which is a table of sources from the provided catalog that fall within a specified field of view
            centered around the object's position. It also includes nearby sources from the UserTable if available.

            The nearby_src_position is a SkyCoord object with the coordinates of the sources in nearby_src_table.

            nbr_var_src is the count of variable sources in the nearby_src_table.

            Args:
                object_data (dict): A dictionary containing object information, including position.
                user_table (Table): A table containing user-defined source information.
                catalog (Table): A catalog of sources to search for nearby sources.

            Returns:
                Tuple (Table, SkyCoord, int): A tuple containing the NearbySRC_Table, nearby_src_position, and the number of variable sources.

            Raises:
                ValueError: If the input tables or data are not in the expected format.
        """
        fiel_of_view = 15
        
        minRA = object_data['OBJposition'].ra - fiel_of_view*u.arcmin
        maxRA = object_data['OBJposition'].ra + fiel_of_view*u.arcmin

        minDEC = object_data['OBJposition'].dec - fiel_of_view*u.arcmin
        maxDEC = object_data['OBJposition'].dec + fiel_of_view*u.arcmin
        
        self.modif_table = Table(names=catalog.colnames,
                                 dtype=catalog.dtype)
        
        self.nearby_src_table = Table(names=catalog.colnames,
                                      dtype=catalog.dtype)

        for number in range(len(catalog)):
            if minRA/u.deg < catalog["SC_RA"][number] < maxRA/u.deg and minDEC/u.deg < catalog["SC_DEC"][number] < maxDEC/u.deg:
                self.modif_table.add_row(catalog[number])

        nbr_source = len(self.modif_table)
        SRCposition = SkyCoord(ra=self.modif_table['SC_RA'], dec=self.modif_table['SC_DEC'], unit=u.deg)
        
        nbr_var_src = 0
        if len(user_table) != 0:
            for number in range(nbr_source):
                if F.ang_separation(object_data["OBJposition"], SRCposition[number]) < 5*u.arcmin:
                    self.nearby_src_table.add_row(self.modif_table[number])
                    
            User_position = SkyCoord(ra=user_table["Right Ascension"], dec=user_table["Declination"], unit=u.deg)
            
            for number in range(len(user_table)):
                if F.ang_separation(object_data["OBJposition"], User_position[number]) < 5*u.arcmin:
                    new_row = [0, user_table["Name"][number], user_table["Right Ascension"][number], user_table["Declination"][number]] + [0] * 28 + [user_table["Var Value"][number]] + [0]*11 + ['no website URL']
                    self.nearby_src_table.add_row(new_row)
                    nbr_var_src += 1

        else:
            for number in range(nbr_source):
                if F.ang_separation(object_data["OBJposition"], SRCposition[number]) < 5*u.arcmin:
                    self.nearby_src_table.add_row(self.modif_table[number]) 
        
        self.nearby_src_position = SkyCoord(ra=self.nearby_src_table['SC_RA'], dec=self.nearby_src_table['SC_DEC'], unit=u.deg)
        
        return self.nearby_src_table, self.nearby_src_position, nbr_var_src

    
    def neighbourhood_of_object(self, nearby_src_table, variability_table, simulation_data):
        
        self.nearby_src_table = nearby_src_table
        
        nbr_src_var = len(variability_table)
        nbr_src = len(self.nearby_src_table)
        obj_ra = simulation_data["Object_data"]["OBJposition"].ra/u.deg
        obj_dec = simulation_data["Object_data"]["OBJposition"].dec/u.deg
            
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
        fig.suptitle(f"Neighbourhood of {simulation_data['Object_data']['ObjectName']}", fontsize=20)
        
        ax0 = axes[0]
        ax0.invert_xaxis()
        ax0.scatter(list(self.nearby_src_table["SC_RA"]), list(self.nearby_src_table["SC_DEC"]), color='black', s=10, label="Nearby sources")
        ax0.scatter(obj_ra, obj_dec, color='red', marker='x', s=50, label=f"Position of {simulation_data['Object_data']['ObjectName']}")
        ax0.legend(loc="lower right", ncol=2)
        ax0.set_xlabel("Right Ascension")
        ax0.set_ylabel("Declination")
        ax0.set_title(f"Sources close to {simulation_data['Object_data']['ObjectName']}, {nbr_src} sources")
        
        ax1 = axes[1]
        ax1.invert_xaxis()
        ax1.scatter(invar_sc_ra, invar_sc_dec, color='black', s=10, label=f"Invariant sources, {len(invar_sc_ra)}")
        ax1.scatter(obj_ra, obj_dec, color='red', marker='x', s=50, label=f"Position of {simulation_data['Object_data']['ObjectName']}")
        ax1.scatter(sc_ra_in_x2a, sc_dec_in_x2a, color='darkorange', marker="*", label=f"Variable sources in X2A, {len(sc_ra_in_x2a)}")
        ax1.scatter(sc_ra_in_dr11, sc_dec_in_dr11, color='royalblue', marker="+", label=f"Variable sources not in X2A, {len(sc_ra_in_dr11)}")
        ax1.legend(loc="lower right", ncol=2)
        ax1.set_xlabel("Right Ascension")
        ax1.set_ylabel("Declination")
        ax1.set_title(f"Variable and invariable sources close to {simulation_data['Object_data']['ObjectName']} ")
        
        plt.show()


class Xmm2Athena:
    """
        A class for processing and analyzing XMM-Newton (XMM) DR11 and XMM-Newton to Athena (X2A) catalog data.
        
        This class provides methods for opening XMM-Newton DR11 and XMM-Newton to Athena catalog files,
        as well as adding columns for average logNH and PhotonIndex values to a given nearby sources table.

        Parameters:
        - XMM_DR11_path (str): The file path to the XMM-Newton DR11 catalog.
        - XMM_2_Athena_path (str): The file path to the XMM-Newton to Athena catalog.

        Attributes:
        - XMM_DR11 (astropy.table.Table): The XMM-Newton DR11 catalog data.
        - XMM_2_ATHENA (astropy.table.Table): The XMM-Newton to Athena catalog data.
        - NearbySources_Table_DR11 (astropy.table.Table): The nearby sources table based on XMM-Newton DR11 data.
        - NearbySources_Table_X2A (astropy.table.Table): The nearby sources table based on XMM-Newton to Athena data.
        - average_logNH_value (float): The average logNH value calculated from XMM-Newton to Athena data.
        - average_PhotonIndex_value (float): The average PhotonIndex value calculated from XMM-Newton to Athena data.

        Methods:
        - open_catalog(XMM_DR11_path, XMM_2_Athena_path): Open and load XMM-Newton DR11 and XMM-Newton to Athena catalog data.

        - add_nh_photo_index(NearbySources_Table): Add columns for logNH and PhotonIndex to the nearby sources table
        based on matching DETID values with XMM-Newton to Athena data.

        Returns:
        astropy.table.Table: The nearby sources table with additional columns for logNH and PhotonIndex.
    """
    
    
    def __init__(self, XMM_DR11_PATH, XMM_2_ATHENA_PATH):
        self.XMM_DR11, self.XMM_2_ATHENA = self.open_catalog(XMM_DR11_PATH, XMM_2_ATHENA_PATH)

        self.nearby_src_table_dr11 = None
        self.nearby_src_table_x2a = None


    def open_catalog(self, XMM_DR11_PATH, XMM_2_ATHENA_PATH):
        """
            Open and load XMM-Newton DR11 and XMM-Newton to Athena catalog data.

            Parameters:
            - XMM_DR11_PATH (str): The file path to the XMM-Newton DR11 catalog.
            - XMM_2_ATHENA_PATH (str): The file path to the XMM-Newton to Athena catalog.

            Returns:
            Tuple[astropy.table.Table, astropy.table.Table]: A tuple containing the loaded XMM-Newton DR11 and
            XMM-Newton to Athena catalog data tables.
        """
        with fits.open(XMM_DR11_PATH) as data1, fits.open(XMM_2_ATHENA_PATH) as data2:
            xmm_dr11 = Table(data1[1].data)
            xmm_2_athena = Table(data2[1].data)
            data1.close()
            data2.close()
            
        return xmm_dr11, xmm_2_athena


    def optimization_photon_index(self, number, nearby_src_table):
    
        def power_law(x, constant, photon_index):
            return constant * (x ** photon_index)
        
        col_names = ["SC_EP_1_FLUX", "SC_EP_2_FLUX", "SC_EP_3_FLUX", "SC_EP_4_FLUX", "SC_EP_5_FLUX"]
        
        for name in col_names:
            y_data = [nearby_src_table[name][number] for name in col_names]
            x_data = np.arange(1, len(y_data) + 1)

            popt, pcov = curve_fit(power_law, x_data, y_data)
            constant, photon_index = popt
            
        return photon_index, (x_data, y_data, power_law(x_data, *popt), popt)
        

    def visualization_interpolation(self, tup_data):
        
        nbr_of_interpolation = len(tup_data)
        n_col = 4
        n_row = nbr_of_interpolation/4

        if n_row < 1:
            n_row = 1
        elif n_row % 1 == 0:
            n_row = int(nbr_of_interpolation/4)
        else:
            n_row = int(nbr_of_interpolation/4) + 1
        
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(17, 8))
        fig.subplots_adjust(wspace=0.2, hspace=0.75)

        x_labels = ["", "EP_1_FLUX", "EP_2_FLUX", "EP_3_FLUX", "EP_4_FLUX", "EP_5_FLUX", ""]
        
        count = 0
        for row in range(n_row):
            for col in range(n_col):
                if count < nbr_of_interpolation:
                    x_data = tup_data[count][0]
                    y_data = tup_data[count][1]
                    power_law = tup_data[count][2]
                    
                    constant, photon_index = tup_data[count][3]
                    
                    axes[row][col].plot(x_data, power_law, ls="-.", color="navy")
                    axes[row][col].scatter(x_data, y_data, s=30, color='red', marker="+")
                    axes[row][col].set_title(f"Gamma : {photon_index}", fontsize=7)
                    axes[row][col].set_xticks(np.arange(len(x_labels)))
                    axes[row][col].set_xticklabels(x_labels, fontsize=6)
                    
                count += 1


    def add_nh_photon_index(self, nearby_src_table, user_table):
        user_table = []
        nbr_src = len(nearby_src_table)
        
        name_list = nearby_src_table['IAUNAME']
        
        self.xmm_dr11_table = Table(names=self.XMM_DR11.colnames,
                                    dtype=self.XMM_DR11.dtype)
        
        index_in_xd11 = []
        for name in name_list:
            if name in self.XMM_DR11['IAUNAME']:
                index = list(self.XMM_DR11['IAUNAME']).index(name)
                index_in_xd11.append(index)
                self.xmm_dr11_table.add_row(self.XMM_DR11[index])
            else:
                print(f"{name} is missing")

        index_in_x2a = []
        message = "No data founded"
        for det_id in self.xmm_dr11_table["DETID"]:
            if det_id in self.XMM_2_ATHENA["DETID"]:
                index = list(self.XMM_2_ATHENA["DETID"]).index(det_id)
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
                log_nh.append(self.XMM_2_ATHENA["logNH_med"][number])
                col_photon_index.append(self.XMM_2_ATHENA["PhoIndex_med"][number])
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