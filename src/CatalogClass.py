import numpy as np
from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import Function as F

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


    def neighbourhood_of_object(self, nearby_src_table, object_data, var_src_table):
        """
            Plot a neighborhood map of nearby sources relative to a specified object.

            This function creates a scatter plot of nearby sources using their right ascension (RA) and
            declination (DEC) coordinates. The object of interest is marked with a red dot, while other sources
            are marked with white 'x' markers. The title of the plot includes the object's name and the count
            of nearby sources.

            :param nearby_src_table: Table of nearby sources with RA and DEC coordinates.
            :type nearby_src_table: astropy.table.Table
            :param object_data: Dictionary containing object information, including position and name.
            :type object_data: dict

            This function does not return any values; it displays the plot.

            Example:
            neighbourhood_of_object(NearbySources_Table, {'PSRposition': SkyCoord(...), 'ObjectName': 'ExampleObject'})
        """
        self.NearbySources_Table = nearby_src_table
        nbr_sources = len(self.NearbySources_Table)
        var_ra = [var_src_table['RA'][number] for number in range(len(var_src_table)) if not var_src_table['In_Xmm2Athena'][number]]
        var_dec = [var_src_table['DEC'][number] for number in range(len(var_src_table)) if not var_src_table['In_Xmm2Athena'][number]]
        
        var_ra_x2a = [var_src_table['RA'][number] for number in range(len(var_src_table)) if var_src_table['In_Xmm2Athena'][number]]
        var_dec_x2a = [var_src_table['DEC'][number] for number in range(len(var_src_table)) if var_src_table['In_Xmm2Athena'][number]]
        
        fig, axes = plt.subplots(1, 2, figsize=(17, 7))
        fig.suptitle(f"Neighbourhood of {object_data['ObjectName']}")
        
        ax0, ax1 = axes[0], axes[1]
        
        ra = [self.NearbySources_Table['SC_RA'][RightAsc] for RightAsc in range(nbr_sources) if self.NearbySources_Table['SC_RA'][RightAsc] != object_data['OBJposition'].ra]
        dec = [self.NearbySources_Table['SC_DEC'][Decl] for Decl in range(nbr_sources) if self.NearbySources_Table['SC_DEC'][Decl] != object_data['OBJposition'].dec]        
        
        ax0.invert_xaxis()
        ax0.scatter(ra, dec, color='black', s=10, label="Sources")
        ax0.scatter(object_data['OBJposition'].ra, object_data['OBJposition'].dec, marker='*', s=100, color='red', label=f"{object_data['ObjectName']}")
        ax0.legend(loc='upper right')
        ax0.set_xlabel("Right Ascension")
        ax0.set_ylabel("Declination")
        ax0.set_title(f"Sources close to {object_data['ObjectName']}, {nbr_sources} sources")
        
        
        invar_ra = [self.NearbySources_Table['SC_RA'][RightAsc] for RightAsc in range(nbr_sources) if self.NearbySources_Table['SC_RA'][RightAsc] not in var_ra]
        invar_dec = [self.NearbySources_Table['SC_DEC'][Decl] for Decl in range(nbr_sources) if self.NearbySources_Table['SC_DEC'][Decl] not in var_dec]
        
        ax1.invert_xaxis()
        ax1.scatter(invar_ra, invar_dec, color='black', s=10, label=f"Invariable sources : {len(invar_ra)}")
        ax1.scatter(var_ra, var_dec, color='orange', s=10, label=f"Variable sources : {len(var_ra)}")
        ax1.scatter(var_ra_x2a, var_dec_x2a, color='blue', s=10, label=f"Variable sources in Xmm2Athena : {len(var_ra_x2a)}")
        ax1.scatter(object_data['OBJposition'].ra, object_data['OBJposition'].dec, marker='*', s=100, color='red', label=f"{object_data['ObjectName']}")
        ax1.legend(loc='upper right', ncol=2)
        ax1.set_xlabel("Right Ascension")
        ax1.set_ylabel("Declination")
        ax1.set_title(f"Variable and invariable sources close to {object_data['ObjectName']} ")
        
        plt.show()
        
        return fig


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

# todo modif Index
    def add_nh_photon_index(self, nearby_src_table, user_table):
            """
                Add columns for logNH and PhotonIndex to the nearby sources table
                based on matching DETID values with XMM-Newton to Athena data.

                Parameters:
                - nearby_src_table (astropy.table.Table): The nearby sources table to which columns
                for logNH and PhotonIndex will be added.

                Returns:
                astropy.table.Table: The modified nearby sources table with additional columns for logNH and PhotonIndex.
            """
            self.nearby_src_table_dr11 = Table(names=self.XMM_DR11.colnames,
                                               dtype=self.XMM_DR11.dtype)
            self.nearby_src_table_x2a = Table(names=self.XMM_2_ATHENA.colnames,
                                              dtype=self.XMM_2_ATHENA.dtype)

            iter_number = len(nearby_src_table) - len(user_table)

            for item in nearby_src_table['IAUNAME'][:iter_number]:
                if item in self.XMM_DR11['IAUNAME']:
                    index = list(self.XMM_DR11['IAUNAME']).index(item)
                    self.nearby_src_table_dr11.add_row(self.XMM_DR11[index])

            index_ath = []
            for item in self.nearby_src_table_dr11['DETID']:
                if item in self.XMM_2_ATHENA['DETID']:
                    index = list(self.XMM_2_ATHENA['DETID']).index(item)
                    index_ath.append(index)
                    self.nearby_src_table_x2a.add_row(self.XMM_2_ATHENA[index])

            log_nh, photon_index = [], []
            for item in self.nearby_src_table_dr11['DETID']:
                if item in self.XMM_2_ATHENA['DETID']:
                    index = list(self.XMM_2_ATHENA['DETID']).index(item)

                    log_nh.append(self.XMM_2_ATHENA['logNH_med'][index])
                    photon_index.append(self.XMM_2_ATHENA['PhoIndex_med'][index])
                else:
                    log_nh.append(0.0)
                    photon_index.append(0.0)

            for item in range(len(self.nearby_src_table_dr11)):
                if photon_index[item] == 0:
                    photon_index[item] = 2.0
 
            col_nh = [np.exp(value * np.log(10)) if value != 0.0 else 0.0 for value in log_nh] + [0] * len(user_table)
            col_photon_index = photon_index + [2]*len(user_table)
            
            col_names = ['Photon Index', 'Nh']
            col_data = [col_photon_index, col_nh]
            
            for col, data in zip(col_names, col_data):
                nearby_src_table[col] = data

            return nearby_src_table, index_ath