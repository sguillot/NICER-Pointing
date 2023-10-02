import numpy as np
from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt

class XmmCatalog:
    """
        A class for processing and analyzing XMM-Newton (XMM) DR13 catalog data and related NICER parameters.

        This class provides methods for opening the XMM-Newton DR13 catalog, transforming source coordinates to SkyCoord
        objects, loading NICER parameters from a file, creating a table of nearby sources, plotting a neighborhood map,
        and calculating count rates for each source in the catalog.

        Parameters:
        - XMM_DR13_path (str): The path to the XMM DR13 catalog file.
        - NICER_parameters_path (str): The path to the file containing NICER parameters.

        Attributes:
        - catalog (astropy.table.Table): The XMM-Newton DR13 catalog data.
        - SRCcoord (astropy.coordinates.SkyCoord): SkyCoord object containing transformed source coordinates.
        - NICER_parameters (tuple): NICER parameters (EffArea and OffAxisAngle) loaded from a file.
        - NearbySources_Table (astropy.table.Table): Table of nearby sources.

        Methods:
        - open_catalog(XMM_DR13_path): Open and load the XMM-Newton DR13 catalog as an astropy Table.

        - transform_coord_2_skycoord(catalog): Transform source coordinates in the catalog to SkyCoord objects.

        - nicer_parameters(NICER_parameters_path): Load NICER parameters (EffArea and OffAxisAngle) from a file.

        - create_NearbySource_table(NearbySource, XMM_catalog): Create an astropy Table of nearby sources.

        - neighbourhood_of_object(NearbySources_Table, OBJ_dictionary): Plot a neighborhood map for the object.

        Returns:
        Various data types, depending on the specific method's purpose and return values.
    """


    def __init__(self, XMM_DR13_path, NICER_parameters_path):
        """
            Constructor for XmmCatalog class.

            Initializes an instance of the XmmCatalog class by opening the XMM DR13 catalog file,
            transforming source coordinates to SkyCoord objects, and loading the NICER parameters
            from the specified files.

            :param XMM_DR13_path: The path to the XMM DR13 catalog file.
            :type XMM_DR13_path: str
            :param NICER_parameters_path: The path to the file containing NICER parameters.
            :type NICER_parameters_path: str
        """
        self.catalog = self.open_catalog(XMM_DR13_path)
        self.SRCcoord = self.transform_coord_2_skycoord(self.catalog)
        self.NICER_parameters = self.nicer_parameters(NICER_parameters_path)
        self.NearbySources_Table = None


    def open_catalog(self, XMM_DR13_path):
        """
            Opens an XMM DR13 catalog file and returns it as an astropy Table.

            :param XMM_DR13_path: The path to the XMM DR13 catalog file.
            :type XMM_DR13_path: str
            :return: The catalog Table or None if an error occurs.
            :rtype: astropy.table.Table
        """
        with fits.open(XMM_DR13_path) as data:
            XMM_catalog = Table(data[1].data)
            data.close()
            return XMM_catalog


    def transform_coord_2_skycoord(self, catalog):
        """
            Transforms the coordinates of all sources in the catalog to SkyCoord objects.

            :param catalog: The catalog containing source coordinates.
            :type catalog: astropy.table.Table
            :return: SkyCoord object containing transformed coordinates.
            :rtype: astropy.coordinates.SkyCoord
        """
        return SkyCoord(ra=catalog['SC_RA'], dec=catalog['SC_DEC'], unit=(u.deg, u.deg))


    def nicer_parameters(self, NICER_parameters_path):
        """
            Loads NICER parameters (EffArea and OffAxisAngle) from a file and returns them.

            This method reads NICER parameters from the specified file and returns them as NumPy arrays.

            :param NICER_parameters_path: The path to the file containing NICER parameters.
            :type NICER_parameters_path: str
            :return: EffArea, OffAxisAngle
            :rtype: numpy.ndarray, numpy.ndarray
        """
        try:
            EffArea, OffAxisAngle = np.loadtxt(NICER_parameters_path, unpack=True, usecols=(0, 1))
            return EffArea, OffAxisAngle
        except Exception as err:
            print(f"An error occurred while reading NICER parameters: {str(err)}")
            return None, None  


    def create_NearbySource_table(self, NearbySource, XMM_catalog, UserList):
        """
            Creates an astropy Table of all sources close to the observing object.

            :param NearbySource: List of nearby sources.
            :type NearbySource: list
            :param XMM_catalog: The XMM Catalog.
            :type XMM_catalog: astropy.table.Table
            :return: NearbySources_Table, SRCposition
            :rtype: astropy.table.Table, astropy.coordinates.SkyCoord
        """
        N_SRC = len(NearbySource) - len(UserList)
        NUMBER = [NearbySource[number][0] for number in range(N_SRC)]  
          
        self.NearbySources_Table = Table(names=XMM_catalog.colnames,
                                         dtype=XMM_catalog.dtype)
        
        if len(UserList) != 0:
            for number in NUMBER:
                self.NearbySources_Table.add_row(XMM_catalog[number])

            for name, ra, dec, nan in UserList:
                new_row = [0, name, ra, dec] + [0] * 28 + [nan] + [0]*11 + ['no website URL']
                print(len(new_row))
                self.NearbySources_Table.add_row(new_row)

        else:
            for number in NUMBER:
                self.NearbySources_Table.add_row(XMM_catalog[number])
        
        Nearby_SRCposition = SkyCoord(ra=self.NearbySources_Table['SC_RA'], dec=self.NearbySources_Table['SC_DEC'], unit=(u.degree, u.degree))

        return self.NearbySources_Table, Nearby_SRCposition
        
        
    def neighbourhood_of_object(NearbySources_Table, Object_data, VAR_SRC_Table):
        """
            Plot a neighborhood map of nearby sources relative to a specified object.

            This function creates a scatter plot of nearby sources using their right ascension (RA) and
            declination (DEC) coordinates. The object of interest is marked with a red dot, while other sources
            are marked with white 'x' markers. The title of the plot includes the object's name and the count
            of nearby sources.

            :param NearbySources_Table: Table of nearby sources with RA and DEC coordinates.
            :type NearbySources_Table: astropy.table.Table
            :param OBJ_dictionary: Dictionary containing object information, including position and name.
            :type OBJ_dictionary: dict

            This function does not return any values; it displays the plot.

            Example:
            neighbourhood_of_object(NearbySources_Table, {'PSRposition': SkyCoord(...), 'ObjectName': 'ExampleObject'})
        """
        NBR_sources = len(NearbySources_Table)
        VAR_RA = [VAR_SRC_Table['RA'][number] for number in range(len(VAR_SRC_Table)) if not VAR_SRC_Table['In_Xmm2Athena'][number]]
        VAR_DEC = [VAR_SRC_Table['DEC'][number] for number in range(len(VAR_SRC_Table)) if not VAR_SRC_Table['In_Xmm2Athena'][number]]
        
        VAR_RA_X2A = [VAR_SRC_Table['RA'][number] for number in range(len(VAR_SRC_Table)) if VAR_SRC_Table['In_Xmm2Athena'][number]]
        VAR_DEC_X2A = [VAR_SRC_Table['DEC'][number] for number in range(len(VAR_SRC_Table)) if VAR_SRC_Table['In_Xmm2Athena'][number]]
        
        fig, axes = plt.subplots(1, 2, figsize=(17, 7))
        fig.suptitle(f"Neighbourhood of {Object_data['ObjectName']}")
        
        ax0, ax1 = axes[0], axes[1]
        
        RA = [NearbySources_Table['SC_RA'][RightAsc] for RightAsc in range(NBR_sources) if NearbySources_Table['SC_RA'][RightAsc] != Object_data['OBJposition'].ra]
        DEC = [NearbySources_Table['SC_DEC'][Decl] for Decl in range(NBR_sources) if NearbySources_Table['SC_DEC'][Decl] != Object_data['OBJposition'].dec]        
        
        ax0.scatter(RA, DEC, color='black', s=10, label="Sources")
        ax0.scatter(Object_data['OBJposition'].ra, Object_data['OBJposition'].dec, marker='*', s=100, color='red', label=f"{Object_data['ObjectName']}")
        ax0.legend(loc='upper right')
        ax0.set_xlabel("Right Ascension")
        ax0.set_ylabel("Declination")
        ax0.set_title(f"Sources close to {Object_data['ObjectName']}, {NBR_sources} sources")
        
        
        INVAR_RA = [NearbySources_Table['SC_RA'][RightAsc] for RightAsc in range(NBR_sources) if NearbySources_Table['SC_RA'][RightAsc] not in VAR_RA]
        INVAR_DEC = [NearbySources_Table['SC_DEC'][Decl] for Decl in range(NBR_sources) if NearbySources_Table['SC_DEC'][Decl] not in VAR_DEC]
        
        ax1.scatter(INVAR_RA, INVAR_DEC, color='black', s=10, label=f"Invariable sources : {len(INVAR_RA)}")
        ax1.scatter(VAR_RA, VAR_DEC, color='orange', s=10, label=f"Variable sources : {len(VAR_RA)}")
        ax1.scatter(VAR_RA_X2A, VAR_DEC_X2A, color='blue', s=10, label=f"Variable sources in Xmm2Athena : {len(VAR_RA_X2A)}")
        ax1.scatter(Object_data['OBJposition'].ra, Object_data['OBJposition'].dec, marker='*', s=100, color='red', label=f"{Object_data['ObjectName']}")
        ax1.legend(loc='upper right', ncol=2)
        ax1.set_xlabel("Right Ascension")
        ax1.set_ylabel("Declination")
        ax1.set_title(f"Variable and invariable sources close to {Object_data['ObjectName']} ")
        
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
    
    
    def __init__(self, XMM_DR11_path, XMM_2_Athena_path):
        self.XMM_DR11, self.XMM_2_ATHENA = self.open_catalog(XMM_DR11_path, XMM_2_Athena_path)

        self.NearbySources_Table_DR11 = None
        self.NearbySources_Table_X2A = None
        
        self.average_logNH_value = 0
        self.average_PhotonIndex_value = 0 


    def open_catalog(self, XMM_DR11_path, XMM_2_Athena_path):
        """
            Open and load XMM-Newton DR11 and XMM-Newton to Athena catalog data.

            Parameters:
            - XMM_DR11_path (str): The file path to the XMM-Newton DR11 catalog.
            - XMM_2_Athena_path (str): The file path to the XMM-Newton to Athena catalog.

            Returns:
            Tuple[astropy.table.Table, astropy.table.Table]: A tuple containing the loaded XMM-Newton DR11 and
            XMM-Newton to Athena catalog data tables.
        """
        with fits.open(XMM_DR11_path) as data1, fits.open(XMM_2_Athena_path) as data2:
            cat_XMM_DR11 = Table(data1[1].data)
            cat_XMM_2_ATHENA = Table(data2[1].data)
            data1.close()
            data2.close()
        return cat_XMM_DR11, cat_XMM_2_ATHENA


    def add_nh_photon_index(self, NearbySources_Table, UserList):
            """
                Add columns for logNH and PhotonIndex to the nearby sources table
                based on matching DETID values with XMM-Newton to Athena data.

                Parameters:
                - NearbySources_Table (astropy.table.Table): The nearby sources table to which columns
                for logNH and PhotonIndex will be added.

                Returns:
                astropy.table.Table: The modified nearby sources table with additional columns for logNH and PhotonIndex.
            """
            self.NearbySources_Table_DR11 = Table(names=self.XMM_DR11.colnames,
                                                dtype=self.XMM_DR11.dtype)
            self.NearbySources_Table_X2A = Table(names=self.XMM_2_ATHENA.colnames,
                                                dtype=self.XMM_2_ATHENA.dtype)

            iter_number = len(NearbySources_Table) - len(UserList)

            for item in NearbySources_Table['IAUNAME'][:iter_number]:
                if item in self.XMM_DR11['IAUNAME']:
                    index = list(self.XMM_DR11['IAUNAME']).index(item)
                    self.NearbySources_Table_DR11.add_row(self.XMM_DR11[index])

            INDEX_ATH = []
            for item in self.NearbySources_Table_DR11['DETID']:
                if item in self.XMM_2_ATHENA['DETID']:
                    index = list(self.XMM_2_ATHENA['DETID']).index(item)
                    INDEX_ATH.append(index)
                    self.NearbySources_Table_X2A.add_row(self.XMM_2_ATHENA[index])

            Log_Nh, Photon_Index = [], []
            for item in self.NearbySources_Table_DR11['DETID']:
                if item in self.XMM_2_ATHENA['DETID']:
                    index = list(self.XMM_2_ATHENA['DETID']).index(item)

                    Log_Nh.append(self.XMM_2_ATHENA['logNH_med'][index])
                    Photon_Index.append(self.XMM_2_ATHENA['PhoIndex_med'][index])
                else:
                    Log_Nh.append(0.0)
                    Photon_Index.append(0.0)

            for item in range(len(self.NearbySources_Table_DR11)):
                if Photon_Index[item] == 0:
                    Photon_Index[item] = 2.0
 
            NH = [np.exp(value * np.log(10)) if value != 0.0 else 0.0 for value in Log_Nh] + [0] * len(UserList)
            COLNAMES = ['Photon Index', 'Nh']
            DATA = [Photon_Index + [2.0]*len(UserList), NH]
            
            for col, data in zip(COLNAMES, DATA):
                NearbySources_Table[col] = data

            return NearbySources_Table, INDEX_ATH