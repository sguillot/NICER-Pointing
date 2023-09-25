import numpy as np
from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt

import subprocess
import sys

import Function as F

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
        - CountRates (list): List of calculated count rates for nearby sources.

        Methods:
        - open_catalog(XMM_DR13_path): Open and load the XMM-Newton DR13 catalog as an astropy Table.

        - transform_coord_2_skycoord(catalog): Transform source coordinates in the catalog to SkyCoord objects.

        - nicer_parameters(NICER_parameters_path): Load NICER parameters (EffArea and OffAxisAngle) from a file.

        - create_NearbySource_table(NearbySource, XMM_catalog): Create an astropy Table of nearby sources.

        - neighbourhood_of_object(NearbySources_Table, OBJ_dictionary): Plot a neighborhood map for the object.

        - count_rates(Modified_NearbySources_Table): Calculate count rates for nearby sources and update the table.

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
        self.CountRates = []


    def open_catalog(self, XMM_DR13_path):
        """
            Opens an XMM DR13 catalog file and returns it as an astropy Table.

            :param XMM_DR13_path: The path to the XMM DR13 catalog file.
            :type XMM_DR13_path: str
            :return: The catalog Table or None if an error occurs.
            :rtype: astropy.table.Table
        """
        try:
            with fits.open(XMM_DR13_path) as data:
                XMM_catalog = Table(data[1].data)
                return XMM_catalog
        except Exception as err:
            print(f"An error occurred while opening the catalog: {err}")
            return None


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


    def create_NearbySource_table(self, NearbySource, XMM_catalog):
        """
            Creates an astropy Table of all sources close to the observing object.

            :param NearbySource: List of nearby sources.
            :type NearbySource: list
            :param XMM_catalog: The XMM Catalog.
            :type XMM_catalog: astropy.table.Table
            :return: NearbySources_Table, SRCposition
            :rtype: astropy.table.Table, astropy.coordinates.SkyCoord
        """
        N_SRC = len(NearbySource)        
        NUMBER = [NearbySource[number][0] for number in range(N_SRC)]           

        self.NearbySources_Table = Table(names=XMM_catalog.colnames,
                                         dtype=XMM_catalog.dtype)
        
        for number in NUMBER:
            self.NearbySources_Table.add_row(XMM_catalog[number])
        
        Nearby_SRCposition = SkyCoord(ra=self.NearbySources_Table['SC_RA'], dec=self.NearbySources_Table['SC_DEC'], unit=(u.degree, u.degree))

        return self.NearbySources_Table, Nearby_SRCposition
        
        
    def neighbourhood_of_object(NearbySources_Table, OBJ_dictionary):
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
        RA = NearbySources_Table['SC_RA']
        DEC = NearbySources_Table['SC_DEC']
            
        fig, ax = plt.subplots()
        plt.gca().invert_xaxis()

        plt.scatter(RA, DEC, c='white', s=4, marker='x')
        plt.scatter(OBJ_dictionary['PSRposition'].ra, OBJ_dictionary['PSRposition'].dec, c='red', s=10)
        ax.set_facecolor('black')
        plt.title('Nearby sources for ' + OBJ_dictionary['ObjectName'] + ' N_SRC : ' + str(len(NearbySources_Table)))
        plt.xlabel('Right Ascension')
        plt.ylabel('Declination')
        plt.show()     
               
        
    def count_rates(self, Modified_NearbySources_Table):
        """
            Calculates the count rates for every source and adds them to the NearbySources_Table.

            :param Modified_NearbySources_Table: Table containing nearby sources.
            :type Modified_NearbySources_Table: astropy.table.Table
            :return: CountRates, Updated NearbySources_Table
            :rtype: list, astropy.table.Table
        """

        self.NearbySources_Table = Modified_NearbySources_Table
        xmmflux = self.NearbySources_Table['SC_EP_8_FLUX']
        NH = self.NearbySources_Table['Nh']
        Power_Law = self.NearbySources_Table['Photon Index']

        for flux, nh, power_law in zip(xmmflux, NH, Power_Law):
            pimms_cmds = "instrument nicer 0.3-10.0\nfrom flux ERGS 0.2-12.0\nmodel galactic nh {}\nmodel power {} 0.0\ngo {}\nexit\n".format(nh, power_law, flux)

            with open('pimms_script.xco', 'w') as file:
                file.write(pimms_cmds)
                file.close()

            result = subprocess.run(['pimms', '@pimms_script.xco'], stdout=subprocess.PIPE).stdout.decode(sys.stdout.encoding)
            count_rate = float(result.split("predicts")[1].split('cps')[0])
            self.CountRates.append(count_rate)

        self.NearbySources_Table["Count Rates"] = self.CountRates

        return self.CountRates, self.NearbySources_Table
    

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
        try:
            with fits.open(XMM_DR11_path) as data1, fits.open(XMM_2_Athena_path) as data2:
                cat_XMM_DR11 = Table(data1[1].data)
                cat_XMM_2_ATHENA = Table(data2[1].data)
            return cat_XMM_DR11, cat_XMM_2_ATHENA
        except Exception as err:
            print(f"An error occurred while opening the catalog: {err}")
            return None, None


def add_nh_photo_index(self, NearbySources_Table):
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

        for item in NearbySources_Table['IAUNAME']:
            if item in self.XMM_DR11['IAUNAME']:
                index = list(self.XMM_DR11['IAUNAME']).index(item)
                self.NearbySources_Table_DR11.add_row(self.XMM_DR11[index])
                
        for item in self.NearbySources_Table_DR11['DETID']:
            if item in self.XMM_2_ATHENA['DETID']:
                index = list(self.XMM_2_ATHENA['DETID']).index(item)
                self.NearbySources_Table_X2A.add_row(self.XMM_2_ATHENA[index])
                
        self.average_logNH_value = np.mean(self.NearbySources_Table_X2A['logNH_med'])
        self.average_PhotonIndex_value = np.mean(self.NearbySources_Table_X2A['PhoIndex_med'])
        
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
                
        # NH = [np.exp(Nh * np.log(10)) for Nh in Log_Nh]
        NH = []
        for value, number in zip(Log_Nh, range(len(Log_Nh))):
            if Log_Nh[number] != 0.0:
                NH.append(np.exp(value * np.log(10)))
            else:
                NH.append(0.0)

        COLNAMES = ['Photon Index', 'Nh']
        DATA = [Photon_Index, NH]
        
        for col, data in zip(COLNAMES, DATA):
            NearbySources_Table[col] = data
            
        return NearbySources_Table