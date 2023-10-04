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
        - XMM_DR13_path (str): The path to the XMM DR13 catalog file.
        - NICER_parameters_path (str): The path to the file containing NICER parameters.

        Attributes:
        - catalog (astropy.table.Table): The XMM-Newton DR13 catalog data.
        - NICER_parameters (tuple): NICER parameters (EffArea and OffAxisAngle) loaded from a file.
        - NearbySources_Table (astropy.table.Table): Table of nearby sources.

        Methods:
        - open_catalog(XMM_DR13_path): Open and load the XMM-Newton DR13 catalog as an astropy Table.
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
        self.NICER_parameters = self.nicer_parameters(NICER_parameters_path)
        self.NearbySRC_Table = None
        self.Modif_Table = None


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


    def nicer_parameters(self, NICER_parameters_path):
        """
            Loads NICER parameters (EffArea and OffAxisAngle) from a file and returns them.

            This method reads NICER parameters from the specified file and returns them as NumPy arrays.

            :param NICER_parameters_path: The path to the file containing NICER parameters.
            :type NICER_parameters_path: str
            :return: EffArea, OffAxisAngle
            :rtype: numpy.ndarray, numpy.ndarray
        """
        while True: 
            with open(NICER_parameters_path, "r") as data:
                first_row = data.readline()
                column = first_row.split()
                ncol = len(column)
               
            try:
                if ncol == 2:
                    EffArea, OffAxisAngle = np.loadtxt(NICER_parameters_path, unpack=True, usecols=(0, 1))
                    return EffArea, OffAxisAngle
                else:
                    raise Exception(f"The file {NICER_parameters_path} doesn't have 2 columns")
            except Exception as error:
                print(f"An error occured {error}")
                NICER_parameters_path = str(input('Input another file path : \n'))
            

    def NearbySourcesTable(self, Object_data, UserTable, catalog):
        """
            Create a table of nearby sources within a specified field of view around the object's position.

            Parameters:
                Object_data (dict): A dictionary containing object information, including position.
                UserTable (Table): A table containing user-defined source information.
                catalog (Table): A catalog of sources to search for nearby sources.

            Returns:
                Tuple (Table, SkyCoord, int): A tuple containing the NearbySRC_Table, NearbySRCposition, and the number of variable sources.

            This method generates a NearbySRC_Table, containing sources from the provided catalog that fall within a specified field of view
            centered around the object's position. It also includes nearby sources from the UserTable if available.
            
            The NearbySRCposition is a SkyCoord object with the coordinates of the sources in NearbySRC_Table.
            
            Nbr_Var_SRC is the count of variable sources in the NearbySRC_Table.

            Example:
            NearbySRC_Table, NearbySRCposition, Nbr_Var_SRC = NearbySourcesTable(Object_data, UserTable, catalog)
        """ 
        
        fiel_of_view = 15
        
        minRA = Object_data['OBJposition'].ra - fiel_of_view*u.arcmin
        maxRA = Object_data['OBJposition'].ra + fiel_of_view*u.arcmin

        minDEC = Object_data['OBJposition'].dec - fiel_of_view*u.arcmin
        maxDEC = Object_data['OBJposition'].dec + fiel_of_view*u.arcmin
        
        self.Modif_Table = Table(names=catalog.colnames,
                                 dtype=catalog.dtype)
        
        self.NearbySRC_Table = Table(names=catalog.colnames,
                                     dtype=catalog.dtype)

        for number in range(len(catalog)):
            if minRA/u.deg < catalog["SC_RA"][number] < maxRA/u.deg and minDEC/u.deg < catalog["SC_DEC"][number] < maxDEC/u.deg:
                self.Modif_Table.add_row(catalog[number])

        nbr_source = len(self.Modif_Table)
        SRCposition = SkyCoord(ra=self.Modif_Table['SC_RA'], dec=self.Modif_Table['SC_DEC'], unit=u.deg)
        
        Nbr_Var_SRC = 0
        if len(UserTable) != 0:
            for number in range(nbr_source):
                if F.AngSeparation(Object_data["OBJposition"], SRCposition[number]) < 5*u.arcmin:
                    self.NearbySRC_Table.add_row(self.Modif_Table[number])
                    
            User_position = SkyCoord(ra=UserTable["Right Ascension"], dec=UserTable["Declination"], unit=u.deg)
            
            for number in range(len(UserTable)):
                if F.AngSeparation(Object_data["OBJposition"], User_position[number]) < 5*u.arcmin:
                    new_row = [0, UserTable["Name"][number], UserTable["Right Ascension"][number], UserTable["Declination"][number]] + [0] * 28 + [UserTable["Var Value"][number]] + [0]*11 + ['no website URL']
                    self.NearbySRC_Table.add_row(new_row)
                    Nbr_Var_SRC += 1

        else:
            for number in range(nbr_source):
                if F.AngSeparation(Object_data["OBJposition"], SRCposition[number]) < 5*u.arcmin:
                    self.NearbySRC_Table.add_row(self.Modif_Table[number]) 
        
        self.NearbySRCposition = SkyCoord(ra=self.NearbySRC_Table['SC_RA'], dec=self.NearbySRC_Table['SC_DEC'], unit=u.deg)
        
        return self.NearbySRC_Table, self.NearbySRCposition, Nbr_Var_SRC
        
        
    def neighbourhood_of_object(self, NearbySourcesTable, Object_data, VAR_SRC_Table):
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
        self.NearbySources_Table = NearbySourcesTable
        NBR_sources = len(self.NearbySources_Table)
        VAR_RA = [VAR_SRC_Table['RA'][number] for number in range(len(VAR_SRC_Table)) if not VAR_SRC_Table['In_Xmm2Athena'][number]]
        VAR_DEC = [VAR_SRC_Table['DEC'][number] for number in range(len(VAR_SRC_Table)) if not VAR_SRC_Table['In_Xmm2Athena'][number]]
        
        VAR_RA_X2A = [VAR_SRC_Table['RA'][number] for number in range(len(VAR_SRC_Table)) if VAR_SRC_Table['In_Xmm2Athena'][number]]
        VAR_DEC_X2A = [VAR_SRC_Table['DEC'][number] for number in range(len(VAR_SRC_Table)) if VAR_SRC_Table['In_Xmm2Athena'][number]]
        
        fig, axes = plt.subplots(1, 2, figsize=(17, 7))
        fig.suptitle(f"Neighbourhood of {Object_data['ObjectName']}")
        
        ax0, ax1 = axes[0], axes[1]
        
        RA = [self.NearbySources_Table['SC_RA'][RightAsc] for RightAsc in range(NBR_sources) if self.NearbySources_Table['SC_RA'][RightAsc] != Object_data['OBJposition'].ra]
        DEC = [self.NearbySources_Table['SC_DEC'][Decl] for Decl in range(NBR_sources) if self.NearbySources_Table['SC_DEC'][Decl] != Object_data['OBJposition'].dec]        
        
        ax0.invert_xaxis()
        ax0.scatter(RA, DEC, color='black', s=10, label="Sources")
        ax0.scatter(Object_data['OBJposition'].ra, Object_data['OBJposition'].dec, marker='*', s=100, color='red', label=f"{Object_data['ObjectName']}")
        ax0.legend(loc='upper right')
        ax0.set_xlabel("Right Ascension")
        ax0.set_ylabel("Declination")
        ax0.set_title(f"Sources close to {Object_data['ObjectName']}, {NBR_sources} sources")
        
        
        INVAR_RA = [self.NearbySources_Table['SC_RA'][RightAsc] for RightAsc in range(NBR_sources) if self.NearbySources_Table['SC_RA'][RightAsc] not in VAR_RA]
        INVAR_DEC = [self.NearbySources_Table['SC_DEC'][Decl] for Decl in range(NBR_sources) if self.NearbySources_Table['SC_DEC'][Decl] not in VAR_DEC]
        
        ax1.invert_xaxis()
        ax1.scatter(INVAR_RA, INVAR_DEC, color='black', s=10, label=f"Invariable sources : {len(INVAR_RA)}")
        ax1.scatter(VAR_RA, VAR_DEC, color='orange', s=10, label=f"Variable sources : {len(VAR_RA)}")
        ax1.scatter(VAR_RA_X2A, VAR_DEC_X2A, color='blue', s=10, label=f"Variable sources in Xmm2Athena : {len(VAR_RA_X2A)}")
        ax1.scatter(Object_data['OBJposition'].ra, Object_data['OBJposition'].dec, marker='*', s=100, color='red', label=f"{Object_data['ObjectName']}")
        ax1.legend(loc='upper right', ncol=2)
        ax1.set_xlabel("Right Ascension")
        ax1.set_ylabel("Declination")
        ax1.set_title(f"Variable and invariable sources close to {Object_data['ObjectName']} ")
        
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

# todo modif Index
    def add_nh_photon_index(self, NearbySources_Table, User_table):
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

            iter_number = len(NearbySources_Table) - len(User_table)

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
 
            NH = [np.exp(value * np.log(10)) if value != 0.0 else 0.0 for value in Log_Nh] + [0] * len(User_table)
            COLNAMES = ['Photon Index', 'Nh']
            DATA = [Photon_Index + [2]*len(User_table), NH]
            
            for col, data in zip(COLNAMES, DATA):
                NearbySources_Table[col] = data

            return NearbySources_Table, INDEX_ATH