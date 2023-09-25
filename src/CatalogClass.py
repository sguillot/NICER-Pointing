# -------------------------------------------------- #

                    ## PACKAGES ##

# Sys
import sys

# Subprocess
import subprocess

# Numpy
import numpy as np

# ASTROPY
from astropy.io import fits
from astropy.coordinates import Angle, SkyCoord
from astropy import units as u
from astropy.table import Table

# ASTROQUERY
from astroquery.simbad import Simbad

# -------------------------------------------------- #

class XmmCatalog:
    """
    This class represents an XMM Catalog and provides methods for working with it.

    Attributes:
        None

    Methods:
        1. open_catalog(file_name):
           Opens an XMM Catalog file and returns it as an astropy Table.
        
        2. TransformCoord2SkyCoord(catalog):
           Transforms the coordinates of all sources in the catalog to SkyCoord objects.
        
        3. CreateNearbySourceTable(NearbySource, XMM_catalog):
           Creates an astropy Table of all sources close to the observing object.

        4. CountRates(NearbySources_Table):
           Calculates the count rates for every source and adds them to the NearbySources_Table.

        5. NicerParameters(parameters):
           Imports Nicer parameters (EffArea and OffAxisAngle) to the program.

    Usage:
    Instantiate an XmmCatalog object and use its methods to manipulate XMM Catalog data.
    """

    def open_catalog(file_name):
        """
        Opens an XMM Catalog file and returns it as an astropy Table.

        :param file_name: The name of the XMM Catalog file.
        :type file_name: str
        :return: The catalog Table.
        :rtype: astropy.table.Table
        """
        with fits.open(file_name) as data:
            CATALOG = data[1].data
            SRC_Table = Table(CATALOG)
            return SRC_Table                                # Return the catalog Table.


    def TransformCoord2SkyCoord(catalog):
        """
        Transforms the coordinates of all sources in the catalog to SkyCoord objects.

        :param catalog: The catalog containing source coordinates.
        :type catalog: astropy.table.Table
        :return: SkyCoord object containing transformed coordinates.
        :rtype: astropy.coordinates.SkyCoord
        """
        return SkyCoord(ra=catalog['SC_RA'], dec=catalog['SC_DEC'], unit=(u.deg, u.deg))        # Return SkyCoord object.
    

    def CreateNearbySourceTable(NearbySource, XMM_catalog):
        """
        Creates an astropy Table of all sources close to the observing object.

        :param NearbySource: List of nearby sources.
        :type NearbySource: list
        :param XMM_catalog: The XMM Catalog.
        :type XMM_catalog: astropy.table.Table
        :return: NearbySources_Table, SRCposition
        :rtype: astropy.table.Table, astropy.coordinates.SkyCoord
        """
        N_SRC = len(NearbySource)                                               # Return the number of sources.          
        NUMBER = [NearbySource[number][0] for number in range(N_SRC)]           # Return their index in the catalog Xmm.

        NearbySources_Table = Table(names=XMM_catalog.colnames,                 # Creation of the NearbySources Table with column name and data type of Xmm Catalog.
                                    dtype=XMM_catalog.dtype)
        
        for number in NUMBER:
            NearbySources_Table.add_row(XMM_catalog[number])
        
        SRCposition = SkyCoord(ra=NearbySources_Table['SC_RA']*u.degree, dec=NearbySources_Table['SC_DEC']*u.degree)    # Creation of SkyCoord object for all the sources close to the observing object.

        return NearbySources_Table, SRCposition         # Return to the main file the NearbySources_Table and the SRC position.


    def CountRates(NearbySources_Table):
        """
        Calculates the count rates for every source and adds them to the NearbySources_Table.

        :param NearbySources_Table: Table containing nearby sources.
        :type NearbySources_Table: astropy.table.Table
        :return: CountRates, Updated NearbySources_Table
        :rtype: list, astropy.table.Table
        """
        CountRates = []     # Empty list of count rates

        xmmflux = NearbySources_Table['SC_EP_8_FLUX']     # Take all of the column SC_EP_8_FLUX of the NearbySources_Table to calculate the count rates.
        NH = NearbySources_Table['Nh']
        Power_Law = NearbySources_Table['Photon Index']

        for flux, nh, power_law in zip(xmmflux, NH, Power_Law):
            pimms_cmds = "instrument nicer 0.3-10.0\nfrom flux ERGS 0.2-12.0\nmodel galactic nh {}\nmodel power {} 0.0\ngo {}\nexit\n".format(nh, power_law, flux)     # Call Wepimms to calculate the count rates.

            with open('pimms_script.xco', 'w') as file:
                file.write(pimms_cmds)
                file.close()

            result = subprocess.run(['pimms', '@pimms_script.xco'], stdout=subprocess.PIPE).stdout.decode(sys.stdout.encoding)
            count_rate = float(result.split("predicts")[1].split('cps')[0])
            CountRates.append(count_rate)           # Append to the count rates list the data we obtains with the program 

        NearbySources_Table["Count Rates"] = CountRates                  # Return to the main fil the New NearbySources_Table and their Count Rates

        return CountRates, NearbySources_Table


    def NicerParameters(parameters):
        """
        Imports Nicer parameters (EffArea and OffAxisAngle) to the program.

        :param parameters: The file containing Nicer parameters.
        :type parameters: str
        :return: EffArea, OffAxisAngle
        :rtype: numpy.ndarray, numpy.ndarray
        """
        EffAera, OffAxisAngle = np.loadtxt(parameters, unpack=True, usecols=(0, 1))               
        return EffAera, OffAxisAngle                    # Return EffAera and OffAxisAngle to the main file



class Xmm2Athena:
    """
    A class for processing X-ray data from the XMM-Newton and Athena catalogs.

    This class provides methods for working with X-ray source catalogs from XMM-Newton
    (XMM2) and Athena. It includes functionality for merging and processing data from
    these catalogs to create a combined table of nearby sources.

    Methods:
    1. open_catalog(file_name_xmm2ath, file_path_xmmdr11):
        Opens and reads the XMM2 and XMM-DR11 FITS files and returns tables for further
        processing.

    2. NearbySourcesTable_With_X2M_data(NearbySources_Table, XmmDR11_Table, Xmm2Ath_Table):
        Merges nearby sources data from XMM-DR11 with additional information from XMM2
        catalog. It calculates average values and appends them to the table when specific
        data is missing.

    Attributes:
    None

    Example Usage:
    ```
    xmm2_athena = Xmm2Athena()
    xmm2_table, dr11_table = xmm2_athena.open_catalog('xmm2_catalog.fits', 'xmmdr11_catalog.fits')
    combined_table = xmm2_athena.NearbySourcesTable_With_X2M_data(nearby_sources_table, dr11_table, xmm2_table)
    ```

    Note: Make sure to replace 'xmm2_catalog.fits' and 'xmmdr11_catalog.fits' with
    the actual file paths of your FITS files.
    """
    
    def open_catalog(file_name_xmm2ath, file_path_xmmdr11):

        with fits.open(file_name_xmm2ath) as data:
            Xmm2Ath_Table = Table(data[1].data)
        
        with fits.open(file_path_xmmdr11) as data:
            XmmDR11_Table = Table(data[1].data)
        
        return Xmm2Ath_Table, XmmDR11_Table


    def NearbySourcesTable_With_X2M_data(NearbySources_Table, XmmDR11_Table, Xmm2Ath_Table):
        """
        Merge X-ray source data from XMM-DR11 with additional information from XMM2 catalog.

        This function combines data from two X-ray source catalogs, XMM-DR11 and XMM2, to
        create a merged table containing information about nearby sources. It also calculates
        average values for certain parameters and populates the table with the merged data.

        Parameters:
        - NearbySources_Table (astropy.table.Table): A table containing data from nearby sources.
        - XmmDR11_Table (astropy.table.Table): A table containing data from the XMM-DR11 catalog.
        - Xmm2Ath_Table (astropy.table.Table): A table containing data from the XMM2 catalog.

        Returns:
        - NearbySources_Table (astropy.table.Table): A table with merged data and additional
        columns for 'Photon Index', 'Log Nh', and 'Nh'.
        """
        
        message = 'Not founded'
        
        NearbySources_Table_XMMDR11 = Table(names=XmmDR11_Table.colnames,
                                                dtype=XmmDR11_Table.dtype)
            
        Index_DR11 = []

        for item in  NearbySources_Table['IAUNAME']:
            if item in XmmDR11_Table['IAUNAME']:
                index = list(XmmDR11_Table['IAUNAME']).index(item)
                Index_DR11.append(index)
                NearbySources_Table_XMMDR11.add_row(XmmDR11_Table[index])
            else:
                Index_DR11.append(message)

        NearbySource_Table_Xmm2Athena = Table(names=Xmm2Ath_Table.colnames,
                                                dtype=Xmm2Ath_Table.dtype)
            
        for item in NearbySources_Table_XMMDR11["DETID"]:
            if item in Xmm2Ath_Table['DETID']:
                index = list(Xmm2Ath_Table['DETID']).index(item)
                NearbySource_Table_Xmm2Athena.add_row(Xmm2Ath_Table[index])

        average_value_logNH = np.mean(NearbySource_Table_Xmm2Athena['logNH_med'])
        average_value_PhotonIndex = np.mean(NearbySource_Table_Xmm2Athena['PhoIndex_med'])

        Log_Nh, Photon_Index = [], []

        for item in NearbySources_Table_XMMDR11["DETID"]:
            if item in Xmm2Ath_Table["DETID"]:
                index = list(Xmm2Ath_Table['DETID']).index(item)

                Log_Nh.append(Xmm2Ath_Table['logNH_med'][index])
                Photon_Index.append(Xmm2Ath_Table['PhoIndex_med'][index])
            else:
                Log_Nh.append(0)
                Photon_Index.append(0)

        for item in range(len(NearbySources_Table_XMMDR11)):
            if Log_Nh[item] == 0:
                Log_Nh[item] = average_value_logNH
            if Photon_Index[item] == 0:
                Photon_Index[item] = average_value_PhotonIndex

        NH = [np.exp(Nh * np.log(10)) for Nh in Log_Nh]

        COLNAMES = ['Photon Index', 'Log Nh', 'Nh']
        DATA = [Photon_Index, Log_Nh, NH]

        for col, data in zip(COLNAMES, DATA):
            NearbySources_Table[col] = data

        return NearbySources_Table
        
