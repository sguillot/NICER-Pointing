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

        for f in xmmflux:
            pimms_cmds = "instrument nicer 0.3-10.0\nfrom flux ERGS 0.2-12.0\nmodel power 2.0 0.0\ngo {}\nexit\n".format(f)     # Call Wepimms to calculate the count rates.
            
            with open('pimms_script.xco', 'w') as f:
                f.write(pimms_cmds)
                f.close()

            result = subprocess.run(['pimms', '@pimms_script.xco'], stdout=subprocess.PIPE).stdout.decode(sys.stdout.encoding)
            count_rate = float(result.split("predicts")[1].split('cps')[0])
            CountRates.append(count_rate)           # Append to the count rates list the data we obtains with the program 

        NearbySources_Table["Count Rates"] = CountRates       # Add the count rates column to the NearbySources_Table
        
        return CountRates, NearbySources_Table                  # Return to the main fil the New NearbySources_Table and their Count Rates


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



class Athena():

    def open_catalog(file_name):
        with fits.open(file_name) as data:
            CATALOG = data[1].data
            SRC_Table = Table(CATALOG)
            return SRC_Table