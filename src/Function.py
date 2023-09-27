                # Python's Module

import numpy as np
from astropy.coordinates import SkyCoord, Angle
from astroquery.simbad import Simbad
from astropy import units as u
import matplotlib.pyplot as plt
import argparse
from astropy.table import Table
import subprocess
import sys
import os

                # Function

                    
def Name_to_Short_Name(NAME):
    """
    Converts a given NAME into a shorter version (ShortName) by removing 'J' characters,
    splitting it based on '+' or '-' symbols, and removing spaces from the first part.

    Args:
    NAME (str): The input name to be converted.

    Returns:
    str: The resulting short name after processing.

    Example:
    If NAME is "John+Doe", this function will return "ohn" as the ShortName.

    Note:
    This function assumes that the input NAME contains at least one of the '+' or '-' symbols.
    """
    Word_wihtout_J = NAME.replace("J", "")
    if '+' in NAME:
        Word_wihtout_symbol = Word_wihtout_J.split('+')
    elif '-' in NAME:
        Word_wihtout_symbol = Word_wihtout_J.split('-')
    ShortName = Word_wihtout_symbol[0].replace(" ", "")
    return ShortName


def GetCoordPSR(name):
    """
    Get the PSR coordinates from the SIMBAD database.

    Parameters:
    name (str): The name of the pulsar object.

    Returns:
    SkyCoord: A SkyCoord object representing the coordinates (RA and DEC) of the pulsar.
    """
    return SkyCoord(ra=Simbad.query_object(name)['RA'][0], dec=Simbad.query_object(name)['DEC'][0], unit=(u.hourangle, u.deg))


                # Object Table

PSRfullnames   = ['PSR J0437-4715','PSR J2124-3358','PSR J0751+1807', 'PSR J1231-1411']
PSRshortname = [Name_to_Short_Name(NAME) for NAME in PSRfullnames]
PSRcoordRA = [GetCoordPSR(NAME).ra for NAME in PSRfullnames]
PSRcoordDEC = [GetCoordPSR(NAME).dec for NAME in PSRfullnames]
PSRcountrates  = [1.319,0.1,0.025, 0.27]
PSRtable = Table([PSRfullnames, PSRshortname, PSRcoordRA, PSRcoordDEC, PSRcountrates],
                 names=('FullName', 'ShortName', 'RA', 'DEC','CountRate'))


def is_valid_file_path(file_path):
    """
    Check if a file exists at the given file path.

    Parameters:
        file_path (str): The file path to be checked.

    Returns:
        bool: True if the file exists, False otherwise.

    Raises:
        None
    """
    try:
        if os.path.exists(file_path):
            return True
        else:
            return False
    except Exception as error:
        print(f"An error occured: {error}")


def get_valid_file_path(PATH):
    """
    Prompt the user for a valid file path until a valid one is provided.

    Parameters:
        PATH (str): The initial file path.

    Returns:
        str: A valid file path that exists.

    Raises:
        None
    """
    value = False
    while not value:
        if is_valid_file_path(PATH):
            print(f"The file at {PATH} is valid.")
            value = True
            return PATH
        else:
            print(f"The file at {PATH} doesn't exist or the path is invalid.")
            PATH = str(input("Enter the file path : \n"))
            value = False


def choose_catalog(Catalog):
    """
    Choose a catalog based on the provided keyword and return a valid file path for it.

    Parameters:
        Catalog (str): The catalog keyword, should be 'DR11' or 'DR13'.

    Returns:
        str: A valid file path for the selected catalog.

    Raises:
        argparse.ArgumentError: If an invalid catalog keyword is provided.
    """
    try:
        if Catalog == 'DR13':
            PATH = 'Catalog/4XMM_slim_DR13cat_v1.0.fits'
            VALID_PATH = get_valid_file_path(PATH)
            return VALID_PATH

        elif Catalog == 'DR11':
            PATH = 'Catalog/4xmmdr11slim_210716.fits'
            VALID_PATH = get_valid_file_path(PATH)
            return VALID_PATH
        else:
            raise argparse.ArgumentError(None, "invalid catalog keyword keyword. retry with DR11 or DR13.")
    except argparse.ArgumentError as error:
        print(f'An error occured : {error}')


def initialization_code():
    """
    Initialize and configure the program based on command line arguments, searching for information about
    a celestial object by its name or coordinates, and selecting a catalog.

    Returns:
        tuple: A tuple containing information about the celestial object:
            - str: The name of the object.
            - str: The coordinates of the object in the format "RA, DEC".
            - float: The count rate of the object.
            - str: A valid file path for the selected catalog.

    Raises:
        None
    """
    parser = argparse.ArgumentParser(description='Search for information with object name or coord')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--name', type=str, help='Replace spaces by _')
    group.add_argument('--coord', type=float, nargs=2, help='ra dec (two float values)')
    group.add_argument('--info', help="Have some information about PSR")
    parser.add_argument("--Catalog", type=str, help="Enter a catalog keyword DR11/DR13.")

    args = parser.parse_args()

    if args.info:
        print(PSRtable)
        return
    
    if args.name:
        NAME = args.name.replace("_", " ")
        print(f"Searching for information with name : {NAME}")
        try:
            OBJposition = GetCoordPSR(NAME)
            print(OBJposition)
        except Exception as error:
            print(f"Error: {NAME} isn't in Simbad Database")
        PATH = choose_catalog(args.Catalog)
    elif args.coord:
        ra, dec = args.coord
        print(f"Searching for information with coordinates : (RA={ra}, DEC={dec})")
        try:
            NAME = Simbad.query_region(f"{ra}d {dec}d", radius="1s")['MAIN_ID'][0]
        except Exception as error:
            print(f"Error: There is no object with these coordinates (RA={ra}, DEC={dec}) in the Simbad Database.")
        OBJposition = GetCoordPSR(NAME)
        PATH = choose_catalog(args.Catalog)

    value = False
    while not value:
        if NAME in PSRfullnames:
            CountRate = PSRtable['CountRate'][PSRtable['FullName'] == NAME][0]
            value = True
        else:
            try:
                CountRate = float(input("Enter the count rate of your object: \n"))
                value = True  
            except ValueError as error:
                print(f"Error: {error}")
                print("Please enter a valid float value for Count Rate.")
                continue
            
    Object_data = {'ObjectName':NAME,
               'ShortName': Name_to_Short_Name(NAME),
               'CountRate': CountRate,
               'PSRposition' : OBJposition}
            
    return Object_data, PATH


def AngSeparation(reference, obj):
    """
    Calculate the angular separation between two celestial objects.

    Parameters:
    reference (SkyCoord): The reference object's coordinates.
    obj (SkyCoord): The coordinates of the object to which the separation is calculated.

    Returns:
    Quantity: The angular separation between the two objects.
    """
    return reference.separation(obj)


def FindNearbySources(catalog, SRCposition, obj_name):
    """
    Find nearby sources close to the observing object within a specified angular range.

    Parameters:
    catalog (list): A list of celestial object coordinates.
    SRCposition (list): A list of source positions.
    obj_name (str): The name of the observing object.

    Returns:
    list: A list of tuples containing the number and coordinates of sources close to the observing object.
    """           
    OBJECTposition = GetCoordPSR(obj_name)
    NUMBER = [n for n in range(len(catalog))]
    return [(number, coord) for (number, coord) in zip(NUMBER, SRCposition) if AngSeparation(OBJECTposition, coord) < 8*u.arcmin]


def ScaledCtRate(D, OptCtRate, effareaX, effareaY):
    """
    Scale a given count rate based on an angular distance and effective area.

    Parameters:
    D (float): The angular distance.
    OptCtRate (float): The original count rate.
    effareaX (array-like): Effective area data points (X values).
    effareaY (array-like): Effective area data points (Y values).

    Returns:
    float: The scaled count rate.
    """
    return OptCtRate * np.interp(D,effareaX,effareaY)


def SignaltoNoise(SrcCtsRate, BkgSrcRates, InstBkgd, ExpTime):
    """
    Calculate the signal-to-noise ratio (S/N) given various parameters.

    Parameters:
    SrcCtsRate (float): Source count rate.
    BkgSrcRates (array-like): Count rates of background sources.
    InstBkgd (float): Instrumental and particle background count rate.
    ExpTime (float): Exposure time.

    Returns:
    float: The signal-to-noise ratio (S/N).
    """
    SNR = (SrcCtsRate*ExpTime) / np.sqrt(ExpTime*(SrcCtsRate+np.sum(BkgSrcRates)+InstBkgd))
    return SNR


def NominalPointingInfo(Simulation_data, Nearby_SRCposition):
    """
    Calculate and print various information related to nominal pointing.

    Parameters:
    SIM_parameters (dict): Dictionary containing simulation parameters.
    Nearby_SRCposition (SkyCoord): Coordinates of nearby sources.

    Returns:
    None
    """

    Object_data = Simulation_data['Object_data']
    Telescop_data = Simulation_data['Telescop_data']
    
    SRCnominalDIST = AngSeparation(Nearby_SRCposition, SkyCoord(ra=Object_data['PSRposition'].ra, dec=Object_data['PSRposition'].dec)).arcmin
    SRCscaleRates = ScaledCtRate(SRCnominalDIST, Simulation_data["NearbySources_Table"]["Count Rates"], Telescop_data["EffArea"], Telescop_data["OffAxisAngle"])
    PSRcountrates = Object_data['CountRate']

    print('PSR S/N at Nominal Pointing ' + str(SignaltoNoise(PSRcountrates, SRCscaleRates, Simulation_data["INSTbkgd"], Simulation_data["EXPtime"])))
    print("PSR count rate at Nominal pointing = " + str(PSRcountrates) + "cts/sec")
    print("BKG sources count rate at Nominal Pointing = " + str(np.sum(SRCscaleRates)) + "cts/sec")
    print("             Individual BKG Sources rates:")
    print(str(SRCscaleRates))
    print("             BKG sources distance from PSR (\')")
    print(SRCnominalDIST)
    print("--------------------------------------------------")
    

def CalculateOptiPoint(Simulation_data, Nearby_SRCposition):
    """
    Calculate the optimal pointing for the NICER telescope to maximize the signal-to-noise ratio (S/N).

    This function calculates the optimal pointing of the NICER telescope by adjusting its position
    to maximize the signal-to-noise ratio (S/N) in the presence of a pulsar source and background sources (SRC).
    It iterates over a grid of angular displacements around the pulsar position and calculates the S/N for
    each possible pointing.

    Args:
        SIM_parameters (dict): A dictionary containing simulation parameters, including the pulsar position,
            pulsar count rates, SRC count rates, effective area, off-axis angle, instrumental background, and exposure time.
        Nearby_SRCposition (SkyCoord): Coordinates of the nearby SRC sources.

    Returns:
        tuple: A tuple containing the following information:
            - OptimalPointingIdx (int): The index of the optimal pointing in the resulting arrays.
            - SRCoptimalSEPAR (float): The angular separation between the optimal pointing and the SRC sources.
            - SRCoptimalRATES (float): The SRC count rate at the optimal pointing.
            - Vector_Dictionary (dict): A dictionary containing result vectors, including sampled RA and DEC positions,
                pulsar count rates, SRC count rates, and the S/N ratio for each pointing.

    Note:
        This function assumes the existence of auxiliary functions such as Angle(), AngSeparation(),
        ScaledCtRate(), and SignaltoNoise().
    """
    Object_data = Simulation_data['Object_data']
    Telescop_data = Simulation_data['Telescop_data']

    DeltaRA = Angle(np.arange(-3.0, 3.1, 0.1), unit=u.deg)/60
    DeltaDEC = Angle(np.arange(-3.0, 3.1, 0.1), unit=u.deg)/60
    
    SampleRA, SampleDEC, SNR, PSRrates, SRCrates = np.zeros((5, len(DeltaRA) * len(DeltaDEC)))
    PSRcountrates = Object_data['CountRate']
    
    count = 0
    for i in DeltaRA:
        for j in DeltaDEC:
                NICERpointing = SkyCoord(ra=Object_data["PSRposition"].ra + i, dec=Object_data["PSRposition"].dec + j)
                PSRseparation = AngSeparation(Object_data["PSRposition"], NICERpointing)
                SRCseparation = AngSeparation(Nearby_SRCposition, NICERpointing)

                PSRcountrateScaled = ScaledCtRate(PSRseparation.arcmin, PSRcountrates, Telescop_data["EffArea"], Telescop_data["OffAxisAngle"])
                SRCcountrateScaled = ScaledCtRate(SRCseparation.arcmin, Simulation_data["NearbySources_Table"]["Count Rates"], Telescop_data["EffArea"], Telescop_data["OffAxisAngle"])

                SampleRA[count] = NICERpointing.ra.deg
                SampleDEC[count] = NICERpointing.dec.deg

                PSRrates[count] = PSRcountrateScaled
                SRCrates[count] = np.sum(SRCcountrateScaled)

                SNR[count] = SignaltoNoise(PSRcountrateScaled, SRCcountrateScaled, Simulation_data["INSTbkgd"], Simulation_data["EXPtime"])
                count = count + 1
    
    OptimalPointingIdx = np.where(SNR==max(SNR))[0][0]
    SRCoptimalSEPAR = AngSeparation(Nearby_SRCposition,SkyCoord(ra=SampleRA[OptimalPointingIdx]*u.degree, dec=SampleDEC[OptimalPointingIdx]*u.degree)).arcmin
    SRCoptimalRATES = ScaledCtRate(SRCoptimalSEPAR,Simulation_data["NearbySources_Table"]["Count Rates"], Telescop_data["EffArea"], Telescop_data["OffAxisAngle"])
    
    Vector_Dictionary = {'SampleRA': SampleRA,
                          'SampleDEC': SampleDEC,
                          'PSRrates': PSRrates,
                          'SRCrates': SRCrates,
                          'SNR': SNR
                          }
    
    return OptimalPointingIdx, SRCoptimalSEPAR, SRCoptimalRATES, Vector_Dictionary


def OptimalPointInfos(Vector_Dictionary, OptimalPointingIdx, SRCoptimalRATES):
    """
    Print information for the optimal NICER pointing that maximizes the signal-to-noise ratio (S/N).

    This function prints information about the S/N ratio, pulsar count rate, background sources count rate,
    individual background source rates, and the optimal pointing coordinates for the NICER telescope.

    Args:
        Vector_Dictionary (dict): A dictionary containing result vectors, including sampled RA and DEC positions,
            pulsar count rates, SRC count rates, and the S/N ratio for each pointing.
        OptimalPointingIdx (int): The index of the optimal pointing in the result vectors.
        SRCoptimalRATES (float): The SRC count rate at the optimal pointing.
    """
    
    # Print info for the optimal NICER pointing that maximizes the S/N ratio
    print ("PSR S/N at Optimal Pointing " + str(Vector_Dictionary['SNR'][OptimalPointingIdx]))
    print ("PSR count rate at Optimal pointing = " + str(Vector_Dictionary["PSRrates"][OptimalPointingIdx]) + " cts/sec")
    print ("BKG sources count rate at Optimal pointing = " + str(Vector_Dictionary["SRCrates"][OptimalPointingIdx]) + " cts/sec")
    print ("     Individual BKG Sources: " )
    print (str(SRCoptimalRATES))
    #print "     Distance from Optimal Pointing (\")"
    #print str(SRCoptimalSEPAR)
    print ("Optimal Pointing:  " + str(Vector_Dictionary["SampleRA"][OptimalPointingIdx]) + "  " + str(Vector_Dictionary["SampleDEC"][OptimalPointingIdx]))
    print ("----------------------------------------------------------------------")



def DataMap(Simulation_data, Vector_Dictionary, OptimalPointingIdx, Nearby_SRCposition):
    """
    Plot the map of the Signal-to-Noise (S/N) ratio as a function of NICER pointing.

    Parameters:
    - SIM_parameters (dict): A dictionary containing simulation parameters, including the pulsar position.
    - Vector_Dictionary (dict): A dictionary containing vector data, including SampleRA, SampleDEC, and SNR.
    - OptimalPointingIdx (int): The index of the optimal pointing in Vector_Dictionary.
    - Nearby_SRCposition (SkyCoord): SkyCoord object representing the positions of nearby sources.

    Returns:
    None
    """
    Object_data = Simulation_data['Object_data']
    
        # Plot the map of S/N ratio as function of NICER pointing
    fig = plt.figure(figsize=(10,6.5))
    ax = fig.add_subplot(111)
    plt.gca().invert_xaxis()
    plt.plot(Nearby_SRCposition.ra, Nearby_SRCposition.dec, marker='.', color='black',linestyle='')    
    plt.plot(Object_data["PSRposition"].ra, Object_data["PSRposition"].dec, marker='*', color='green',linestyle='')
    plt.plot(Vector_Dictionary['SampleRA'][OptimalPointingIdx], Vector_Dictionary['SampleDEC'][OptimalPointingIdx], marker='+', color='red', linestyle='')

    # label of the nearby sources
    plt.scatter(Vector_Dictionary['SampleRA'], Vector_Dictionary['SampleDEC'], c=Vector_Dictionary["SNR"], s=10, edgecolor='face')
    plt.xlabel('RA', fontsize='large')
    plt.ylabel('DEC', fontsize='large')
    plt.title("S/N map for " + Object_data["ObjectName"])
    cbar = plt.colorbar()
    cbar.set_label('S/N')
    plt.show()


def count_rates(Table, xmmflux, NH, Power_Law):
    """
        Calculates the count rates for every source and adds them to the NearbySources_Table.

        :param Modified_NearbySources_Table: Table containing nearby sources.
        :type Modified_NearbySources_Table: astropy.table.Table
        :return: CountRates, Updated NearbySources_Table
        :rtype: list, astropy.table.Table
    """
    CountRates = []

    for flux, nh, power_law in zip(xmmflux, NH, Power_Law):
        pimms_cmds = "instrument nicer 0.3-10.0\nfrom flux ERGS 0.2-12.0\nmodel galactic nh {}\nmodel power {} 0.0\ngo {}\nexit\n".format(nh, power_law, flux)

        with open('pimms_script.xco', 'w') as file:
            file.write(pimms_cmds)
            file.close()

        result = subprocess.run(['pimms', '@pimms_script.xco'], stdout=subprocess.PIPE).stdout.decode(sys.stdout.encoding)
        count_rate = float(result.split("predicts")[1].split('cps')[0])
        CountRates.append(count_rate)

    Table["Count Rates"] = CountRates

    return CountRates, Table