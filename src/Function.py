import numpy as np
from astropy.coordinates import SkyCoord, Angle
from astroquery.simbad import Simbad
from astropy import units as u
import matplotlib.pyplot as plt

                    ## FUNCTION ##

def Choice():
    """
    This function allows the user to calculate the optipoint with the name or coordinates of an object. It continuously prompts the user for input until a valid choice is made.

    Returns:
    Tuple: (NAME, OBJposition, PSRcountrates)
        - NAME (str): Name of the object if 'name' is chosen, or determined based on coordinates if 'coord' is chosen.
        - OBJposition (str): The coordinates of the object in the format 'ra dec'.
        - PSRcountrates (float): The count rate entered by the user.
    """
    while True:
        choice = input('Calculate the optipoint with name/coord of the object? (name/coord): ').lower()

        try:
            if choice == 'name':
                NAME = input("Enter the name of the object: ")
                PSRcountrates = float(input('Enter the count rate: '))
                OBJposition = GetCoordPSR(NAME)
                
                return NAME, OBJposition, PSRcountrates

            elif choice == 'coord':
                COORD = input("Enter the coordinate of your object (ra dec): ")
                PSRcountrates = float(input('Enter the count rate: '))
                ra, dec = map(float, COORD.split())
                NAME = Simbad.query_region(SkyCoord(ra=ra, dec=dec))['MAIN_ID'][0]
                OBJposition = GetCoordPSR(NAME)
                
                return NAME, OBJposition, PSRcountrates

            else:
                print("Invalid choice. Please enter 'name' or 'coord'.")
        
        except ValueError:
            print("Invalid input for count rate. Please enter a numeric value.")
        except Exception as err:
            print(f"An error occurred: {err}")

                    
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
