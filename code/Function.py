import numpy as np
from astropy.coordinates import SkyCoord, Angle
from astroquery.simbad import Simbad
from astropy import units as u
import matplotlib.pyplot as plt

                    ## FUNCTION ##
                    
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
    OBJECTposition = GetCoordPSR(obj_name)                  # Call the GetCoordPSR to obtains the coordinates of the observing object
    NUMBER = [n for n in range(len(catalog))]
    return [(number, coord) for (number, coord) in zip(NUMBER, SRCposition) if AngSeparation(OBJECTposition, coord) < 8*u.arcmin]
            # Return the list of tuples who contains the number and the coordinates of all the sources close to the object in a range of 8 arcmin 


def VisualizeData(Object, NearbySource_Table):
    """
    Visualize data by plotting the positions of nearby sources and the pulsar.

    Parameters:
    Object (str): The name of the pulsar.
    NearbySource_Table (DataFrame): A table containing nearby source information.

    Returns:
    None
    """
    PSRposition = GetCoordPSR(Object)
    N_SRC = len(NearbySource_Table)
    
    fig, ax = plt.subplots()
    plt.gca().invert_xaxis()

    NearbyRA = [NearbySource_Table['SC_RA'][ra] for ra in range(N_SRC)]
    NearbyDEC = [NearbySource_Table['SC_DEC'][dec] for dec in range(N_SRC)]

    PSR_RA = PSRposition.ra/u.deg
    PSR_DEC = PSRposition.dec/u.deg

    plt.scatter(NearbyRA, NearbyDEC, c='black', s=1)
    plt.scatter(PSR_RA, PSR_DEC, c='red', s=10)

    plt.title('Nearby sources for ' + Object + ' N_SRC : ' + str(len(NearbySource_Table)))
    plt.xlabel('Right Ascension')
    plt.ylabel('Declination')
    plt.show()


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


def NominalPointingInfo(SIM_parameters, Nearby_SRCposition):
    """
    Calculate and print various information related to nominal pointing.

    Parameters:
    SIM_parameters (dict): Dictionary containing simulation parameters.
    Nearby_SRCposition (SkyCoord): Coordinates of nearby sources.

    Returns:
    None
    """
    
    SRCnominalDIST = AngSeparation(Nearby_SRCposition, SkyCoord(ra=SIM_parameters['PSRposition'].ra, dec=SIM_parameters['PSRposition'].dec)).arcmin
    SRCscaleRates = ScaledCtRate(SRCnominalDIST, SIM_parameters["NearbySource_Table"]["Count Rates"], SIM_parameters["EffArea"], SIM_parameters["OffAxisAngle"])
    
    PSRcountrates = SIM_parameters['PSRtable']['Count Rate'][SIM_parameters["PSRtable"]['FullName'] == SIM_parameters["Pulsar"]][0]
    
    print('PSR S/N at Nominal Pointing ' + str(SignaltoNoise(PSRcountrates, SRCscaleRates, SIM_parameters["INSTbkgd"], SIM_parameters["EXPtime"])))
    print("PSR count rate at Nominal pointing = " + str(PSRcountrates) + "cts/sec")
    print("BKG sources count rate at Nominal Pointing = " + str(np.sum(SRCscaleRates)) + "cts/sec")
    print("             Individual BKG Sources rates:")
    print(str(SRCscaleRates))
    print("             BKG sources distance from PSR (\')")
    print(SRCnominalDIST)
    print("--------------------------------------------------")
    

def CalculateOptiPoint(SIM_parameters, Nearby_SRCposition):
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
    DeltaRA = Angle(np.arange(-3.0, 3.1, 0.1), unit=u.deg)/60
    DeltaDEC = Angle(np.arange(-3.0, 3.1, 0.1), unit=u.deg)/60
    
    SampleRA, SampleDEC, SNR, PSRrates, SRCrates = np.zeros((5, len(DeltaRA) * len(DeltaDEC)))
    PSRcountrates = SIM_parameters['PSRtable']['Count Rate'][SIM_parameters["PSRtable"]['FullName'] == SIM_parameters["Pulsar"]][0]
    
    count = 0
    for i in DeltaRA:
        for j in DeltaDEC:
                NICERpointing = SkyCoord(ra=SIM_parameters["PSRposition"].ra + i, dec=SIM_parameters["PSRposition"].dec + j)
                PSRseparation = AngSeparation(SIM_parameters["PSRposition"], NICERpointing)
                SRCseparation = AngSeparation(Nearby_SRCposition, NICERpointing)

                PSRcountrateScaled = ScaledCtRate(PSRseparation.arcmin, PSRcountrates, SIM_parameters["EffArea"], SIM_parameters["OffAxisAngle"])
                SRCcountrateScaled = ScaledCtRate(SRCseparation.arcmin, SIM_parameters["NearbySource_Table"]["Count Rates"], SIM_parameters["EffArea"], SIM_parameters["OffAxisAngle"])

                SampleRA[count] = NICERpointing.ra.deg
                SampleDEC[count] = NICERpointing.dec.deg

                PSRrates[count] = PSRcountrateScaled
                SRCrates[count] = np.sum(SRCcountrateScaled)

                SNR[count] = SignaltoNoise(PSRcountrateScaled, SRCcountrateScaled, SIM_parameters["INSTbkgd"], SIM_parameters["EXPtime"])
                count = count + 1
    
    OptimalPointingIdx = np.where(SNR==max(SNR))[0][0]
    SRCoptimalSEPAR = AngSeparation(Nearby_SRCposition,SkyCoord(ra=SampleRA[OptimalPointingIdx]*u.degree, dec=SampleDEC[OptimalPointingIdx]*u.degree)).arcmin
    SRCoptimalRATES = ScaledCtRate(SRCoptimalSEPAR,SIM_parameters["NearbySource_Table"]["Count Rates"], SIM_parameters["EffArea"], SIM_parameters["OffAxisAngle"])
    
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



def DataMap(SIM_parameters, Vector_Dictionary, OptimalPointingIdx, Nearby_SRCposition):
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
        # Plot the map of S/N ratio as function of NICER pointing
    fig = plt.figure(figsize=(10,6.5))
    ax = fig.add_subplot(111)
    plt.gca().invert_xaxis()
    plt.plot(Nearby_SRCposition.ra, Nearby_SRCposition.dec, marker='.', color='black',linestyle='')    
    plt.plot(SIM_parameters["PSRposition"].ra, SIM_parameters["PSRposition"].dec, marker='*', color='green',linestyle='')
    plt.plot(Vector_Dictionary['SampleRA'][OptimalPointingIdx], Vector_Dictionary['SampleDEC'][OptimalPointingIdx], marker='+', color='red', linestyle='')

    # label of the nearby sources
    plt.scatter(Vector_Dictionary['SampleRA'], Vector_Dictionary['SampleDEC'], c=Vector_Dictionary["SNR"], s=10, edgecolor='face')
    plt.xlabel('RA', fontsize='large')
    plt.ylabel('DEC', fontsize='large')
    plt.title("S/N map for " + SIM_parameters["Pulsar"])
    cbar = plt.colorbar()
    cbar.set_label('S/N')
    plt.show()
