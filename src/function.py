# --------------- Packages --------------- #

from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from scipy.optimize import curve_fit
from termcolor import colored
from astroquery.simbad import Simbad

import sys
import os
import argparse
import numpy as np
import subprocess
import matplotlib.pyplot as plt

# ---------------------------------------- #

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
    while True :
        if is_valid_file_path(PATH):
            print(f"The file at {colored(PATH, 'yellow')} is {colored('valid', 'green')}.")
            break
        else:
            print(f"The file at {colored(PATH, 'yellow')} doesn't exist or the path is {colored('invalid', 'red')}.")
            PATH = str(input("Enter the file path : \n"))
    return PATH


def choose_catalog(args_catalog):
    """
    Choose a catalog based on the provided keyword and return a valid file path for it.

    Parameters:
        Catalog (str): The catalog keyword, should be 'DR11' or 'DR13'.

    Returns:
        str: A valid file path for the selected catalog.

    Raises:
        argparse.ArgumentError: If an invalid catalog keyword is provided.
    """
    while True:
        try:
            if args_catalog == 'Xmm_DR13':
                print("\n")
                print(f"{colored(args_catalog, 'yellow')} catalog is loading")
                catalog_path = "data/4XMM_slim_DR13cat_v1.0.fits.gz"
                print("-"*50)
                valid_path = get_valid_file_path(catalog_path)
                print("-"*50, "\n")
                return valid_path, args_catalog
            elif args_catalog == 'CSC_2.0':
                print("\n")
                print(f"{colored(args_catalog, 'yellow')} catalog is loading")
                print("-"*50)
                valid_path = get_valid_file_path("data\Chandra.fits")
                print((f"The file at {colored(valid_path, 'yellow')} is {colored('valid', 'green')}."))
                print("-"*50, "\n")
                return valid_path, args_catalog
            elif args_catalog == 'Swift':
                print("\n")
                print(f"{colored(args_catalog, 'yellow')} catalog is loading")
                catalog_path = "data/Swift.fits"
                print("-"*50)
                valid_path = get_valid_file_path(catalog_path)
                print("-"*50, "\n")
                return valid_path, args_catalog
            elif args_catalog == 'eRosita':
                print("\n")
                print(f"{colored(args_catalog, 'yellow')} catalog is loading")
                catalog_path = "data/eRosita.fits"
                print("-"*50)
                valid_path = get_valid_file_path(catalog_path)
                print("-"*50, "\n")
                return valid_path, args_catalog
            elif args_catalog == "compare_catalog":
                print("\n")
                print("Enter catalog keyword (Xmm_DR13/CSC_2.0/Swift/eRosita)")
                catalog_1 = str(input("First catalog : "))
                while True:
                    if catalog_1 == "Xmm_DR13":
                        catalog_1_path = "data/4XMM_slim_DR13cat_v1.0.fits.gz"
                        break
                    elif catalog_1 == "CSC_2.0":
                        catalog_1_path = "data/Chandra.fits"
                        break
                    elif catalog_1 == "Swift":
                        catalog_1_path = "data/Swift.fits"
                        break
                    elif catalog_1 == "eRosita":
                        catalog_1_path = "data/eRosita.fits"
                        break
                    else:
                        catalog_1 = str(input("Keyword unnaccepted, retry : "))
                valid_path_1 = get_valid_file_path(catalog_1_path)
                
                catalog_2 = str(input("Second catalog : "))
                while True:
                    if catalog_2 == "Xmm_DR13":
                        catalog_2_path = "data/4XMM_slim_DR13cat_v1.0.fits.gz"
                        break
                    elif catalog_2 == "CSC_2.0":
                        catalog_2_path = "data/Chandra.fits"
                        break
                    elif catalog_2 == "Swift":
                        catalog_2_path = "data/Swift.fits"
                        break
                    elif catalog_2 == "eRosita":
                        catalog_2_path = "data/eRosita.fits"
                        break
                    else:
                        catalog_2 = str(input("Keyword unnaccepted, retry : "))
                valid_path_2 = get_valid_file_path(catalog_2_path)
                
                valid_path = (valid_path_1, valid_path_2, catalog_1, catalog_2)
                return valid_path, args_catalog
            else:
                raise argparse.ArgumentError(None, "invalid catalog keyword keyword. retry with Xmm_DR13, CSC_2.0, Swift, eRosita")
        except argparse.ArgumentError as error:
            print(f'An error occured : {error}')
            args_catalog = str(input("Enter a new key word : \n"))


def define_sources_list():
    """
    Prompts the user to define a list of sources for calculations. This function allows users to add sources either manually or by importing data from a file.

    Returns:
    SRC_LIST (list): A list of source tuples, each containing the source name, right ascension (ra), and declination (dec).

    The function begins by asking the user whether they want to add sources to the calculation.
    If the user chooses to add sources manually, they will be prompted to enter the source details (name, ra, and dec) for each source.
    If the user chooses to import data from a file, the function reads data from the specified file and extracts the necessary columns.
    The extracted data is then added to the SRC_LIST as source tuples.

    The function continues to prompt the user until they have either manually added all the sources they need or imported sources from a file.
    If the user enters invalid input at any point, the function will display an error message and continue to prompt for valid input.
    
    """
    UserList = []

    while True:
        try:
            choice = str(input("Add sources to calculation? (yes/y or no/n): ")).lower()

            if choice in ['no', 'n']:
                return UserList
            elif choice in ['yes', 'y']:
                break
            else:
                raise ValueError("Invalid input. Please enter 'yes' or 'y' for yes, 'no' or 'n' for no.")
        except ValueError as error:
            print(f"An error occured {error}")

    while True :
        open_file = str(input("Import data_src from a file? (yes/y or no/n): ")).lower()

        try:
            if open_file in ['yes', 'y']:
                print("Try : Catalog/exemple_src.txt")
                FILE_PATH = str(input("Enter a file_path : \n"))
                print("\n")
                print('-'*50)
                file_path = get_valid_file_path(FILE_PATH)
                print('-'*50)

                col1, ra, dec, value_var = np.loadtxt(file_path, unpack=True, usecols=(0, 1, 2, 3), dtype={'names': ('col1', 'ra', 'dec', 'valueVar'), 'formats': ('S25', 'f8', 'f8', 'f4')})
                name = [col1[data].decode().replace("_", " ") for data in range(len(col1))]

                for value in range(len(col1)):
                    UserList.append((name[value], ra[value], dec[value],value_var[value]))
                break

            elif open_file in ['no', 'n']:

                nbr_src = int(input("How many sources do you need to add ? "))
                item = 0
                print("\n")

                while item < nbr_src:
                    name = input('Enter source name : ')
                    ra = float(input('Enter right ascension : '))
                    dec = float(input('Enter declination : '))
                    value_var = input("Enter the value of variability rate of your object, enter nan if the object is invariant : ")
                    if value_var == 'nan':
                        value_var = np.nan
                    UserList.append((name, ra, dec, value_var))
                    item += 1
                    print(f"{colored(item, 'blue')} item added to the list \n")

                break
            else:
                raise ValueError("Invalid input. Please enter 'yes' or 'y' for yes, 'no' or 'n' for no.")
        except ValueError as error:
            print(f"An error occured {error}")
   
    return UserList


def get_coord_psr(name):
    """
    Get the PSR coordinates from the SIMBAD database.

    Parameters:
    name (str): The name of the pulsar object.

    Returns:
    SkyCoord: A SkyCoord object representing the coordinates (RA and DEC) of the pulsar.
    """
    return SkyCoord(ra=Simbad.query_object(name)['RA'][0], dec=Simbad.query_object(name)['DEC'][0], unit=(u.hourangle, u.deg))


def scaled_ct_rate(D, OptCtRate, effareaX, effareaY):
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


def ang_separation(reference, obj):
    """
    Calculate the angular separation between two celestial objects.

    Parameters:
    reference (SkyCoord): The reference object's coordinates.
    obj (SkyCoord): The coordinates of the object to which the separation is calculated.

    Returns:
    Quantity: The angular separation between the two objects.
    """
    return reference.separation(obj)


def signal_to_noise(SrcCtsRate, BkgSrcRates, InstBkgd, ExpTime):
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


def nominal_pointing_info(simulation_data, NearbySRCposition):
    """
    Calculate and print various information related to nominal pointing.

    Parameters:
    SIM_parameters (dict): Dictionary containing simulation parameters.
    NearbySRCposition (SkyCoord): Coordinates of nearby sources.

    Returns:
    None
    """

    object_data = simulation_data['object_data']
    telescop_data = simulation_data['telescop_data']
    
    SRCnominalDIST = ang_separation(NearbySRCposition, SkyCoord(ra=object_data['object_position'].ra, dec=object_data['object_position'].dec)).arcmin
    SRCscaleRates = scaled_ct_rate(SRCnominalDIST, simulation_data["NearbySRC_Table"]["Count Rates"], telescop_data["EffArea"], telescop_data["OffAxisAngle"])
    PSRcountrates = object_data['CountRate']

    print('PSR S/N at Nominal Pointing ' + str(signal_to_noise(PSRcountrates, SRCscaleRates, simulation_data["INSTbkgd"], simulation_data["EXPtime"])))
    print("PSR count rate at Nominal pointing = " + str(PSRcountrates) + "cts/sec")
    print("BKG sources count rate at Nominal Pointing = " + str(np.sum(SRCscaleRates)) + "cts/sec")
    print("             Individual BKG Sources rates:")
    print(str(SRCscaleRates))
    print("             BKG sources distance from PSR (\')")
    print(SRCnominalDIST)
    print("--------------------------------------------------")


def calculate_opti_point(simulation_data, nearby_src_position):
    """
    Calculate the optimal pointing for the NICER telescope to maximize the signal-to-noise ratio (S/N).

    This function calculates the optimal pointing of the NICER telescope by adjusting its position
    to maximize the signal-to-noise ratio (S/N) in the presence of a pulsar source and background sources (SRC).
    It iterates over a grid of angular displacements around the pulsar position and calculates the S/N for
    each possible pointing.

    Args:
        SIM_parameters (dict): A dictionary containing simulation parameters, including the pulsar position,
            pulsar count rates, SRC count rates, effective area, off-axis angle, instrumental background, and exposure time.
        NearbySRCposition (SkyCoord): Coordinates of the nearby SRC sources.

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
    object_data = simulation_data['object_data']
    telescop_data = simulation_data['telescop_data']

    DeltaRA = Angle(np.arange(-3.0, 3.1, 0.1), unit=u.deg)/60
    DeltaDEC = Angle(np.arange(-3.0, 3.1, 0.1), unit=u.deg)/60

    SampleRA, SampleDEC, SNR, PSRrates, SRCrates = np.zeros((5, len(DeltaRA) * len(DeltaDEC)))
    PSRcountrates = object_data['CountRate']

    count = 0
    for i in DeltaRA:
        for j in DeltaDEC:
                NICERpointing = SkyCoord(ra=object_data["object_position"].ra + i, dec=object_data["object_position"].dec + j)
                PSRseparation = ang_separation(object_data["object_position"], NICERpointing)
                SRCseparation = ang_separation(nearby_src_position, NICERpointing)

                PSRcountrateScaled = scaled_ct_rate(PSRseparation.arcmin, PSRcountrates, telescop_data["EffArea"], telescop_data["OffAxisAngle"])
                SRCcountrateScaled = scaled_ct_rate(SRCseparation.arcmin, simulation_data["NearbySRC_Table"]["Count Rates"], telescop_data["EffArea"], telescop_data["OffAxisAngle"])

                SampleRA[count] = NICERpointing.ra.deg
                SampleDEC[count] = NICERpointing.dec.deg

                PSRrates[count] = PSRcountrateScaled
                SRCrates[count] = np.sum(SRCcountrateScaled)

                SNR[count] = signal_to_noise(PSRcountrateScaled, SRCcountrateScaled, simulation_data["INSTbkgd"], simulation_data["EXPtime"])
                count = count + 1

    OptimalPointingIdx = np.where(SNR==max(SNR))[0][0]
    SRCoptimalSEPAR = ang_separation(nearby_src_position, SkyCoord(ra=SampleRA[OptimalPointingIdx]*u.degree, dec=SampleDEC[OptimalPointingIdx]*u.degree)).arcmin
    SRCoptimalRATES = scaled_ct_rate(SRCoptimalSEPAR, simulation_data["NearbySRC_Table"]["Count Rates"], telescop_data["EffArea"], telescop_data["OffAxisAngle"])

    Vector_Dictionary = {'SampleRA': SampleRA,
                         'SampleDEC': SampleDEC,
                         'PSRrates': PSRrates,
                         'SRCrates': SRCrates,
                         'SNR': SNR
                         }

    return OptimalPointingIdx, SRCoptimalSEPAR, SRCoptimalRATES, Vector_Dictionary


def optimal_point_infos(vector_dictionary, OptimalPointingIdx, SRCoptimalRATES):
    """
    Print information for the optimal NICER pointing that maximizes the signal-to-noise ratio (S/N).

    This function prints information about the S/N ratio, pulsar count rate, background sources count rate,
    individual background source rates, and the optimal pointing coordinates for the NICER telescope.

    Args:
        vector_dictionary (dict): A dictionary containing result vectors, including sampled RA and DEC positions,
                                  pulsar count rates, SRC count rates, and the S/N ratio for each pointing.
        OptimalPointingIdx (int): The index of the optimal pointing in the result vectors.
        SRCoptimalRATES (float): The SRC count rate at the optimal pointing.
    """

    # Print info for the optimal NICER pointing that maximizes the S/N ratio
    print ("PSR S/N at Optimal Pointing " + str(vector_dictionary['SNR'][OptimalPointingIdx]))
    print ("PSR count rate at Optimal pointing = " + str(vector_dictionary["PSRrates"][OptimalPointingIdx]) + " cts/sec")
    print ("BKG sources count rate at Optimal pointing = " + str(vector_dictionary["SRCrates"][OptimalPointingIdx]) + " cts/sec")
    print ("     Individual BKG Sources: " )
    print (str(SRCoptimalRATES))
    #print "     Distance from Optimal Pointing (\")"
    #print str(SRCoptimalSEPAR)
    print ("Optimal Pointing:  " + str(vector_dictionary["SampleRA"][OptimalPointingIdx]) + "  " + str(vector_dictionary["SampleDEC"][OptimalPointingIdx]))
    print ("----------------------------------------------------------------------")


def data_map(simulation_data, vector_dictionary, OptimalPointingIdx, NearbySRCposition):
    """
    Plot the map of the Signal-to-Noise (S/N) ratio as a function of NICER pointing.

    Parameters:
    - SIM_parameters (dict): A dictionary containing simulation parameters, including the pulsar position.
    - Vector_Dictionary (dict): A dictionary containing vector data, including SampleRA, SampleDEC, and SNR.
    - OptimalPointingIdx (int): The index of the optimal pointing in Vector_Dictionary.
    - NearbySRCposition (SkyCoord): SkyCoord object representing the positions of nearby sources.

    Returns:
    None
    """

    object_data = simulation_data['object_data']

    # Plot the map of S/N ratio as function of NICER pointing
    fig = plt.figure(figsize=(10,6.5))
    ax = fig.add_subplot(111)
    plt.gca().invert_xaxis()
    plt.plot(NearbySRCposition.ra, NearbySRCposition.dec, marker='.', color='black',linestyle='')    
    plt.plot(object_data["object_position"].ra, object_data["object_position"].dec, marker='*', color='green',linestyle='')
    plt.plot(vector_dictionary['SampleRA'][OptimalPointingIdx], vector_dictionary['SampleDEC'][OptimalPointingIdx], marker='+', color='red', linestyle='')

    # label of the nearby sources
    plt.scatter(vector_dictionary['SampleRA'], vector_dictionary['SampleDEC'], c=vector_dictionary["SNR"], s=10, edgecolor='face')
    plt.xlabel('RA', fontsize='large')
    plt.ylabel('DEC', fontsize='large')
    plt.title("S/N map for " + object_data["object_name"])
    cbar = plt.colorbar()
    cbar.set_label('S/N')
    plt.show()


def count_rates(nearby_src_table, model_dictionary, telescop_data):
    """
        Calculate X-ray count rates for nearby sources using PIMMS modeling.

        This function calculates X-ray count rates for a set of nearby sources based on their model information and associated parameters.
        It uses PIMMS (Portable, Interactive Multi-Mission Simulator) to perform the modeling and computes the count rates.

        Parameters:
            nearby_src_table (Table): A table containing data on nearby sources, including model information, X-ray flux, and column density.
            model_dictionary (dict): A dictionary mapping sources (e.g., "src_0") to model and parameter information.

        Returns:
            tuple: A tuple containing two elements:
                - A NumPy array of X-ray count rates for each nearby source.
                - An updated 'nearby_src_table' with the added 'Count Rates' column.

        Note:
        - PIMMS modeling commands are generated for each source based on the model, model value, X-ray flux, and column density.
        - The 'Count Rates' column is added to the 'nearby_src_table' with the calculated count rates.
    """
    number_source = len(model_dictionary)
    count_rates = np.array([], dtype=float)
    
    telescop_name = telescop_data['telescop_name']
    min_value = telescop_data['min_value']
    max_value = telescop_data['max_value']
    energy_band = telescop_data['energy_band']
    
    for item in range(number_source):
        model = model_dictionary[f"src_{item}"]["model"]
        model_value = model_dictionary[f"src_{item}"]["model_value"]
        xmm_flux =  model_dictionary[f"src_{item}"]["flux"]
        nh_value = model_dictionary[f"src_{item}"]["column_dentsity"]
                
        pimms_cmds = f"instrument {telescop_name} {min_value}-{max_value}\nfrom flux ERGS {energy_band}\nmodel galactic nh {nh_value}\nmodel {model} {model_value} 0.0\ngo {xmm_flux}\nexit\n"
        
        with open('pimms_script.xco', 'w') as file:
            file.write(pimms_cmds)
            file.close()

        result = subprocess.run(['pimms', '@pimms_script.xco'], stdout=subprocess.PIPE).stdout.decode(sys.stdout.encoding)
        count_rate = float(result.split("predicts")[1].split('cps')[0])
        count_rates = np.append(count_rates, count_rate)
        
    nearby_src_table["Count Rates"] = count_rates
        
    return count_rates, nearby_src_table


def sources_to_unique_sources(result_table, column_names):
    """
    Given a table of sources (result_table) and a list of column names (column_names), 
    this function generates a new table that contains unique sources based on the 
    specified column(s). If there are multiple rows with the same value in the 
    specified column, the function computes the mean of the corresponding data 
    in other columns.

    Parameters:
    result_table (Table): A table containing source data.
    column_names (list): A list of column names, with the first element being 
                         the column by which uniqueness is determined.

    Returns:
    Table: A new table with unique sources, where duplicates in the specified 
           column(s) have been resolved by either taking the single value 
           or computing the mean of other columns.
    """
    name_col = column_names[0]
    name_list = result_table[name_col]
    unique_name_list = list(set(result_table[name_col]))

    data_list = []
    for col in column_names:
        data = []
        for name in unique_name_list:
            number_iter = list(name_list).count(name)
            if number_iter == 1:
                index = list(result_table[name_col]).index(name)
                if 'IAUNAME' in col:
                    data.append(name)
                else:
                    data.append(result_table[col][index])
            else:
                index_list = [index for index, unique_name in enumerate(name_list) if unique_name == name]
                index = index_list[0]
                
                if 'IAUNAME' in col:
                    data.append(name)
                elif 'RA' or 'DEC' in col:
                    data.append(result_table[col][index])
                else:
                    mean_data = []
                    for item in index_list:
                        mean_data.append(result_table[col][item])
                    data.append(np.mean(mean_data))
    
        data_list.append(data)
    
    nearby_src_table = Table(names=column_names,
                             data=data_list)
    
    return nearby_src_table