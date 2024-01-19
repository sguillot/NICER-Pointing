# --------------- Packages --------------- #

from astropy import units as u
from astropy.table import Table
from termcolor import colored
from astroquery.simbad import Simbad
from jaxspec.model.multiplicative import Tbabs
from jaxspec.model.additive import Powerlaw
from jax.config import config
from jaxspec.data.instrument import Instrument

# ---------- import class ---------- #

from catalog_class.XmmClass import XmmCatalog
from catalog_class.ChandraClass import ChandraCatalog
from catalog_class.SwiftClass import SwiftCatalog
from catalog_class.eRositaClass import eRositaCatalog
from catalog_class.CompareCatalogClass import CompareCatalog
from catalog_class.MatchClass import MatchCatalog

# ---------------------------------- #

# ---------- import function ---------- #

import function.init_function as i_f
import function.calculation_function as c_f
import function.software_function as s_f
import function.jaxspec_function as j_f

# ------------------------------------- #

import argparse
import numpy as np
import os
import subprocess
import sys
import shlex
import catalog_information as dict_cat
import numpyro
import platform

# ---------------------------------------- #

# ---------- for documentation --------- #

# from src.catalog_class.XmmClass import XmmCatalog
# from src.catalog_class.ChandraClass import ChandraCatalog
# from src.catalog_class.SwiftClass import SwiftCatalog
# from src.catalog_class.eRositaClass import eRositaCatalog
# from src.catalog_class.CompareCatalogClass import CompareCatalog
# from src.catalog_class.MatchClass import MatchCatalog
# import src.function.init_function as i_f
# import src.function.calculation_function as c_f
# import src.function.software_function as s_f
# import src.function.jaxspec_function as j_f
# import src.catalog_information as dict_cat

# -------------------------------------- #

def main():
    """
    This script (main.py) serves as the entry point for the Optimal Pointing Point Code for the NICER telescope. It leverages various classes and functions to load data, perform calculations, and generate visualizations and outputs for astronomical studies.

    Note:
        The script is structured into several main sections:
        
        - Initialization: Setting up basic parameters and loading necessary catalogs.
        - Argument Processing: Using `argparse` to handle command-line options.
        - Data Loading and Analysis: Loading data from specified catalogs and performing analysis.
        - Modeling and Visualization: Performing modeling calculations and generating visualizations.

    Each section is documented to aid in understanding and maintaining the code.
    
    Example:
        This is how we can execute the program:
        
        >>> Optimal_Pointing_Point_Code > python ./src/main.py [*args]


    Important:
        Here are the different arguments to execute the project:
        
        - --info, -i : Display a table of pulsars.
        - --name, -n : Specify the name of a celestial object for analysis.
        - --coord, -co : Specify the coordinates of a celestial object.
        - --radius, -r : Define the radius of the area to be analyzed.
        - --exp_time, -e_t : Set the exposure time for modeling.
        - --catalog, -ca : Choose the catalog to use for analysis:
            - Key : Xmm_DR13 / CSC_2.0 / Swift / eRosita / match / compare_catalog
        
    Example:
        Here is an example demonstrating the use of the argument:
        
        With the name of the object -->
        
        >>> Optimal_Pointing_Point_Code > python ./src/main.py --name OBJECT_NAME --radius RADIUS_VALUE --exp_time MODELING_TIME --CATALOG CATALOG_KEY
        
        With the coordinates of the object -->
        
        >>> Optimal_Pointing_Point_Code > python ./src/main.py --coord RA DEC --radius RADIUS_VALUE --exp_time MODELING_TIME --CATALOG CATALOG_KEY
        
        
    Initialization:
    ===============
        This script serves as the primary interface for the Optimal Pointing Point Code designed for the NICER telescope. It initializes a set of predefined 
        astronomical catalogs and sets up a command-line argument parser to process user inputs. The available catalogs include 'XMM', 'Chandra', 'Swift', 'eRosita'. 

        Users can interact with the script by specifying options such as displaying a pulsar table, providing the name or coordinates of a celestial object, defining the 
        field of view radius, setting the exposure time for data modeling, or selecting a specific catalog for analysis. The script then performs data collection and 
        processing based on these inputs. For instance, if a celestial object name is provided, the script fetches its coordinates from the Simbad Database and proceeds 
        with the chosen catalog for further analysis. Similarly, when coordinates are provided, the script identifies the corresponding celestial object. 
        The script is designed to handle exceptions and guide users through valid inputs, ensuring a smooth data retrieval and processing experience.
        
    Dictionary to stock data:
    =========================
    This script section initializes several dictionaries and data structures, crucial for the NICER telescope pointing optimization process.

    object data Dictionary:
    -----------------------
    The 'object_data' dictionary holds crucial information about the celestial object of interest. It includes:
    
    - 'object_name': Name of the celestial object.
    - 'object_position': Astropy SkyCoord object representing the object's celestial coordinates.
    - 'count_rate': The observed count rate of the object.
    
    This dictionary is pivotal for various calculations and data retrieval related to the target object.
    
    Example:
    
    >>> object_data = {"object_name": "name",
                       "object_position": object_position,
                       "count_rate": count_rate
                       }

    telescop data Dictionary:
    -------------------------
    The 'telescop_data' dictionary contains parameters specific to the NICER telescope and its observational capabilities. It includes:
    
    - 'telescop_name': Name of the telescope, in this case, 'nicer'.
    - 'EffArea': The effective area of the telescope.
    - 'OffAxisAngle': Off-axis angle data.
    - 'nicer_data_arf', 'nicer_data_rmf': Paths to the Auxiliary Response File (ARF) and the Response Matrix File (RMF).
    - 'min_value', 'max_value': The minimum and maximum values for certain telescope parameters.
    - 'energy_band': The energy band of interest for observations.
    
    Example:
    
    >>> telescop_data = {"telescop_name": "nicer",
                        "EffArea": EffArea,
                        "OffAxisAngle": OffAxisAngle,
                        "nicer_data_arf": nicer_data_arf,
                        "nicer_data_rmf": nicer_data_rmf,
                        "min_value": 0.3,
                        "max_value": 10.0,
                        "energy_band": "0.2-12.0"
                        }

    simulation data Dictionary:
    ---------------------------
    The 'simulation_data' dictionary aggregates data required for simulation and modeling. It includes:
    
    - 'object_data': A reference to the 'object_data' dictionary.
    - 'telescop_data': A reference to the 'telescop_data' dictionary.
    - 'INSTbkgd': The instrumental background value.
    - 'EXPtime': The exposure time for the observations.
    
    Example:
    
    >>> simulation_data = {"object_data": object_data,
                           "telescop_data": telescop_data,
                           "INSTbkgd": 0.2,
                           "EXPtime": args.exp_time
                           }
    
    This dictionary is used extensively throughout the simulation process.

    Note:
        - The accuracy and reliability of the simulation are heavily dependent on the correctness and precision of the data provided in these dictionaries.
        - Users should verify the integrity and relevance of the data, especially when using custom input or dealing with unfamiliar celestial objects.

    Important:
        - Changes to these dictionaries should be made cautiously, as they can significantly impact the simulation results.
        - Ensure that the paths to data files and software tools are correctly set according to your system's configuration.

    
    This section of the script is dedicated to handling various astronomical catalogs for the NICER telescope pointing optimization. Depending on the user-selected catalog, the script sets up the necessary directories, data paths, and initializes specific classes for modeling and analysis.


    Accessing Class Functionality
    =============================
    
    XMM-DR13 Catalog Handling [Xmm_DR13]
    ------------------------------------
    
    This section of the script focuses on using the XMM-DR13 catalog for optimizing NICER telescope pointing. The XMM-Newton observatory, from which this catalog is derived, is known for its detailed X-ray observations of the universe.

    Functionality:
        - The script establishes a dedicated directory specifically for the 4XMM_DR13 catalog, organizing data and images for analysis.
        - The `XmmCatalog` class is initialized to handle the processing and analysis of data from the XMM-DR13 catalog.
        - Essential directories for image storage and catalog data are created to support the processing.

    XMM-DR13 Catalog Processing:
        - Configuration settings for the XMM-DR13 catalog, including paths for modeling files and software tools, are stored in `os_dictionary`.
        - The `XmmCatalog` class specializes in analyzing the high-energy X-ray data from the XMM-Newton observatory.
        - The script sets up columns and parameters specific to the XMM-DR13 catalog, facilitating the analysis process.

    Note:
        - XMM-DR13 is an invaluable resource for studying cosmic X-ray sources, providing extensive data for NICER's pointing optimization.
        - The catalog's detailed spectral and timing data enhance the understanding of various astrophysical phenomena.

    Important:
        - Users must ensure that the XMM-DR13 catalog data is current and correctly structured for effective integration into the analysis.
        - Precise configuration in the `XmmCatalog` class is crucial for accurate data processing and extraction of meaningful results.

    Upon completing the setup and processing of the XMM-DR13 catalog data, the script updates `simulation_data` for subsequent analysis stages, leveraging the detailed X-ray observations from XMM-Newton.


    Chandra Catalog Handling [CSC_2.0]
    ----------------------------------
    This segment of the script is tailored for the analysis and optimization of NICER telescope pointing using the Chandra catalog. The Chandra X-ray Observatory's data is renowned for its high spatial resolution and depth in X-ray astronomy.

    Functionality:
        - Creation of a Chandra-specific directory to manage and organize the catalog data efficiently.
        - The `ChandraCatalog` class is initialized for detailed processing and analysis of the Chandra catalog data.
        - Directories for image storage and the closest catalog data are established to aid in data handling.

    Chandra Catalog Processing:
        - Configuration settings specific to the Chandra catalog are stored in `os_dictionary`, including paths for modeling files, images, and relevant software tools.
        - The `ChandraCatalog` class manages the intricacies of Chandra data, focusing on high-resolution X-ray observations.
        - Users are given the option to choose between the standard Chandra catalog and the CS_Chandra (cone search) catalog for their analysis.

    Note:
        - The Chandra catalog is particularly useful for studying detailed structures in X-ray emitting regions, benefiting NICER's pointing accuracy.
        - The choice between Chandra and CS_Chandra catalog versions allows for flexibility in analysis, depending on the specific requirements of the study.

    Important:
        - Users must verify the availability and correctness of Chandra catalog data in the specified directories for successful processing.
        - The choice between the two Chandra catalog versions should be made based on the nature of the data and the specific goals of the analysis.

    After setting up the necessary configuration and making the choice of catalog, the script proceeds with the processing of the Chandra data, updating `simulation_data` accordingly for the next stages of analysis.


    Swift Catalog Handling [Swift]
    ------------------------------
    This part of the script is focused on utilizing the Swift catalog for NICER telescope pointing optimization. The Swift Gamma-Ray Burst Mission provides valuable multi-wavelength astronomical data, which is crucial for various astrophysical studies.

    Functionality:
        - The script creates a dedicated directory for the Swift catalog to organize the processing and storage of data.
        - The `SwiftCatalog` class is initialized for handling the data analysis and modeling from the Swift catalog.
        - Necessary directories for image storage and catalog data are established.

    Swift Catalog Processing:
        - Configuration for the Swift catalog, including paths for modeling files, images, and relevant software tools, is stored in `os_dictionary`.
        - The `SwiftCatalog` class efficiently processes data from the Swift mission, particularly focusing on gamma-ray burst data and other astronomical phenomena.

    Note:
        - Swift's rapid-response capability and broad wavelength coverage provide a unique dataset for astrophysical research, enhancing NICER's observational strategies.
        - The catalog's data is instrumental in studying transient and high-energy phenomena in the universe.

    Important:
        - Users must ensure that the Swift catalog data is up-to-date and accurately formatted for seamless integration into the analysis process.
        - Adjustments in the `SwiftCatalog` class or the parameters used for analysis should be approached with careful consideration to preserve the accuracy of the results.

    After processing the Swift catalog data, the script updates `simulation_data` and prepares for subsequent stages of analysis, leveraging the unique capabilities of the Swift mission for NICER's optimization.


    eRosita Catalog Handling [eRosita]
    ----------------------------------
    This section of the script focuses on the utilization of the eRosita catalog for the NICER telescope pointing optimization. The eRosita space telescope provides comprehensive all-sky survey data in X-ray bands, making it invaluable for astronomical research and analysis.

    Functionality:
        - The script sets up a dedicated directory for the eRosita catalog, ensuring organized storage and retrieval of data.
        - The `eRositaCatalog` class is initialized to process and analyze the data from this catalog.
        - Directories for storing images and the closest catalog data are created to facilitate effective data handling and visualization.

    eRosita Catalog Processing:
        - Configuration settings specific to the eRosita catalog are stored in `os_dictionary`, including paths for modeling files, images, and software tools.
        - The `eRositaCatalog` class manages the data from the eRosita space telescope, focusing on optimal point determination for NICER.

    Note:
        - The eRosita catalog's comprehensive sky coverage offers a wide range of data for analysis, enhancing the scope of the NICER pointing optimization.
        - The data from eRosita is crucial for understanding large-scale structures and energetic phenomena in the universe.

    Important:
        - Users should ensure that the eRosita catalog data is current and correctly formatted for effective processing.
        - Adjustments in the `eRositaCatalog` class or the analysis parameters should be made with caution to maintain the integrity of the results.

    Upon completion of processing, the script updates the `simulation_data` with key information from the eRosita catalog and prepares the data for subsequent analysis stages.


    Combined XMM and Chandra Catalog Handling [match]
    -------------------------------------------------
    If 'match' is chosen:
    
    Functionality:
        - The script creates a dedicated directory for combined XMM and Chandra data, allowing for an organized approach to handle these two significant sources of astronomical data.
        - The `MatchCatalog` class is specifically designed to merge and analyze data from both the XMM and Chandra catalogs.
        - Essential paths and directories are configured to facilitate data processing, storage, and visualization.

    Modeling Spectra with JAXSpec:
        - The script utilizes the JAXSpec library to model the spectra of astronomical sources.
        - The spectral model is defined using a combination of absorption (Tbabs) and power-law components.
        - Instrument parameters for NICER are loaded, and spectra from the combined catalogs are processed and visualized.

    
    Note:
        - The combination of XMM and Chandra data provides a broader dataset, enhancing the accuracy and reliability of the optimal pointing analysis.
        - Care is taken to synchronize data formats and parameters from both catalogs for consistency.

    Important:
        - Users should ensure that the data from both XMM and Chandra catalogs are up-to-date and correctly formatted.
        - Modifications in the `MatchCatalog` class or spectral modeling parameters should be approached with caution, as they can significantly impact the results of the analysis.

    After processing, the script outputs spectral plots and exits, completing the analysis for the 'xmmXchandra' catalog choice.


    Comparing Data Across Two Catalogs [compare_catalog]
    ------------------------------------------------------

    This section of the script is dedicated to comparing data across two different astronomical catalogs. This comparison is essential for validating data consistency and accuracy in astronomical studies.

    Functionality:
        - The script sets up a 'Compare_catalog' directory for the comparison of data between two catalogs. This directory structure facilitates organized data management and analysis.
        - The `CompareCatalog` class is initialized to handle the comparison process. This class is specifically designed to analyze and contrast data from different catalogs.
        - Essential directories for storing images and closest catalog data are also created.

    Catalog Comparison Setup:
        - Configuration settings for the comparison process are stored in `os_dictionary`.
        - Paths for data, modeling files, and software tools like stilts and topcat are defined to support the comparison process.
        - The `CompareCatalog` class handles the intricacies of comparing data across catalogs, ensuring the process is efficient and accurate.

    Note:
        - This comparison approach is crucial for cross-validating observations and results from different sources.
        - It helps in identifying discrepancies or similarities in data which are key to robust astronomical analysis.

    Important:
        - Users must ensure that both catalogs involved in the comparison have compatible data formats and are correctly aligned.
        - Careful consideration is needed when interpreting comparison results, as differences may arise from various observational or processing techniques.

    After executing the comparison process, the script terminates, concluding the analysis for the 'compare_catalog' catalog choice.

    Count Rate - Optimal Pointing Point - Vignetting Factor
    =======================================================
    
    The choice of keys "match" and "compare_catalog" allows the algorithm to perform the necessary calculations within the classes or the dedicated section for these keys.

    If the user selects one of the following keys: {"Xmm_DR13", "CSC_2.0", "Swift", "eRosita"}, then the classes respectively generate the nearby sources table (nearby_sources_table), 
    a list of the positions of these sources (nearby_sources_position), photon index data, hydrogen density column, and a dictionary (model_dictionary) containing the model for determining count rates, 
    the value of this model, the median flux, and Nh.
    
    First, we will call the 'c_f.count_rates' function from the 'calculation_function.py' module, which, with the help of 'model_dictionary' and the WebPIMMS software, will accurately 
    determine the value of the count rates
    
    Example:
        Use c_f.count_rate function -->
        
        >>> count_rates, nearby_sources_table = c_f.count_rates(nearby_sources_table, model_dictionary, telescop_data)
    
    Subsequently, now that we have all the necessary data at our disposal, we can determine the optimal line of sight for the NICER telescope. To do this, we will utilize the 'c_f.calculate_opti_point' 
    function from the 'calculation_function.py' module.
    
    Example:
        Use c_f.calculate_opti_point function -->
        
        >>> OptimalPointingIdx, SRCoptimalSEPAR, SRCoptimalRATES, vector_dictionary = c_f.calculate_opti_point(simulation_data, nearby_sources_position)
    
    Here is an example of this function
    
    .. image:: C:/Users/plamb_v00y0i4/OneDrive/Bureau/Optimal_Pointing_Point_Code/modeling_result/PSR_J0437-4715/Chandra/img/Chandra_SNR_PSR_J0437-4715.png
        :align: center
        
    Once all these values are determined, we can calculate the vignetting factor based on this line of sight. This factor will enable us to generate the spectra of the nuisance sources in the most
    effective manner, allowing us to determine the sum of contributions around the targeted object. We can get these values using "c_f.vignetting_factor"from the"calculation_function.py" module.
    
    Example:
        Use c_f.vignetting_factor function -->
        
        >>> vignetting_factor, nearby_sources_table = c_f.vignetting_factor(OptimalPointingIdx=OptimalPointingIdx, vector_dictionary=vector_dictionary, simulation_data=simulation_data, data=column_dictionary["data_to_vignetting"], nearby_sources_table=nearby_sources_table)

    Variable Sources detected with Erwan's Quintin Code
    ===================================================
    
    With a portion of code from the S.T.O.N.K.S project, it is possible for us to determine the sources around the targeted object that have been classified as variable. To achieve this, we will use several 
    other catalogs such as RASS, Stacked, Slew, WGACAT in addition to our 4 catalogs. The idea is to compare these data to identify whether the sources exhibit variations or not. A table named 'Master_Source.fits'
    stores all the source names, including those that are common across all catalogs.

    All the variable sources are stored in the table 'Master_source_cone.fits'. With this information, we can, for example, generate spectra for these sources.
    
    Note:
        - Ensure that you have the necessary data files, including 'Master_Source.fits' and 'Master_source_cone.fits', to perform variable source detection.

    Important:
        - The variable source detection process may require significant computational resources, depending on the size of the datasets and the complexity of the analysis.
    
    .. image:: C:/Users/plamb_v00y0i4/OneDrive/Bureau/Optimal_Pointing_Point_Code/modeling_result/PSR_J0437-4715/plot_var_sources/sources_plot_6.png
        :align: center
        
    Link to the S.T.O.N.K.S project : <https://github.com/ErwanQuintin/STONKS>
    
    Spectra Modeling with Jaxspec
    =============================

    Jaxspec, developed at Irap by Dr. Simon Dupourqué and Dr. Didier Barret, is a Python package designed to facilitate spectral analysis.
    Link to the project : <https://github.com/renecotyfanboy/jaxspec>

    Note:
        - Before using Jaxspec, make sure to install it by following the instructions provided in the project's documentation.

    Important:
        - Ensure that you have the necessary data files, including 'Master_source_cone.fits', 'arf', and 'rmf', to perform spectral modeling.

    Initially, we have the table of nearby sources and the table of variable sources at our disposal. We need to determine which of our nearby sources are variable. 
    To do this, we will use the 'Master_source_cone.fits' table to find them. To achieve this, we will use the 'j_f.cross_catalog_index' function from the 'jaxspec_function.py' 
    module. This function is used to create a list of indices for the variable sources based on the nearby sources table.

    Example:
        Use j_f.cross_catalog function -->

        >>> var_index =  j_f.cross_catalog_index(output_name=output_name, key=key, iauname=column_dictionary["data_to_vignetting"][2], nearby_sources_table=nearby_sources_table)

    Next, we will define a model to generate these spectra and load the NICER instrument parameters (the .arf and .rmf files).

    Example:
        >>> model = Tbabs() * Powerlaw()
        >>> instrument = Instrument.from_ogip_file(nicer_data_arf, nicer_data_rmf, exposure=args.exp_time)

    Finally, we only need to generate the spectra of the sources, which we do using the 'j_f.modeling_source_spectra' function from the 'jaxspec_function.py' module.

    Example:
        Use j_f.modeling_source_spectra function -->

        >>> total_spectra, total_var_spectra = j_f.modeling_source_spectra(nearby_sources_table=nearby_sources_table, instrument=instrument, model=model, var_index=var_index)

    Once these spectra are obtained, we can plot the different spectra of these sources as well as the sum of contributions using the 'j_f.total_plot_spectra' function from the 'jaxspec_function.py' module.

    Example:
        Use j_f.total_plot_spectra function ->

        >>> data = j_f.total_plot_spectra(total_spectra=total_spectra, total_var_spectra=total_var_spectra, instrument=instrument, simulation_data=simulation_data, catalog_name=args.catalog)

    Here is an example of this plot :

    .. image:: C:/Users/plamb_v00y0i4/OneDrive/Bureau/Optimal_Pointing_Point_Code/modeling_result/PSR_J0437-4715/4XMM_DR13/img/XMM_spectral_modeling_close_to_PSR_J0437-4715.png
        :align: center

    In this graph, we have an envelope of variability from the variable sources in the modeling. The last function we use, 'j_f.write_twt_file' from the 'jaxspec_function.py' module, allows us
    to generate a .txt file containing the boundary values of this envelope using the 'data'.

    Example:
        Use j_f.write_txt_file function ->

        >>> j_f.write_txt_file(simulation_data=simulation_data, data=data)

    """
    # --------------- Initialization --------------- #
 
    catalogs = ["XMM", "Chandra", "Swift", "eRosita", "Slew", "RASS", "WGACAT", "Stacked"]

    parser = argparse.ArgumentParser(description="Code optimal pointing point for NICER",
                                    epilog="Focus an object with his name or his coordinate")

    main_group = parser.add_mutually_exclusive_group()
    main_group.add_argument("--info", '-i', action='store_true',
                            help="Display a pulsar table")
    main_group.add_argument('--name', '-n', type=str, 
                            help="Enter an object name")
    main_group.add_argument('--coord', '-co', type=float, 
                            nargs=2, help="Enter your object coordinates : ra dec")

    parser.add_argument('--radius', '-r', type=float, 
                        help="Enter the radius of the field of view (unit = arcmin)")

    parser.add_argument('--exp_time', '-e_t', type=int,
                        help="Enter the exposure time to modeling data")

    parser.add_argument('--catalog', '-ca', type=str, 
                        help="Enter catalog keyword : Xmm_DR13/CSC_2.0/Swift/eRosita/compare_catalog/match")

    args = parser.parse_args()

    psr_name = np.array(["PSR J0437-4715", "PSR J2124-3358", "PSR J0751+1807", "PSR J1231-1411"], dtype=str)
    psr_coord = np.array([f"{i_f.get_coord_psr(name).ra} {i_f.get_coord_psr(name).dec}" for name in psr_name])
    psr_count_rate = np.array([1.319, 0.1, 0.025, 0.27])
    psr_table = Table(names=["full psr name", "psr coord", "psr count rate"],
                        data=[psr_name, psr_coord, psr_count_rate])

    if args.info :
        print(psr_table)
        sys.exit()
        
    if args.name:
        while True:
            if '_' in args.name:
                object_name = args.name.replace('_', " ")
                print(f"\nCollecting data for {colored(object_name, 'magenta')}")
            try:
                object_position = i_f.get_coord_psr(object_name)
                print(f"\n{colored(object_name, 'green')} is in Simbad Database, here is his coordinate :\n{object_position}")
                break
            except Exception as error:
                print(f"Error : {colored(object_name, 'red')}, isn't in Simbad Database")
                object_name = str(input("Enter another name : \n"))
                args.name = object_name
                print(f"\nCollecting data for {colored(object_name, 'magenta')}")
        catalog_path, catalog_name = i_f.choose_catalog(args.catalog)
    elif args.coord:
        ra, dec = args.coord
        while True:
            print(f"\nCollecting data for coord : {colored([ra, dec], 'magenta')}")
            try:
                object_name = Simbad.query_region(f"{ra}d {dec}d", radius="1s")['MAIN_ID'][0]
                print(f"{colored([ra, dec], 'green')} is in Simbad Database, here is his name :\n{object_name}")
                break
            except Exception as error:
                print(f"{colored([ra, dec], 'red')} isn't Simbad Database")
                new_coord = str(input("Enter new coordinates : ra dec\n"))
                ra, dec = new_coord.split()
        object_position = i_f.get_coord_psr(object_name)
        catalog_path, catalog_name = i_f.choose_catalog(args.catalog)
        
    while True:
        if object_name in psr_name:
            count_rate = psr_table["psr count rate"][psr_table['full psr name'] == object_name][0]
            break
        else:
            try:
                count_rate = float(input("Enter the count rate of your object : \n"))
                break
            except ValueError as error:
                print(f"Error: {error}")
                print("Please enter a valid float value for Count Rate.")
                continue

    # ------------------------------------------------- #

    # --------------- object_data --------------- #

    object_data = {"object_name": object_name,
                "object_position": object_position,
                "count_rate": count_rate}

    # ------------------------------------------- #

    # --------------- modeling file --------------- #

    # get the active workflow path
    active_workflow = os.getcwd()
    active_workflow = active_workflow.replace("\\","/")
    data_path = os.path.join(active_workflow, "data").replace("\\", "/")

    # catalog_data_path
    catalog_datapath = os.path.join(data_path, "catalog_data").replace("\\", "/")

    # path of stilts and topcat software 
    stilts_software_path = os.path.join(data_path, 'softwares/stilts.jar').replace("\\", "/")
    topcat_software_path = os.path.join(data_path, 'softwares/topcat-extra.jar').replace("\\", "/")

    result_path = os.path.join(active_workflow, "modeling_result")
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    # creation of modeling file 
    name = object_data['object_name'].replace(" ", "_")
    modeling_file_path = os.path.join(active_workflow, 'modeling_result', name).replace("\\", "/")

    if not os.path.exists(modeling_file_path):
        os.mkdir(modeling_file_path)

    # creation of plot_var_sources
    plot_var_sources_path = os.path.join(modeling_file_path, "plot_var_sources").replace("\\", "/")
    if not os.path.exists(plot_var_sources_path):
        os.mkdir(plot_var_sources_path)

    output_name = os.path.join(modeling_file_path, 'Pointings').replace("\\", "/")
    if not os.path.exists(output_name):
        os.mkdir(output_name)

    # --------------------------------------------- #

    # --------------- User table --------------- #

    add_source_table = i_f.add_source_list(active_workflow=active_workflow)

    if len(add_source_table) != 0:
        colnames = ['Name', 'Right Ascension', 'Declination', 'Var Value']
        print("\nHere is the list given by the User : \n", add_source_table, "\n")
    else:
        print("\nUser don't defined any additionnal sources. \n")

    # ------------------------------------------ #

    # --------------- Load Nicer parameters --------------- #

    print('-'*50)
    print(f"{colored('Load NICER parameters : ', 'magenta')}")
    nicer_data_path = os.path.join(data_path, "NICER_data")
    PSF_data = os.path.join(nicer_data_path, "NICER_PSF.dat")
    ARF_data = os.path.join(nicer_data_path, "nixtiaveonaxis20170601v005.arf")
    RMF_data = os.path.join(nicer_data_path, "nixtiref20170601v003.rmf")
    nicer_parameters_path = i_f.get_valid_file_path(PSF_data)
    nicer_data_arf = i_f.get_valid_file_path(ARF_data)
    nicer_data_rmf = i_f.get_valid_file_path(RMF_data)
    EffArea, OffAxisAngle = np.loadtxt(nicer_parameters_path, unpack=True, usecols=(0, 1))
    print('-'*50, '\n')

    telescop_data = {"telescop_name": "nicer",
                    "EffArea": EffArea,
                    "OffAxisAngle": OffAxisAngle,
                    "nicer_data_arf": nicer_data_arf,
                    "nicer_data_rmf": nicer_data_rmf,
                    "min_value": 0.3,
                    "max_value": 10.0,
                    "energy_band": "0.2-12.0"}

    # ----------------------------------------------------- #

    # --------------- simulation_data --------------- #

    simulation_data = {"object_data": object_data,
                       "telescop_data": telescop_data,
                       "INSTbkgd": 0.2,
                       "EXPtime": args.exp_time
                       }

    # ----------------------------------------------- #

    radius = args.radius*u.arcmin

    if catalog_name == "Xmm_DR13":
        # Find the optimal pointing point with the Xmm_DR13 catalog
        
        # creation of 4XMM_DR13 directory
        xmm_directory = os.path.join(modeling_file_path, '4XMM_DR13'.replace("\\", "/"))
        xmm_img = os.path.join(xmm_directory, 'img'.replace("\\", "/"))
        xmm_closest_catalog = os.path.join(xmm_directory, "closest_catalog")
        if not os.path.exists(xmm_directory):
            os.mkdir(xmm_directory)
            os.mkdir(xmm_img)
            os.mkdir(xmm_closest_catalog)
        
        os_dictionary = {"active_workflow": active_workflow,
                        "catalog_datapath": catalog_datapath,
                        "modeling_file_path": modeling_file_path,
                        "plot_var_sources_path": plot_var_sources_path,
                        "catalog_directory" : xmm_directory,
                        "cloesest_dataset_path": xmm_closest_catalog,
                        "img": xmm_img,
                        "stilts_software_path": stilts_software_path,
                        "topcat_software_path": topcat_software_path}
        
        simulation_data["os_dictionary"] = os_dictionary
        
        # call XmmCatalog Class to make modeling
        xmm = XmmCatalog(catalog_path=catalog_path, radius=radius, simulation_data=simulation_data, user_table=add_source_table)
        nearby_sources_table, nearby_sources_position = xmm.nearby_sources_table,  xmm.nearby_sources_position
        model_dictionary = xmm.model_dictionary
        
        key = "XMM"
        column_dictionary = {"band_flux_obs" : dict_cat.dictionary_catalog['XMM']["band_flux_obs"],
                            "band_flux_obs_err": dict_cat.dictionary_catalog["XMM"]["band_flux_obs_err"],
                            "energy_band": [0.35, 0.75, 1.5, 3.25, 8.25],
                            "sigma": np.array(list(np.linspace(1e-20, 1e-24, len(dict_cat.dictionary_catalog[key]["energy_band_center"])))),
                            "data_to_vignetting": ["SC_RA", "SC_DEC", "IAUNAME"]}
        
        simulation_data["os_dictionary"]["catalog_key"] = key
        
    elif catalog_name == "CSC_2.0":
        # Find the optimal pointing point with the Chandra catalog
        
        # creation of Chandra directory
        chandra_directory = os.path.join(modeling_file_path, 'Chandra'.replace("\\", "/"))
        chandra_img = os.path.join(chandra_directory, 'img'.replace("\\", "/"))
        chandra_closest_catalog = os.path.join(chandra_directory, "closest_catalog")
        if not os.path.exists(chandra_directory):
            os.mkdir(chandra_directory)
            os.mkdir(chandra_img)
            os.mkdir(chandra_closest_catalog)
        
        os_dictionary = {"active_workflow": active_workflow,
                        "modeling_file_path": modeling_file_path,
                        "plot_var_sources_path": plot_var_sources_path,
                        "catalog_directory": chandra_directory,
                        "cloesest_dataset_path": chandra_closest_catalog,
                        "img": chandra_img,
                        "stilts_software_path": stilts_software_path,
                        "topcat_software_path": topcat_software_path}
        
        simulation_data["os_dictionary"] = os_dictionary
        
                        # cs = cone search (Harvard features)
        # call Chandra Class to make modeling
        csc = ChandraCatalog(catalog_path=catalog_path, radius=radius, simulation_data=simulation_data, user_table=add_source_table)
        table_1, sources_1 = csc.nearby_sources_table, csc.nearby_sources_position
        table_2, sources_2 = csc.cone_search_catalog, csc.cs_nearby_sources_position
        
        answer = str(input(f"Which Table do you chose to follow the modeling ? {colored('Chandra / CS_Chandra', 'magenta')}\n"))
        while True:
            if answer == "Chandra":
                key = "Chandra"
                nearby_sources_table, nearby_sources_position = table_1, sources_1
                column_dictionary = {"band_flux_obs": dict_cat.dictionary_catalog[key]["band_flux_obs"],
                                    "band_flux_obs_err": dict_cat.dictionary_catalog[key]["band_flux_obs_err"],
                                    "energy_band": dict_cat.dictionary_catalog[key]["energy_band_center"],
                                    "sigma": np.array(list(np.linspace(1e-20, 1e-24, len(dict_cat.dictionary_catalog[key]["energy_band_center"])))),
                                    "data_to_vignetting": ["RA", "DEC", "Chandra_IAUNAME"]}
                model_dictionary = csc.model_dictionary
                simulation_data["os_dictionary"]["catalog_key"] = key
                break
            elif answer == "CS_Chandra":
                key = "CS_Chandra"
                nearby_sources_table, nearby_sources_position = table_2, sources_2
                column_dictionary = {"band_flux_obs": dict_cat.dictionary_catalog[key]["band_flux_obs"],
                                    "band_flux_obs_err": dict_cat.dictionary_catalog[key]["band_flux_obs_err"],
                                    "energy_band": dict_cat.dictionary_catalog[key]["energy_band_center"],
                                    "sigma": np.array(list(np.linspace(1e-20, 1e-24, len(dict_cat.dictionary_catalog[key]["energy_band_center"])))),
                                    "data_to_vignetting": ["ra", "dec", "name"]}
                model_dictionary = csc.cs_model_dictionary
                simulation_data["os_dictionary"]["catalog_key"] = key
                break
            else:
                print(f"{colored('Key error ! ', 'red')}. Please retry !")
                answer = str(input(f"Which Table do you chose to follow the modeling ? {colored('Chandra / CS_Chandra', 'magenta')}\n"))
        
    elif catalog_name == "Swift":
        # Find the optimal pointing point with the Swift catalog
        
        # creation of Swift directory
        swi_directory = os.path.join(modeling_file_path, 'Swift'.replace("\\", "/"))
        swi_img = os.path.join(swi_directory, 'img'.replace("\\", "/"))
        swi_closest_catalog = os.path.join(swi_directory, "closest_catalog")
        if not os.path.exists(swi_directory):
            os.mkdir(swi_directory)
            os.mkdir(swi_img)
            os.mkdir(swi_closest_catalog)
        
        os_dictionary = {"active_workflow": active_workflow,
                        "modeling_file_path": modeling_file_path,
                        "plot_var_sources_path": plot_var_sources_path,
                        "catalog_directory" : swi_directory,
                        "cloesest_dataset_path": swi_closest_catalog,
                        "img": swi_img,
                        "stilts_software_path": stilts_software_path,
                        "topcat_software_path": topcat_software_path}
        
        simulation_data["os_dictionary"] = os_dictionary
        
        # call Swift Class to make modeling
        swi = SwiftCatalog(catalog_path=catalog_path, radius=radius, simulation_data=simulation_data, user_table=add_source_table)
        nearby_sources_table, nearby_sources_position = swi.nearby_sources_table, swi.nearby_sources_position
        model_dictionary = swi.model_dictionary
        
        key = "Swift"
        column_dictionary = {"band_flux_obs": dict_cat.dictionary_catalog[key]["band_flux_obs"],
                            "band_flux_obs_err": dict_cat.dictionary_catalog[key]["band_flux_obs_err"],
                            "energy_band": dict_cat.dictionary_catalog[key]["energy_band_center"],
                            "sigma": np.array(list(np.linspace(1e-20, 1e-24, len(dict_cat.dictionary_catalog[key]["energy_band_center"])))),
                            "data_to_vignetting": ["RA", "DEC", "Swift_IAUNAME"]}
        
        simulation_data["os_dictionary"]["catalog_key"] = key
        
    elif catalog_name == "eRosita":
        # Find the optimal pointing with the eRosita catalog
        
        # creation of eRosita directory
        eRo_directory = os.path.join(modeling_file_path, 'eRosita'.replace("\\", "/"))
        eRo_img = os.path.join(eRo_directory, 'img'.replace("\\", "/"))
        eRo_closest_catalog = os.path.join(eRo_directory, "closest_catalog")
        if not os.path.exists(eRo_directory):
            os.mkdir(eRo_directory)
            os.mkdir(eRo_img)
            os.mkdir(eRo_closest_catalog)
            
        os_dictionary = {"active_workflow": active_workflow,
                        "modeling_file_path": modeling_file_path,
                        "plot_var_sources_path": plot_var_sources_path,
                        "catalog_directory" : eRo_directory,
                        "cloesest_dataset_path": eRo_closest_catalog,
                        "img": eRo_img,
                        "stilts_software_path": stilts_software_path,
                        "topcat_software_path": topcat_software_path}
        
        # call eRosita Class to make modeling
        eRo = eRositaCatalog(catalog_path=catalog_path, radius=radius, simulation_data=simulation_data, user_table=add_source_table)
        nearby_sources_table, nearby_sources_position = eRo.nearby_sources_table, eRo.nearby_sources_position
        model_dictionary = eRo.model_dictionary
        
        key = "eRosita"
        column_dictionary = {"band_flux_obs": dict_cat.dictionary_catalog[key]["band_flux_obs"],
                            "band_flux_obs_err": dict_cat.dictionary_catalog[key]["band_flux_obs_err"],
                            "energy_band": dict_cat.dictionary_catalog[key]["energy_band_center"],
                            "sigma": np.array(list(np.linspace(1e-20, 1e-24, len(dict_cat.dictionary_catalog[key]["energy_band_center"])))),
                            "data_to_vignetting": ["RA", "DEC", "Swift_IAUNAME"]}
        
        simulation_data["os_dictionary"]["catalog_key"] = key
        
    elif catalog_name == "match":
        # Find optimal pointing point with using two catalog Xmm and Chandra
        
        # creation of match directory
        mixed_directory = os.path.join(modeling_file_path, 'xmmXchandra'.replace("\\", "/"))
        mixed_img = os.path.join(mixed_directory, 'img'.replace("\\", "/"))
        mixed_closest_catalog = os.path.join(mixed_directory, "closest_catalog")
        if not os.path.exists(mixed_directory):
            os.mkdir(mixed_directory)
            os.mkdir(mixed_img)
            os.mkdir(mixed_closest_catalog)
        
        os_dictionary = {"active_workflow": active_workflow,
                        "data_path": data_path,
                        "plot_var_sources_path": plot_var_sources_path,
                        "catalog_datapath": catalog_datapath,
                        "stilts_software_path": stilts_software_path,
                        "topcat_software_path": topcat_software_path,
                        "output_name": output_name,
                        "modeling_file_path": modeling_file_path,
                        "catalog_directory": mixed_directory,
                        "cloesest_dataset_path": mixed_closest_catalog,
                        "img": mixed_img}
        
        simulation_data["os_dictionary"] = os_dictionary
        os_dictionary["catalog_key"] = "xmmXchandra"
        
        # call CatalogMatch Class to make modeling
        mixed_catalog = MatchCatalog(catalog_name=("Xmm_DR13", "Chandra"), radius=radius, simulation_data=simulation_data)
        nearby_sources_table = mixed_catalog.nearby_sources_table
        var_index = mixed_catalog.var_index
        
        # --------------- modeling spectra with jaxspec --------------- #

        # setup jaxspec
        config.update("jax_enable_x64", True)
        numpyro.set_platform("cpu")

        # define caracteristic model here --> exp(-nh*$\sigma$) * x ** (-$\Gamma$)
        model = Tbabs() * Powerlaw()

        # load instrument parameters
        instrument = Instrument.from_ogip_file(nicer_data_arf, nicer_data_rmf, exposure=args.exp_time)

        # load all of the sources spetcra
        total_spectra, total_var_spectra = j_f.modeling_source_spectra(nearby_sources_table=nearby_sources_table, instrument=instrument, model=model, var_index=var_index)

        # plot of all spectra data
        data = j_f.total_plot_spectra(total_spectra=total_spectra, total_var_spectra=total_var_spectra, instrument=instrument, simulation_data=simulation_data, catalog_name="xmmXchandra")

        # output spectre plot
        j_f.write_txt_file(simulation_data=simulation_data, data=data)
        
        # ------------------------------------------------------------- # 
        
        sys.exit()

    elif catalog_name == "compare_catalog":
        # Find the optimal pointing point with two catalogs to compare data
        
        # creation of compare_catalog directory
        compare_catalog_directory = os.path.join(modeling_file_path, 'Compare_catalog'.replace("\\", "/"))
        compare_catalog_img = os.path.join(compare_catalog_directory, 'img'.replace("\\", "/"))
        compare_catalog_closest_catalog = os.path.join(compare_catalog_directory, "closest_catalog")
        if not os.path.exists(compare_catalog_directory):
            os.mkdir(compare_catalog_directory)
            os.mkdir(compare_catalog_img)
            os.mkdir(compare_catalog_closest_catalog)
        
        os_dictionary = {"active_workflow": active_workflow,
                        "data_path": data_path,
                        "modeling_file_path": modeling_file_path,
                        "catalog_datapath": catalog_datapath,
                        "output_name": output_name,
                        "plot_var_sources_path": plot_var_sources_path,
                        "stilts_software_path": stilts_software_path,
                        "catalog_directory": compare_catalog_directory,
                        "cloesest_dataset_path": compare_catalog_closest_catalog,
                        "img": compare_catalog_img}
        
        simulation_data["os_dictionary"] = os_dictionary
        
        # call CompareCatalog Class to make calculation
        compare_class = CompareCatalog(catalog_path=catalog_path, radius=radius, simulation_data=simulation_data, exp_time=args.exp_time)
        sys.exit()
        
    else:
        print(f"{colored('Invalid key workd !', 'red')}")
        sys.exit()
        
    # --------------- count_rates --------------- #

    excel_data_path = os.path.join(data_path, 'excel_data').replace("\\", "/")
    if not os.path.exists(excel_data_path):
        os.mkdir(excel_data_path)
        
    if platform.system() != "Windows":
        count_rates, nearby_sources_table = c_f.count_rates(nearby_sources_table, model_dictionary, telescop_data)
        # i_f.py_to_xlsx(excel_data_path=excel_data_path, count_rates=count_rates, object_data=object_data, args=(args.catalog, key), radius=args.radius)
    elif platform.system() == "Windows":
        count_rates, nearby_sources_table = i_f.xlsx_to_py(excel_data_path=excel_data_path, nearby_sources_table=nearby_sources_table, object_data=object_data, args=(args.catalog, key), radius=args.radius)
    else:
        sys.exit()
        
    simulation_data['nearby_sources_table'] = nearby_sources_table

    # -------------------------------------------------- #

    # --------------- Nominal pointing infos --------------- #
                
    c_f.nominal_pointing_info(simulation_data, nearby_sources_position)

    # ------------------------------------------------------ #

    # --------------- Value of optimal pointing point and infos --------------- #

                
    OptimalPointingIdx, SRCoptimalSEPAR, SRCoptimalRATES, vector_dictionary = c_f.calculate_opti_point(simulation_data, nearby_sources_position)

    c_f.optimal_point_infos(vector_dictionary, OptimalPointingIdx, SRCoptimalRATES)

    # ------------------------------------------------------------------------- #

    # --------------- Visualized data Matplotlib with S/N --------------- #

    c_f.data_map(simulation_data, vector_dictionary, OptimalPointingIdx, nearby_sources_position)

    # ------------------------------------------------------------------- #

    # --------------- Calculate vignetting factor --------------- #

    vignetting_factor, nearby_sources_table = c_f.vignetting_factor(OptimalPointingIdx=OptimalPointingIdx, vector_dictionary=vector_dictionary, simulation_data=simulation_data, data=column_dictionary["data_to_vignetting"], nearby_sources_table=nearby_sources_table)

    # ----------------------------------------------------------- #

    # --------------- Modeling nearby sources --------------- #

    c_f.modeling(vignetting_factor=vignetting_factor, simulation_data=simulation_data, column_dictionary=column_dictionary, catalog_name=args.catalog)

    # ------------------------------------------------------- #

    # --------------- write fits file --------------- #

    c_f.write_fits_file(nearby_sources_table=nearby_sources_table, simulation_data=simulation_data)

    # ----------------------------------------------- #

    # --------------- software --------------- # 

    master_source_path = os.path.join(catalog_datapath, 'Master_source.fits').replace("\\", "/")


    def select_master_sources_around_region(ra, dec, radius, output_name):
        """Radius is in arcminutes"""
        print(f"Extracting sources around region: RA {ra} and Dec {dec}")
        master_cone_path = os.path.join(output_name, 'Master_source_cone.fits').replace("\\", "/")
        command = (f"java -jar {stilts_software_path} tpipe {master_source_path} cmd='"+
                f'select skyDistanceDegrees({ra},{dec},MS_RA,MS_DEC)*60<{radius} '+
                f"' out={master_cone_path}")
        command = shlex.split(command)
        subprocess.run(command)


    def select_catalogsources_around_region(output_name):
        print('Selecting catalog sources')
        master_cone_path = os.path.join(output_name, 'Master_source_cone.fits').replace("\\", "/")
        for cat in catalogs:
            path_to_cat_init = os.path.join(catalog_datapath, cat).replace("\\", "/")
            path_to_cat_final = os.path.join(output_name, cat).replace("\\", "/")
            command = (f"java -jar {stilts_software_path} tmatch2 matcher=exact \
                    in1='{master_cone_path}' in2='{path_to_cat_init}.fits' out='{path_to_cat_final}.fits'\
                        values1='{cat}' values2='{cat}_IAUNAME' find=all progress=none")
            command = shlex.split(command)
            subprocess.run(command)

    right_ascension = object_data["object_position"].ra.value
    declination = object_data["object_position"].dec.value
    try:
        print(f"\n{colored('Load Erwan s code for :', 'yellow')} {object_data['object_name']}")
        select_master_sources_around_region(ra=right_ascension, dec=declination, radius=radius.value, output_name=output_name)
        select_catalogsources_around_region(output_name=output_name)
        master_sources = s_f.load_master_sources(output_name)
        s_f.master_source_plot(master_sources=master_sources, simulation_data=simulation_data, number_graph=len(master_sources))
    except Exception as error :
        print(f"{colored('An error occured : ', 'red')} {error}")

    # ---------------------------------------- #

    # --------------- modeling spectra with jaxspec --------------- #

    var_index =  j_f.cross_catalog_index(output_name=output_name, key=key, iauname=column_dictionary["data_to_vignetting"][2], nearby_sources_table=nearby_sources_table)

    # setup jaxspec
    config.update("jax_enable_x64", True)
    numpyro.set_platform("cpu")

    # define caracteristic model here --> exp(-nh*$\sigma$) * x ** (-$\Gamma$)
    model = Tbabs() * Powerlaw()

    # load instrument parameters
    instrument = Instrument.from_ogip_file(nicer_data_arf, nicer_data_rmf, exposure=args.exp_time)

    # load all of the sources spetcra
    total_spectra, total_var_spectra = j_f.modeling_source_spectra(nearby_sources_table=nearby_sources_table, instrument=instrument, model=model, var_index=var_index)

    # plot of all spectra data
    data = j_f.total_plot_spectra(total_spectra=total_spectra, total_var_spectra=total_var_spectra, instrument=instrument, simulation_data=simulation_data, catalog_name=args.catalog)

    # output spectre plot
    j_f.write_txt_file(simulation_data=simulation_data, data=data)

    # ------------------------------------------------------------- # 

if __name__ == "__main__":
    main()