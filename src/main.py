# -------------------------------------------------- #

from CatalogClass import XmmCatalog, Xmm2Athena, Chandra, Swift, ERosita
import Function as F
import argparse
from astropy.table import Table
from astroquery.simbad import Simbad
import numpy as np
from astropy import units as u 
       
# -------------------------------------------------- #

                # PSR table

psr_full_name   = ['PSR J0437-4715','PSR J2124-3358','PSR J0751+1807', 'PSR J1231-1411']
psr_short_name = [F.name_to_short_name(NAME) for NAME in psr_full_name]
psr_coord_ra = [F.get_coord_psr(NAME).ra for NAME in psr_full_name]
psr_coord_dec = [F.get_coord_psr(NAME).dec for NAME in psr_full_name]
psr_count_rates  = [1.319,0.1,0.025, 0.27]
psr_table = Table([psr_full_name, psr_short_name, psr_coord_ra, psr_coord_dec, psr_count_rates],
                 names=('FullName', 'ShortName', 'RA', 'DEC','CountRate'))

# -------------------------------------------------- #

print("Code to find an optimal pointing point with NICER for an astrophysical object.")
print("Here is the PSR Table : \n", psr_table, "\n")

# -------------------------------------------------- #

                # Initialization Code   

parser = argparse.ArgumentParser(description='Search for information with object name or coord')

find_object = parser.add_mutually_exclusive_group()
find_object.add_argument('--name', type=str, help='Replace spaces by _')
find_object.add_argument('--coord', type=float, nargs=2, help='ra dec (two float values)')
parser.add_argument('--catalog', type=str, help='Xmm_DR13, CSC_2.0, Swift, eRosita')

args = parser.parse_args()

# -------------------------------------------------- #

if args.name:
    while True : 
        if '_' in args.name:
            NAME = args.name.replace("_", " ")
        print(f"Searching for information with name : {NAME}")
        try:
            OBJ_POSITION = F.get_coord_psr(NAME)
            print(OBJ_POSITION, '\n')
            break
        except Exception as error:
            print(f"Error: {NAME} isn't in Simbad Database")
            args.name = str(input("Enter another objet name : "))
    catalog_path, catalog_name = F.choose_catalog(args.catalog)
elif args.coord:
    ra, dec = args.coord
    RA, DEC = float(ra), float(dec)
    while True:
        print(f"Searching for information with coordinates : (RA={ra}, DEC={dec})")
        try:
            NAME = Simbad.query_region(f"{ra}d {dec}d", radius="1s")['MAIN_ID'][0]
            break
        except Exception as error:
            print(f"Error: There is no object with these coordinates (RA={RA}, DEC={DEC}) in the Simbad Database.")
            RA = float(input('Enter new right ascension : '))
            DEC =  float(input('Enter new declination'))
    OBJ_POSITION = F.get_coord_psr(NAME)
    catalog_path, catalog_name = F.choose_catalog(args.catalog)

while True:
    if NAME in psr_full_name:
        COUNT_RATE = psr_table['CountRate'][psr_table['FullName'] == NAME][0]
        break
    else:
        try:
            COUNT_RATE = float(input("Enter the count rate of your object: \n"))
            break
        except ValueError as error:
            print(f"Error: {error}")
            print("Please enter a valid float value for Count Rate.")
            continue

object_data = {'object_name':NAME,
               'CountRate': COUNT_RATE,
               'object_position' : OBJ_POSITION
               }

user_list = F.define_sources_list() 

if len(user_list) != 0:
    colnames = ['Name', 'Right Ascension', 'Declination', 'Var Value']
    user_table = Table(rows=user_list, names=colnames)
    print("Here is the list given by the User : \n", user_table, "\n")
else:
    user_table = Table()
    print("User don't defined any additionnal sources. \n")

# -------------------------------------------------- #
                # load nicer parameters

print('-'*50)
nicer_parameters_path = F.get_valid_file_path("Catalog/NICER_PSF.dat")
EffArea, OffAxisAngle = np.loadtxt("Catalog/NICER_PSF.dat", unpack=True, usecols=(0, 1))
print('-'*50)

                # code verification fichier nicer_parameter

# while True: 
#     with open(nicer_parameter_path, "r") as data:
#         first_row = data.readline()
#         column = first_row.split()
#         ncol = len(column)
        
#     try:
#         if ncol == 2:
#             EffArea, OffAxisAngle = np.loadtxt(nicer_parameter_path, unpack=True, usecols=(0, 1))
#             return EffArea, OffAxisAngle
#         else:
#             raise Exception(f"The file {nicer_parameter_path} doesn't have 2 columns")
#     except Exception as error:
#         print(f"An error occured {error}")
#         nicer_parameter_path = str(input('Input another file path : \n'))

# -------------------------------------------------- #

                # useful dictionary

telescop_data = {'telescop_name': 'nicer',
                 'EffArea':EffArea,
                 'OffAxisAngle': OffAxisAngle
                 }

simulation_data = {'object_data': object_data,
                   'telescop_data': telescop_data,
                   'INSTbkgd': 0.2,
                   'EXPtime': 1e6
                   }

if catalog_name == "Xmm_DR13":
    
    print('-'*50)
    xmm_dr11_path = F.get_valid_file_path('Catalog/4XMM_DR11cat_v1.0.fits')
    xmm_2_athena_path = F.get_valid_file_path('Catalog/xmm2athena_D6.1_V3.fits')
    print('-'*50)
    
    radius = 5*u.arcmin
    xmm = XmmCatalog(catalog_path=catalog_path, radius=radius, dictionary=object_data, user_table=user_table)
    nearby_src_table, nearby_src_position = xmm.nearby_src_table, xmm.nearby_src_position
    
    x2a = Xmm2Athena(xmm_dr11_path=xmm_dr11_path, xmm_2_athena_path=xmm_2_athena_path)
    
    value_list = [0.3, 10.0, '0.2-12.0']
    key_list = ['min_value', 'max_value', 'energy_band']
    for key, value in zip(key_list, value_list):
        telescop_data[key] = value
        
    catalog = {'current_catalog': xmm.catalog,
               'xmm_dr11':x2a.xmm_dr11,
               'xmm_2_ath':x2a.xmm_2_athena 
               }
    
    simulation_data['catalog'] = catalog
        
    nearby_src_table, index_table = x2a.add_nh_photon_index(nearby_src_table, user_table)
    variability_table = F.variability_rate(index_table, nearby_src_table, simulation_data)
    xmm.neighbourhood_of_object(nearby_src_table, variability_table, simulation_data)
    model_dictionary = xmm.model_dictionary()
        
elif catalog_name == "CSC_2.0":
    
    radius = 5*u.arcmin
    csc = Chandra(catalog_path=catalog_path, radius=radius, dictionary=object_data)
    # nearby_src_table, nearby_src_position = csc.nearby_src_table, csc.nearby_src_position
    nearby_src_table, nearby_src_position = csc.cone_search_catalog, csc.cone_search_src_position
    gamma = csc.powlaw_gamma()
    model_dictionary = csc.model_dictionary(gamma)
    
    value_list = [0.6, 6.9, '0.5-7.0']
    key_list = ['min_value', 'max_value', 'energy_band']
    for key, value in zip(key_list, value_list):
        telescop_data[key] = value
        
elif catalog_name == "Swift":
    radius = 8*u.arcmin
    swi = Swift(catalog_path=catalog_path, radius=radius, dictionary=object_data)
    nearby_src_table, nearby_src_position = swi.nearby_src_table, swi.nearby_src_position
    
elif catalog_name == "eRosita":
    radius = 8*u.arcmin
    ero = ERosita(catalog_path=catalog_path, radius=radius, dictionary=object_data)
    nearby_src_table, nearby_src_position = ero.nearby_src_table, ero.nearby_src_position
    
# -------------------------------------------------- #

count_rates, nearby_src_table = F.count_rates(nearby_src_table, model_dictionary, telescop_data)
simulation_data['NearbySRC_Table'] = nearby_src_table

# -------------------------------------------------- #

# -------------------------------------------------- #

                # Nominal pointing infos
            
F.nominal_pointing_info(simulation_data, nearby_src_position)

# -------------------------------------------------- #

# -------------------------------------------------- #

                # Value of optimal pointing point and infos
            
OptimalPointingIdx, SRCoptimalSEPAR, SRCoptimalRATES, vector_dictionary = F.calculate_opti_point(simulation_data, nearby_src_position)

F.optimal_point_infos(vector_dictionary, OptimalPointingIdx, SRCoptimalRATES)

# -------------------------------------------------- #

# -------------------------------------------------- #

                # Visualized data Matplotlib with S/N

F.data_map(simulation_data, vector_dictionary, OptimalPointingIdx, nearby_src_position)

# -------------------------------------------------- #