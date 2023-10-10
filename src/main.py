# -------------------------------------------------- #

from CatalogClass import XmmCatalog, Xmm2Athena
import Function as F
import argparse
from astropy.table import Table
from astroquery.simbad import Simbad
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
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--name', type=str, help='Replace spaces by _')
group.add_argument('--coord', type=float, nargs=2, help='ra dec (two float values)')
parser.add_argument("--catalog", type=str, help="Enter a catalog keyword DR11/DR13.")
args = parser.parse_args()

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
    PATH = F.choose_catalog(args.catalog)
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
    PATH = F.choose_catalog(args.catalog)

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
        
object_data = {'ObjectName':NAME,
               #'ShortName': F.Name_to_Short_Name(NAME),
               'CountRate': COUNT_RATE,
               'OBJposition' : OBJ_POSITION}


user_list = F.define_sources_list() 

if len(user_list) != 0:
    colnames = ['Name', 'Right Ascension', 'Declination', 'Var Value']
    user_table = Table(rows=user_list, names=colnames)
    print("Here is the list given by the User : \n", user_table, "\n")
else:
    user_table = Table()
    print("User don't defined any additionnal sources. \n")

print('-'*50)
NICER_PARAMETERS_PATH = F.get_valid_file_path("Catalog/NICER_PSF.dat")
XMM_DR_11_PATH = F.get_valid_file_path('Catalog/4XMM_DR11cat_v1.0.fits')
XMM_2_ATHENA_PATH = F.get_valid_file_path('Catalog/xmm2athena_D6.1_V3.fits')
print('-'*50)

XMM = XmmCatalog(PATH, NICER_PARAMETERS_PATH)
XMM_DR_13 = XMM.catalog
EffArea, OffAxisAngle = XMM.NICER_PARAMETERS

X2A = Xmm2Athena(XMM_DR_11_PATH, XMM_2_ATHENA_PATH)
XMMDR11 = X2A.XMM_DR11
XMM2ATH = X2A.XMM_2_ATHENA

# -------------------------------------------------- #
                # Useful dictionary

catalog = {'CurrentCatalog': XMM_DR_13,
            'XMMDR11':XMMDR11,
            'XMM2ATH':XMM2ATH
            }

telescop_data = {'TelescopeName': 'NICER',
                'EffArea': EffArea,
                'OffAxisAngle': OffAxisAngle
                }

simulation_data = {'Object_data': object_data,
                   'Telescop_data': telescop_data,
                   'Catalog': catalog,
                   'INSTbkgd': 0.2,
                   'EXPtime': 1e6
                   }

# -------------------------------------------------- #

print("\n")
try : 
    nearby_src_table, nearby_src_position, nbr_var_src = XMM.nearby_sources_table(object_data, user_table, XMM_DR_13)
    if len(nearby_src_table) == 0:
        print(f"No sources detected close to {object_data['ObjectName']}")
    else:
        print(f"We have detected {len(nearby_src_table)} sources close to {object_data['ObjectName']}")
except Exception as error :
    print(f"An error occured : {error}")
    
# -------------------------------------------------- #

nearby_src_table, index_table = X2A.add_nh_photon_index(nearby_src_table=nearby_src_table, user_table=user_table)

variability_table = F.variability_rate(index_table, nearby_src_table, simulation_data)

# -------------------------------------------------- #

# -------------------------------------------------- #
                # Visualized data Matplotlib without S/N

XMM.neighbourhood_of_object(nearby_src_table, variability_table, simulation_data)

# # -------------------------------------------------- #

# # -------------------------------------------------- #

#                 # Count Rates and complete NearbySources_Table 
            
# Count_Rates, NearbySRC_Table = F.count_rates(nearby_src_table, xmmflux=nearby_src_table['SC_EP_8_FLUX'], NH=nearby_src_table['Nh'], Power_Law=nearby_src_table['Photon Index'])

# simulation_data['NearbySRC_Table'] = NearbySRC_Table

# # -------------------------------------------------- #

# # -------------------------------------------------- #

#                 # Nominal pointing infos
            
# F.nominal_pointing_info(simulation_data, nearby_src_position)

# # -------------------------------------------------- #

# # -------------------------------------------------- #

#                 # Value of optimal pointing point and infos
            
# OptimalPointingIdx, SRCoptimalSEPAR, SRCoptimalRATES, vector_dictionary = F.calculate_opti_point(simulation_data, nearby_src_position)

# F.optimal_point_infos(vector_dictionary, OptimalPointingIdx, SRCoptimalRATES)

# # -------------------------------------------------- #

# # -------------------------------------------------- #

#                 # Visualized data Matplotlib with S/N

# F.data_map(simulation_data, vector_dictionary, OptimalPointingIdx, nearby_src_position)

# # -------------------------------------------------- #