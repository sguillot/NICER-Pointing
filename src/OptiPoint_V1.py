# -------------------------------------------------- #

from CatalogClass import XmmCatalog, Xmm2Athena
import Function as F
import argparse
from astropy.table import Table
from astroquery.simbad import Simbad
from astropy import units as u
       
# -------------------------------------------------- #

                # PSR table

PSRfullnames   = ['PSR J0437-4715','PSR J2124-3358','PSR J0751+1807', 'PSR J1231-1411']
PSRshortname = [F.Name_to_Short_Name(NAME) for NAME in PSRfullnames]
PSRcoordRA = [F.GetCoordPSR(NAME).ra for NAME in PSRfullnames]
PSRcoordDEC = [F.GetCoordPSR(NAME).dec for NAME in PSRfullnames]
PSRcountrates  = [1.319,0.1,0.025, 0.27]
PSRtable = Table([PSRfullnames, PSRshortname, PSRcoordRA, PSRcoordDEC, PSRcountrates],
                 names=('FullName', 'ShortName', 'RA', 'DEC','CountRate'))

# -------------------------------------------------- #

print("Code to find an optimal pointing point with NICER for an astrophysical object.")
print("Here is the PSR Table : \n", PSRtable, "\n")

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
            OBJposition = F.GetCoordPSR(NAME)
            print(OBJposition, '\n')
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
    OBJposition = F.GetCoordPSR(NAME)
    PATH = F.choose_catalog(args.catalog)

while True:
    if NAME in PSRfullnames:
        CountRate = PSRtable['CountRate'][PSRtable['FullName'] == NAME][0]
        break
    else:
        try:
            CountRate = float(input("Enter the count rate of your object: \n"))
            break
        except ValueError as error:
            print(f"Error: {error}")
            print("Please enter a valid float value for Count Rate.")
            continue
        
Object_data = {'ObjectName':NAME,
               #'ShortName': F.Name_to_Short_Name(NAME),
               'CountRate': CountRate,
               'OBJposition' : OBJposition}


UserList = F.define_sources_list() 

if len(UserList) != 0:
    colnames = ['Name', 'Right Ascension', 'Declination', 'Var Value']
    User_table = Table(rows=UserList, names=colnames)
    print("Here is the list given by the User : \n", User_table, "\n")
else:
    User_table = Table()
    print("User don't defined any additionnal sources. \n")

print('-'*50)
NICER_parameters_path = F.get_valid_file_path("Catalog/NICER_PSF.dat")
XMM_DR_11_path = F.get_valid_file_path('Catalog/4XMM_DR11cat_v1.0.fits')
XMM_2_ATHENA_path = F.get_valid_file_path('Catalog/xmm2athena_D6.1_V3.fits')
print('-'*50)

XMM = XmmCatalog(PATH, NICER_parameters_path)
XMM_DR_13_CATALOG = XMM.catalog
EffArea, OffAxisAngle = XMM.NICER_parameters

X2A = Xmm2Athena(XMM_DR_11_path, XMM_2_ATHENA_path)
XMMDR11 = X2A.XMM_DR11
XMM2ATH = X2A.XMM_2_ATHENA

# -------------------------------------------------- #
                # Useful dictionary

Catalog = {'CurrentCatalog': XMM_DR_13_CATALOG,
            'XMMDR11':XMMDR11,
            'XMM2ATH':XMM2ATH
            }

Telescop_data = {'TelescopeName': 'NICER',
                'EffArea': EffArea,
                'OffAxisAngle': OffAxisAngle
                }

Simulation_data = {'Object_data': Object_data,
                   'Telescop_data': Telescop_data,
                   'Catalog': Catalog,
                   'INSTbkgd': 0.2,
                   'EXPtime': 1e6
                   }


# -------------------------------------------------- #

try : 
    NearbySRC_Table, NearbySRCposition, Nbr_Var_SRC = XMM.NearbySourcesTable(Object_data, User_table, XMM_DR_13_CATALOG)
    if len(NearbySRC_Table) == 0:
        print(f"No sources detected close to {Object_data['ObjectName']}")
    else:
        print(f"We have detected {len(NearbySRC_Table)} sources close to {Object_data['ObjectName']}")
except Exception as error :
    print(f"An error occured : {error}")
    
# -------------------------------------------------- #

NearbySRC_Table, INDEX_ATH = X2A.add_nh_photon_index(NearbySRC_Table, User_table)

VAR_SRC_Table = F.variability_rate(NearbySRC_Table, Simulation_data, INDEX_ATH, Nbr_Var_SRC)

# -------------------------------------------------- #

# -------------------------------------------------- #
                # Visualized data Matplotlib without S/N

XMM.neighbourhood_of_object(NearbySourcesTable=NearbySRC_Table, Object_data=Simulation_data['Object_data'], VAR_SRC_Table=VAR_SRC_Table)

# -------------------------------------------------- #

# -------------------------------------------------- #

                # Count Rates and complete NearbySources_Table 
                
Count_Rates, NearbySRC_Table = F.count_rates(NearbySRC_Table, xmmflux=NearbySRC_Table['SC_EP_8_FLUX'], NH=NearbySRC_Table['Nh'], Power_Law=NearbySRC_Table['Photon Index'])

Simulation_data['NearbySRC_Table'] = NearbySRC_Table

# -------------------------------------------------- #

# -------------------------------------------------- #

                # Nominal pointing infos
                
F.NominalPointingInfo(Simulation_data, NearbySRCposition)

# -------------------------------------------------- #

# -------------------------------------------------- #

                # Value of optimal pointing point and infos
                
OptimalPointingIdx, SRCoptimalSEPAR, SRCoptimalRATES, Vector_Dictionary = F.CalculateOptiPoint(Simulation_data, NearbySRCposition)

F.OptimalPointInfos(Vector_Dictionary, OptimalPointingIdx, SRCoptimalRATES)

# -------------------------------------------------- #

# -------------------------------------------------- #

                # Visualized data Matplotlib with S/N

F.DataMap(Simulation_data, Vector_Dictionary, OptimalPointingIdx, NearbySRCposition)

# -------------------------------------------------- #