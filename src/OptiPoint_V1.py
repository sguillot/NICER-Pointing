from CatalogClass import XmmCatalog, Xmm2Athena
from astropy.table import Table
import Function as F

# -------------------------------------------------- #

                # Arg_parser function
             
# --------------- # 
import argparse
from astroquery.simbad import Simbad
# --------------- #      
          
PSRfullnames   = ['PSR J0437-4715','PSR J2124-3358','PSR J0751+1807', 'PSR J1231-1411']
PSRcoordRA = [F.GetCoordPSR(NAME).ra for NAME in PSRfullnames]
PSRcoordDEC = [F.GetCoordPSR(NAME).dec for NAME in PSRfullnames]
PSRcountrates  = [1.319,0.1,0.025, 0.27]
PSRtable = Table([PSRfullnames, PSRcoordRA, PSRcoordDEC, PSRcountrates],
                 names=('FullName', 'RA', 'DEC','Count Rate'))

parser = argparse.ArgumentParser(description='Search for information with object name or coord')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--name', type=str, help='Replace spaces by _')
group.add_argument('--coord', type=float, nargs=2, help='ra dec (two float values)')
group.add_argument('--info', help="Have some information about PSR")

args = parser.parse_args()

if args.name:
    NAME = args.name.replace("_", " ")
    print(f"Searching for information with name : {NAME}")
    try:
        OBJposition = F.GetCoordPSR(NAME)
        print(OBJposition)
    except Exception as error:
        print(f"Error: {NAME} isn't in Simbad Database")
elif args.coord:
    ra, dec = args.coord
    print(f"Searching for information with coordinates : (RA={ra}, DEC={dec})")
    try:
        NAME = Simbad.query_region(f"{ra}d {dec}d", radius="1s")['MAIN_ID'][0]
    except Exception as error:
        print(f"Error: There is no object with these coordinates (RA={ra}, DEC={dec}) in the Simbad Database.")
    OBJposition = F.GetCoordPSR(NAME)
elif args.info:
    print(PSRtable)

# CountRate = PSRtable['Count Rate'][PSRtable['FullName'] == NAME][0]
CountRate = 0.1
# -------------------------------------------------- #

# -------------------------------------------------- #

                # Choice function (wainting for improvement (arg_parser))

# NAME, OBJposition, CountRate = F.Choice()

# -------------------------------------------------- #


XMM_DR_13_path = "Catalog/4XMM_slim_DR13cat_v1.0.fits"
NICER_parameters_path = "Catalog/NICER_PSF.dat"

XMM = XmmCatalog(XMM_DR_13_path, NICER_parameters_path)

XMM_DR_13_CATALOG = XMM.catalog
EffArea, OffAxisAngle = XMM.NICER_parameters
SRCposition = XMM.SRCcoord

XMM_DR_11_path = 'Catalog/4XMM_DR11cat_v1.0.fits'
XMM_2_ATHENA_path = 'Catalog/xmm2athena_D6.1_V3.fits'

X2A = Xmm2Athena(XMM_DR_11_path, XMM_2_ATHENA_path)

# -------------------------------------------------- #
                # Useful dictionary

Object_data = {'ObjectName':NAME,
               'ShortName': F.Name_to_Short_Name(NAME),
               'CountRate': CountRate,
               'PSRposition' : OBJposition}

Telescop_data = {'TelescopeName': 'NICER',
                 'EffArea': EffArea,
                 'OffAxisAngle': OffAxisAngle}

Simulation_data = {'Object_data': Object_data,
                   'Telescop_data': Telescop_data,
                   'INSTbkgd': 0.2,
                   'EXPtime': 1e6,
                   }

print(Simulation_data)

# -------------------------------------------------- #

NearbySource = F.FindNearbySources(XMM_DR_13_CATALOG, SRCposition, Object_data['ObjectName'])
NearbySources_Table, Nearby_SRCposition = XMM.create_NearbySource_table(NearbySource, XMM_DR_13_CATALOG)
NearbySources_Table = X2A.add_nh_photo_index(NearbySources_Table=NearbySources_Table)

# -------------------------------------------------- #

# -------------------------------------------------- #

                # Visualized data Matplotlib without S/N

XmmCatalog.neighbourhood_of_object(NearbySources_Table, Object_data)

# -------------------------------------------------- #

# -------------------------------------------------- #

                # Count Rates and complete NearbySources_Table 
                
Count_Rates, NearbySources_Table = XMM.count_rates(NearbySources_Table)

Simulation_data['NearbySources_Table'] = NearbySources_Table

# -------------------------------------------------- #

# -------------------------------------------------- #

                # Nominal pointing infos
                
F.NominalPointingInfo(Simulation_data, Nearby_SRCposition)

# -------------------------------------------------- #

# -------------------------------------------------- #

                # Value of optimal pointing point and infos
                
OptimalPointingIdx, SRCoptimalSEPAR, SRCoptimalRATES, Vector_Dictionary = F.CalculateOptiPoint(Simulation_data, Nearby_SRCposition)

F.OptimalPointInfos(Vector_Dictionary, OptimalPointingIdx, SRCoptimalRATES)

# -------------------------------------------------- #

# -------------------------------------------------- #

                # Visualized data Matplotlib with S/N

F.DataMap(Simulation_data, Vector_Dictionary, OptimalPointingIdx, Nearby_SRCposition)

# -------------------------------------------------- #