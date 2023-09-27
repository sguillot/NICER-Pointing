from CatalogClass import XmmCatalog, Xmm2Athena
import Function as F

# -------------------------------------------------- #

                # Arg_parser function

Object_data, PATH = F.initialization_code()

# -------------------------------------------------- #

NICER_parameters_path = F.get_valid_file_path("Catalog/NICER_PSF.dat")

XMM = XmmCatalog(PATH, NICER_parameters_path)

XMM_DR_13_CATALOG = XMM.catalog
EffArea, OffAxisAngle = XMM.NICER_parameters
SRCposition = XMM.SRCcoord

XMM_DR_11_path = F.get_valid_file_path('Catalog/4XMM_DR11cat_v1.0.fits')
XMM_2_ATHENA_path = F.get_valid_file_path('Catalog/xmm2athena_D6.1_V3.fits')

X2A = Xmm2Athena(XMM_DR_11_path, XMM_2_ATHENA_path)

# -------------------------------------------------- #
                # Useful dictionary

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

NearbySource = F.FindNearbySources(XMM_DR_13_CATALOG, SRCposition, Simulation_data['Object_data']['ObjectName'])
NearbySources_Table, Nearby_SRCposition = XMM.create_NearbySource_table(NearbySource, XMM_DR_13_CATALOG)
NearbySources_Table = X2A.add_nh_photon_index(NearbySources_Table=NearbySources_Table)

# -------------------------------------------------- #

# -------------------------------------------------- #

                # Visualized data Matplotlib without S/N

XmmCatalog.neighbourhood_of_object(NearbySources_Table, Simulation_data['Object_data'])

# -------------------------------------------------- #

# -------------------------------------------------- #

                # Count Rates and complete NearbySources_Table 
                
Count_Rates, NearbySources_Table = F.count_rates(NearbySources_Table, xmmflux=NearbySources_Table['SC_EP_8_FLUX'], NH=NearbySources_Table['Nh'], Power_Law=NearbySources_Table['Photon Index'])

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