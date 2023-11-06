# --------------- Packages --------------- #

from catalog_class import *
from astropy import units as u
from astropy.table import Table
from termcolor import colored
from astroquery.simbad import Simbad

import argparse
import function as f
import numpy as np
import sys

# ---------------------------------------- #

# --------------- Initialization --------------- #

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

parser.add_argument('--catalog', '-ca', type=str, 
                    help="Enter catalog keyword : Xmm_DR13/CSC_2.0/Swift/eRosita/compare_catalog")

args = parser.parse_args()

psr_name = np.array(["PSR J0437-4715", "PSR J2124-3358", "PSR J0751+1807", "PSR J1231-1411"], dtype=str)
psr_coord = np.array([f"{f.get_coord_psr(name).ra} {f.get_coord_psr(name).dec}" for name in psr_name])
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
            print(f"Collecting data for {colored(object_name, 'magenta')}")
        try:
            object_position = f.get_coord_psr(object_name)
            print(f"{colored(object_name, 'green')} is in Simbad Database, here is his coordinate :\n{object_position}")
            break
        except Exception as error:
            print(f"Error : {colored(object_name, 'red')}, isn't in Simbad Database")
            object_name = str(input("Enter another name : \n"))
            args.name = object_name
            print(f"Collecting data for {colored(object_name, 'magenta')}")
    catalog_path, catalog_name = f.choose_catalog(args.catalog)
elif args.coord:
    ra, dec = args.coord
    while True:
        print(f"Collecting data for coord : {colored([ra, dec], 'magenta')}")
        try:
            object_name = Simbad.query_region(f"{ra}d {dec}d", radius="1s")['MAIN_ID'][0]
            print(f"{colored([ra, dec], 'green')} is in Simbad Database, here is his name :\n{object_name}")
            break
        except Exception as error:
            print(f"{colored([ra, dec], 'red')} isn't Simbad Database")
            new_coord = str(input("Enter new coordinates : ra dec\n"))
            ra, dec = new_coord.split()
    object_position = f.get_coord_psr(object_name)
    catalog_path, catalog_name = f.choose_catalog(args.catalog)
    
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

# --------------- User table --------------- #

user_list = f.define_sources_list() 

if len(user_list) != 0:
    colnames = ['Name', 'Right Ascension', 'Declination', 'Var Value']
    user_table = Table(rows=user_list, names=colnames)
    print("Here is the list given by the User : \n", user_table, "\n")
else:
    user_table = Table()
    print("User don't defined any additionnal sources. \n")

# ------------------------------------------ #


# --------------- Load Nicer parameters --------------- #

print('-'*50)
nicer_parameters_path = f.get_valid_file_path("data/NICER_PSF.dat")
EffArea, OffAxisAngle = np.loadtxt(nicer_parameters_path, unpack=True, usecols=(0, 1))
print('-'*50)

telescop_data = {"telescop_name": "nicer",
                 "EffArea": EffArea,
                 "OffAxisAngle": OffAxisAngle,
                 "min_value": 0.3,
                 "max_value": 10.0,
                 "energy_band": "0.2-12.0"}

# ----------------------------------------------------- #

# --------------- Useful dictionary --------------- #

object_data = {"object_name": object_name,
               "object_position": object_position,
               "count_rate": count_rate}

simulation_data = {"object_data": object_data,
                   "telescop_data": telescop_data,
                   "INSTbkgd": 0.2,
                   "EXPtime": 1e6
                   }

# ------------------------------------------------- #

radius = args.radius*u.arcmin

if args.catalog == "Xmm_DR13":
    # Find the optimal pointing point with the Xmm_DR13 catalog
    xmm = XmmCatalog(catalog_path=catalog_path, radius=radius, dictionary=object_data, user_table=user_table)
    nearby_sources_table, nearby_sources_position = xmm.nearby_sources_table,  xmm.nearby_sources_position
    model_dictionary = xmm.model_dictionary
elif args.catalog == "CSC_2.0":
    # Find the optimal pointing point with the Chandra catalog
                    # cs = cone search (Harvard features)
    csc = Chandra(catalog_path=catalog_path, radius=radius, dictionary=object_data, user_table=user_table)
    # nearby_sources_table, nearby_sources_position = csc.nearby_sources_table, csc.nearby_sources_position
    cs_nearby_soucres_table, cs_nearby_sources_position = csc.cs_nearby_sources_table, csc.cs_nearby_sources_position
    nearby_sources_table, nearby_sources_position = cs_nearby_soucres_table, cs_nearby_sources_position
    # nearby_soucres_table = csc.cone_search_catalog.to_table()
    model_dictionary = csc.model_dictionary
elif args.catalog == "Swift":
    # Find the optimal pointing point with the Swift catalog
    swi = Swift(catalog_path=catalog_path, radius=radius, dictionary=object_data, user_table=user_table)
    nearby_sources_table, nearby_sources_position = swi.nearby_sources_table, swi.nearby_sources_position
    model_dictionary = swi.dictionary_model
elif args.catalog == "eRosita":
    # Find the optimal pointing with the eRosita catalog
    eRo = eRosita(catalog_path=catalog_path, radius=radius, dictionary=object_data, user_table=user_table)
    nearby_sources_table, nearby_sources_position = eRo.nearby_sources_table, eRo.nearby_sources_position
    model_dictionary = eRo.dictionary_model
elif args.catalog == "compare_catalog":
    # Find the optimal pointing point with two catalogs to compare data
    compare_class = CompareCatalog(catalog_path=catalog_path, radius=radius, dictionary=object_data, user_table=user_table)
    compare_class.opti_point_calcul(simulation_data=simulation_data)
    sys.exit()
    
# -------------------------------------------------- #

count_rates, nearby_soucres_table = f.count_rates(nearby_sources_table, model_dictionary, telescop_data)
simulation_data['nearby_soucres_table'] = nearby_soucres_table

# -------------------------------------------------- #

# -------------------------------------------------- #

                # Nominal pointing infos
            
f.nominal_pointing_info(simulation_data, nearby_sources_position)

# -------------------------------------------------- #

# -------------------------------------------------- #

                # Value of optimal pointing point and infos
            
OptimalPointingIdx, SRCoptimalSEPAR, SRCoptimalRATES, vector_dictionary = f.calculate_opti_point(simulation_data, nearby_sources_position)

f.optimal_point_infos(vector_dictionary, OptimalPointingIdx, SRCoptimalRATES)

# -------------------------------------------------- #

# -------------------------------------------------- #

                # Visualized data Matplotlib with S/N

f.data_map(simulation_data, vector_dictionary, OptimalPointingIdx, nearby_sources_position)

# -------------------------------------------------- #