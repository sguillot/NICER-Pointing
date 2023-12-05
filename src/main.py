# --------------- Packages --------------- #

from catalog_class import *
from astropy import units as u
from astropy.table import Table
from termcolor import colored
from astroquery.simbad import Simbad
from jaxspec.model.multiplicative import Tbabs
from jaxspec.model.additive import Powerlaw
from jax.config import config
from jaxspec.data.instrument import Instrument

import argparse
import function as f
import numpy as np
import os
import subprocess
import sys
import shlex
import catalog_information as dict_cat
import numpyro
import platform

# ---------------------------------------- #

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
            print(f"\nCollecting data for {colored(object_name, 'magenta')}")
        try:
            object_position = f.get_coord_psr(object_name)
            print(f"\n{colored(object_name, 'green')} is in Simbad Database, here is his coordinate :\n{object_position}")
            break
        except Exception as error:
            print(f"Error : {colored(object_name, 'red')}, isn't in Simbad Database")
            object_name = str(input("Enter another name : \n"))
            args.name = object_name
            print(f"\nCollecting data for {colored(object_name, 'magenta')}")
    catalog_path, catalog_name = f.choose_catalog(args.catalog)
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

# --------------- object_data --------------- #

object_data = {"object_name": object_name,
               "object_position": object_position,
               "count_rate": count_rate}

# ------------------------------------------- #

# --------------- modeling file --------------- #

# get the active workflow path
active_workflow = os.getcwd()
active_workflow = active_workflow.replace("\\","/")

# catalog_data_path
catalog_datapath = os.path.join(active_workflow, "catalog_data").replace("\\", "/")

# path of stilts software
stilts_software_path = os.path.join(active_workflow, 'softwares/stilts.jar').replace("\\", "/")

# creation of modeling file 
name = object_data['object_name'].replace(" ", "_")
modeling_file_path = os.path.join(active_workflow, 'modeling_result', name).replace("\\", "/")

if not os.path.exists(modeling_file_path):
    os.mkdir(modeling_file_path)

output_name = os.path.join(modeling_file_path, 'Pointings').replace("\\", "/")
if not os.path.exists(output_name):
    os.mkdir(output_name)

os_dictionary = {"modeling_file_path": modeling_file_path}

# --------------------------------------------- #

# --------------- User table --------------- #

add_source_table = f.add_source_list(active_workflow=active_workflow)

if len(add_source_table) != 0:
    colnames = ['Name', 'Right Ascension', 'Declination', 'Var Value']
    print("\nHere is the list given by the User : \n", add_source_table, "\n")
else:
    print("\nUser don't defined any additionnal sources. \n")

# ------------------------------------------ #

# --------------- Load Nicer parameters --------------- #

print('-'*50)
print(f"{colored('Load NICER parameters : ', 'magenta')}")
nicer_parameters_path = f.get_valid_file_path("NICER_data/NICER_PSF.dat")
nicer_data_arf = f.get_valid_file_path("NICER_data/nixtiaveonaxis20170601v005.arf")
nicer_data_rmf = f.get_valid_file_path("NICER_data/nixtiref20170601v003.rmf")
EffArea, OffAxisAngle = np.loadtxt(nicer_parameters_path, unpack=True, usecols=(0, 1))
print('-'*50, '\n')

telescop_data = {"telescop_name": "nicer",
                 "EffArea": EffArea,
                 "OffAxisAngle": OffAxisAngle,
                 "min_value": 0.3,
                 "max_value": 10.0,
                 "energy_band": "0.2-12.0"}

# ----------------------------------------------------- #

# --------------- simulation_data --------------- #

simulation_data = {"object_data": object_data,
                   "telescop_data": telescop_data,
                   "INSTbkgd": 0.2,
                   "EXPtime": 1e6
                   }

# ----------------------------------------------- #

radius = args.radius*u.arcmin

if args.catalog == "Xmm_DR13":
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
                     "modeling_file_path": modeling_file_path,
                     "catalog_directory" : xmm_directory,
                     "cloesest_dataset_path": xmm_closest_catalog,
                     "img": xmm_img}
    
    simulation_data["os_dictionary"] = os_dictionary
    
    xmm = XmmCatalog(catalog_path=catalog_path, radius=radius, dictionary=object_data, user_table=add_source_table, os_dictionary=os_dictionary)
    nearby_sources_table, nearby_sources_position = xmm.nearby_sources_table,  xmm.nearby_sources_position
    model_dictionary = xmm.model_dictionary
    
    column_dictionary = {"band_flux_obs" : dict_cat.dictionary_catalog['XMM']["band_flux_obs"],
                         "band_flux_obs_err": dict_cat.dictionary_catalog["XMM"]["band_flux_obs_err"],
                         "energy_band": [0.35, 0.75, 1.5, 3.25, 8.25],
                         "sigma" : np.array([1e-20, 5e-21, 1e-22, 1e-23, 1e-24], dtype=float),
                         "data_to_vignetting": ["SC_RA", "SC_DEC", "IAUNAME"]}
    key = "XMM"
    
elif args.catalog == "CSC_2.0":
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
                     "catalog_directory": chandra_directory,
                     "cloesest_dataset_path": chandra_closest_catalog,
                     "img": chandra_img}
    
    simulation_data["os_dictionary"] = os_dictionary
    
                    # cs = cone search (Harvard features)
    csc = Chandra(catalog_path=catalog_path, radius=radius, dictionary=object_data, user_table=add_source_table, os_dictionary=os_dictionary)
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
                                 "sigma": np.array([1e-20, 1e-22, 1e-24], dtype=float),
                                 "data_to_vignetting": ["RA", "DEC", "Chandra_IAUNAME"]}
            model_dictionary = csc.model_dictionary
            break
        elif answer == "CS_Chandra":
            key = "CS_Chandra"
            nearby_sources_table, nearby_sources_position = table_2, sources_2
            column_dictionary = {"band_flux_obs": dict_cat.dictionary_catalog[key]["band_flux_obs"],
                                 "band_flux_obs_err": dict_cat.dictionary_catalog[key]["band_flux_obs_err"],
                                 "energy_band": dict_cat.dictionary_catalog[key]["energy_band_center"],
                                 "sigma": np.array([1e-20, 1e-22, 1e-24], dtype=float),
                                 "data_to_vignetting": ["ra", "dec", "name"]}
            model_dictionary = csc.cs_model_dictionary
            break
        else:
            print(f"{colored('Key error ! ', 'red')}. Please retry !")
            answer = str(input(f"Which Table do you chose to follow the modeling ? {colored('Chandra / CS_Chandra', 'magenta')}\n"))
    
elif args.catalog == "Swift":
    # Find the optimal pointing point with the Swift catalog
    swi = Swift(catalog_path=catalog_path, radius=radius, dictionary=object_data, user_table=add_source_table)
    nearby_sources_table, nearby_sources_position = swi.nearby_sources_table, swi.nearby_sources_position
    model_dictionary = swi.dictionary_model
elif args.catalog == "eRosita":
    # Find the optimal pointing with the eRosita catalog
    eRo = eRosita(catalog_path=catalog_path, radius=radius, dictionary=object_data, user_table=add_source_table)
    nearby_sources_table, nearby_sources_position = eRo.nearby_sources_table, eRo.nearby_sources_position
    model_dictionary = eRo.dictionary_model
elif args.catalog == "compare_catalog":
    # Find the optimal pointing point with two catalogs to compare data
    compare_class = CompareCatalog(catalog_path=catalog_path, radius=radius, dictionary=object_data, user_table=add_source_table)
    compare_class.opti_point_calcul(simulation_data=simulation_data)
    sys.exit()
    
# --------------- count_rates --------------- #

excel_data_path = os.path.join(active_workflow, 'excel_data').replace("\\", "/")

if platform.system() == "Linux":
    count_rates, nearby_sources_table = f.count_rates(nearby_sources_table, model_dictionary, telescop_data)
    f.py_to_xlsx(excel_data_path=excel_data_path, count_rates=count_rates, object_data=object_data, args=(args.catalog, key), radius=args.radius)
elif platform.system() == "Windows":
    count_rates, nearby_sources_table = f.xlsx_to_py(excel_data_path=excel_data_path, nearby_sources_table=nearby_sources_table, object_data=object_data, args=(args.catalog, key), radius=args.radius)
else:
    sys.exit()
    
simulation_data['nearby_sources_table'] = nearby_sources_table

# -------------------------------------------------- #

# --------------- Nominal pointing infos --------------- #
            
f.nominal_pointing_info(simulation_data, nearby_sources_position)

# ------------------------------------------------------ #

# --------------- Value of optimal pointing point and infos --------------- #

            
OptimalPointingIdx, SRCoptimalSEPAR, SRCoptimalRATES, vector_dictionary = f.calculate_opti_point(simulation_data, nearby_sources_position)

f.optimal_point_infos(vector_dictionary, OptimalPointingIdx, SRCoptimalRATES)

# ------------------------------------------------------------------------- #

# --------------- Visualized data Matplotlib with S/N --------------- #

f.data_map(simulation_data, vector_dictionary, OptimalPointingIdx, nearby_sources_position)

# ------------------------------------------------------------------- #

# --------------- Calculate vignetting factor --------------- #

vignetting_factor, nearby_sources_table = f.vignetting_factor(OptimalPointingIdx=OptimalPointingIdx, vector_dictionary=vector_dictionary, simulation_data=simulation_data, data=column_dictionary["data_to_vignetting"])

# ----------------------------------------------------------- #

# --------------- Modeling nearby sources --------------- #

f.modeling(vignetting_factor=vignetting_factor, simulation_data=simulation_data, column_dictionary=column_dictionary, catalog_name=args.catalog)

# ------------------------------------------------------- #

# --------------- write fits file --------------- #

f.write_fits_file(nearby_sources_table=nearby_sources_table, simulation_data=simulation_data)

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
    master_sources = f.load_master_sources(output_name)
    f.master_source_plot(master_sources=master_sources, object_data=object_data, number_graph=2)
except Exception as error :
    print(f"{colored('An error occured : ', 'red')} {error}")

# ---------------------------------------- #

# --------------- modeling spectra with jaxspec --------------- #

# setup jaxspec
config.update("jax_enable_x64", True)
numpyro.set_platform("cpu")

# define caracteristic model here --> exp(-nh*$\sigma$) * x ** (-$\Gamma$)
model = Tbabs() * Powerlaw()

# load instrument parameters
instrument = Instrument.from_ogip_file(nicer_data_arf, nicer_data_rmf, exposure=50_000)

# load all of the sources spetcra
total_spectra = f.modeling_source_spectra(nearby_sources_table=nearby_sources_table, instrument=instrument, model=model)

# plot of all spectra data
f.total_plot_spectra(total_spectra=total_spectra, instrument=instrument, simulation_data=simulation_data, catalog_name=args.catalog)

# ------------------------------------------------------------- # 
