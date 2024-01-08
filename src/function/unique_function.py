# ------------------------------ #
        # Python's packages
        
from astropy.table import Table
from typing import Dict, List, Tuple


import numpy as np
import catalog_information as dict_cat

# ------------------------------ #

"""
This module is designed for reducing a multiple sources table to a unique source table in astronomical data analysis. It provides functions for creating and manipulating unique source catalogs based on various criteria and data from different astronomical catalogs. The module integrates with the astropy library for table manipulation and utilizes custom catalog information for specific data fields.

Functions:
- unique_dict: Creates a dictionary associating names with their indices in a list, identifying duplicates.
- insert_row: Inserts a new row into a dictionary while maintaining a sorted order based on values.
- replace_nan_value: Replaces NaN values in specified columns of a table with their minimum non-NaN values.
- create_unique_sources_catalog: Creates a unique sources catalog based on a nearby sources table and specific column names.

The module is particularly useful in scenarios where data from various astronomical observations need to be consolidated into a single, coherent catalog. It handles nuances such as NaN value replacement and source name duplication, ensuring data integrity and usability in further analysis.

Note:
- This module assumes familiarity with astronomical data formats and the specific requirements of different astronomical catalogs.
- It relies on external catalog information provided through a separate module (`catalog_information`) for specific data fields and processing rules.
"""

def unique_dict(name_list: List) -> Dict:
    """
    Create a dictionary that associates names with their indices in a list.

    Args:
        name_list (List): A list of names.

    Returns:
        Dict: A dictionary where keys are names and values are lists of corresponding indices.
    """
    index_dict = {}
    duplicate_dict = {}
    
    for index, item in enumerate(name_list):
        if item in index_dict:
            if item in duplicate_dict:
                duplicate_dict[item].append(index)
            else:
                duplicate_dict[item] = [index_dict[item], index]
        else:
            index_dict[item] = index
    return duplicate_dict


def insert_row(duplicate_dict: Dict, new_row: List[Tuple]) -> Dict:
    """
    Insert a new row into a dictionary and maintain sorted order based on values.

    Args:
        duplicate_dict (Dict): A dictionary with names as keys and lists of indices as values.
        new_row (List[Tuple]): A list of tuples containing (name, index) to be inserted.

    Returns:
        Dict: Updated dictionary with the new row inserted.
    """
    
    new_row.sort(key=lambda x: x[1])
    
    for key, value in new_row:
        # Convertir le dictionnaire actuel en liste de paires clé-valeur
        items = list(duplicate_dict.items())
        # Trouver l'emplacement approprié pour la nouvelle paire
        for index, (_, liste_valeurs) in enumerate(items):
            if liste_valeurs[0] > value:
                # Insérer la nouvelle paire avant cette position
                items.insert(index, (key, [value]))
                break
        else:
            # Si aucune valeur plus grande n'a été trouvée, ajouter à la fin
            items.append((key, [value]))
        # Recréer le dictionnaire
        duplicate_dict = dict(items)
    return duplicate_dict  


def replace_nan_value(key: str, unique_table: Table) -> Table:
    """
    Replace NaN values in a table's specified columns with their minimum non-NaN values.

    Args:
        key (str): The catalog key to determine which columns to process.
        unique_table (Table): A table containing data with NaN values.

    Returns:
        Table: Updated table with NaN values replaced.
    """
    
    flux_obs = dict_cat.dictionary_catalog[key]["flux_obs"]
    flux_obs_err = dict_cat.dictionary_catalog[key]["flux_obs_err"]
    band_flux_obs = dict_cat.dictionary_catalog[key]["band_flux_obs"]
    band_flux_obs_err = dict_cat.dictionary_catalog[key]["band_flux_obs_err"]

    flux_list = [flux_obs, flux_obs_err, band_flux_obs, band_flux_obs_err[0], band_flux_obs_err[1]]

    flux_name = []

    for flux in flux_list:
        if isinstance(flux, str):
            flux_name.append(flux)
        else:
            for item in range(len(flux)):
                flux_name.append(flux[item])

    for name in flux_name:
        flux_data = []
        index_data = []
        for index, flux in enumerate(unique_table[name]):
            if not np.isnan(flux):
                flux_data.append(flux)
            else:
                index_data.append(index)
        min_value = np.min(flux_data)
        
        for index in index_data:
            unique_table[name][index] = min_value
            
    return unique_table


def create_unique_sources_catalog(nearby_sources_table: Table, column_name: List) -> Table:
    """
    Create a unique sources catalog based on a nearby sources table and catalog-specific column names.

    Args:
        nearby_sources_table (Table): A table containing nearby sources data.
        column_name (List): A list of column names used for catalog-specific data.

    Returns:
        Table: A table representing the unique sources catalog.
    """
    
    key = column_name["catalog_name"]

    dict_flux_name = {"flux_obs": dict_cat.dictionary_catalog[key]["flux_obs"],
                      "flux_obs_err": dict_cat.dictionary_catalog[key]["flux_obs_err"],
                      "band_flux_obs": dict_cat.dictionary_catalog[key]["band_flux_obs"],
                      "band_flux_obs_err": dict_cat.dictionary_catalog[key]["band_flux_obs_err"]}
    
    list_flux_name = [dict_flux_name["flux_obs"], dict_flux_name["flux_obs_err"], dict_flux_name["band_flux_obs"], dict_flux_name["band_flux_obs_err"][0], dict_flux_name["band_flux_obs_err"][1]]
    
    flux_name = []
    for value in list_flux_name:
        if isinstance(value, str):
            flux_name.append(value)
        else:
            for item in value:
                flux_name.append(item)
                
    for flux in flux_name:
        min_value = np.nanmean(nearby_sources_table[flux])
        nan_mask = np.isnan(nearby_sources_table[flux])
        nearby_sources_table[flux][nan_mask] = min_value

    duplicate_dict = unique_dict(nearby_sources_table[column_name["source_name"]])
    
    new_row = []
    for index, name in enumerate(nearby_sources_table[column_name["source_name"]]):
        if name not in duplicate_dict.keys():
            new_row.append((name, index))
        
    sources_dict = insert_row(duplicate_dict=duplicate_dict, new_row=new_row)
    
    if key == "Chandra":
        
        iauname_col, ra_col, dec_col = [], [], []
        for key, value in list(sources_dict.items()):
            iauname_col.append(key)
            ra_col.append(np.mean([nearby_sources_table[column_name["right_ascension"]][index] for index in value]))
            dec_col.append(np.mean([nearby_sources_table[column_name["declination"]][index] for index in value]))
        
        unique_table = Table()
        unique_table["Chandra_IAUNAME"] = iauname_col
        unique_table["RA"] = ra_col
        unique_table["DEC"] = dec_col
        
    if key == "Swift":
        
        iauname_col, ra_col, dec_col = [], [], []
        for key, value in list(sources_dict.items()):
            iauname_col.append(key)
            ra_col.append(np.mean([nearby_sources_table[column_name["right_ascension"]][index] for index in value]))
            dec_col.append(np.mean([nearby_sources_table[column_name["declination"]][index] for index in value]))
        
        unique_table = Table()
        unique_table["Swift_IAUNAME"] = iauname_col
        unique_table["RA"] = ra_col
        unique_table["DEC"] = dec_col
        
    if key == "eRosita":
    
        iauname_col, ra_col, dec_col = [], [], []
        for key, value in list(sources_dict.items()):
            iauname_col.append(key)
            ra_col.append(np.mean([nearby_sources_table[column_name["right_ascension"]][index] for index in value]))
            dec_col.append(np.mean([nearby_sources_table[column_name["declination"]][index] for index in value]))
        
        unique_table = Table()
        unique_table["eRosita_IAUNAME"] = iauname_col
        unique_table["RA"] = ra_col
        unique_table["DEC"] = dec_col
        
    for flux in flux_name:
        data = []
        for value in list(sources_dict.values()):
            if len(value) != 1:
                new_value = np.mean([nearby_sources_table[flux][index] for index in value])
            else:
                new_value = nearby_sources_table[flux][value[0]]
            data.append(new_value)
        unique_table[flux] = data
        
    return unique_table

