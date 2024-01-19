# ------------------------------ #
        # Python's packages
        
from astropy.table import Table
from typing import Dict, List, Tuple


import numpy as np
import catalog_information as dict_cat

# ------------------------------ #

# ---------- for documentation ---------- #

# import src.catalog_information as dict_cat

# --------------------------------------- #

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
    Creates a dictionary mapping each unique name in a list to its indices.

    This function processes a list of names and constructs a dictionary where each unique name is a key, 
    and its value is a list of indices where that name appears in the original list. It's particularly useful 
    for identifying and handling duplicates in a list of names.

    Args:
        name_list (List[str]): A list of names, which can include duplicates.

    Returns:
        Dict[str, List[int]]: A dictionary where each key is a unique name from the list, and the value is a 
                              list of indices at which this name appears in the original list. Only names that 
                              appear more than once in the list are included as keys in the dictionary.

    The function iterates through the list, tracking the occurrence of each name. If a name appears more than 
    once, it's added to the dictionary with a list of all its indices in the original list.

    Note:
        - Names that appear only once in the list are not included in the returned dictionary.
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
    Inserts new items into a dictionary and maintains sorted order based on the values.

    This function updates a dictionary by inserting new items from a list of tuples. Each tuple contains 
    a name and an index. The dictionary has names as keys and lists of indices as values. After insertion, 
    the indices in the values are kept in sorted order, ensuring that the dictionary remains organized.

    Args:
        duplicate_dict (Dict[str, List[int]]): A dictionary where keys are names and values are lists of indices.
        new_row (List[Tuple[str, int]]): A list of tuples, with each tuple containing a name (str) and an index (int) to be inserted.

    Returns:
        Dict[str, List[int]]: The updated dictionary with new items inserted. For each name in the new row, 
                              the corresponding index is inserted into the list of indices in the dictionary 
                              and sorted to maintain order.

    The function iterates through each tuple in the new row, updating the dictionary accordingly. If the name 
    already exists as a key in the dictionary, the new index is added to its list and then sorted. If the name 
    is new, it is added as a new key with its index as the first item in the list.
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
    Replaces NaN values in specific columns of an astropy Table with the minimum non-NaN values from the same column.

    This function is designed to process an astropy Table, targeting specific columns identified using a catalog key. 
    In each specified column, NaN values are replaced with the minimum non-NaN value found in that column. This is 
    particularly useful for cleaning up astronomical data tables where NaN values can impede analysis.

    Args:
        key (str): The catalog key used to identify which columns in the table should be processed.
        unique_table (Table): An astropy Table object containing data, some of which may include NaN values.

    Returns:
        Table: The updated astropy Table with NaN values replaced by the minimum non-NaN values in their respective columns.

    The function identifies columns based on the catalog key and iterates through them. For each column, it determines 
    the minimum non-NaN value and replaces any NaN values with this minimum value. This operation is performed in-place,
    modifying the original table.

    Note:
        - The function depends on the structure defined in `dict_cat.dictionary_catalog[key]` to identify relevant columns.
        - This method does not create a new table but modifies the existing one.
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
    Creates a catalog of unique astronomical sources based on data from a nearby sources table.

    This function processes a table of nearby source data, focusing on consolidating duplicate entries and 
    normalizing data across various columns. It uses specified column names to identify and aggregate data 
    for each unique source, such as their average positions and fluxes. The function is tailored to work 
    with different astronomical catalogs by adapting to their specific column naming conventions.

    Args:
        nearby_sources_table (Table): An astropy Table containing data of nearby astronomical sources.
        column_name (Dict): A dictionary specifying the column names to use, including keys for catalog names, 
                            source names, right ascension, declination, and various flux measurements.

    Returns:
        Table: An astropy Table representing the unique sources catalog. This table includes averaged 
               positional data and flux measurements for each unique source, based on the catalog key provided.

    The function identifies unique sources, calculates their mean right ascension and declination, and averages 
    flux data where duplicates are found. The output table is structured to include columns for source identifiers 
    (e.g., IAUNAME), average coordinates, and flux data. The structure of the output table varies based on the 
    catalog key specified in `column_name`.

    Note:
        - The function handles NaN values in flux columns by replacing them with the mean of non-NaN values.
        - This method is particularly useful for cleaning and organizing astronomical data from surveys 
          with multiple observations of the same sources.
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

