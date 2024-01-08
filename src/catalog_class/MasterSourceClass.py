# --------------- Packages --------------- #

import catalog_information as dict_cat
import numpy as np

# ---------------------------------------- #

class BandFlux:
    """
    A class representing the observed flux and its error in a specific energy band.

    Attributes:
    flux (float): The observed flux value in the energy band.
    flux_err (float): The error associated with the observed flux value.

    The `BandFlux` class is designed to encapsulate the properties of an observed flux in a specific energy band of an astronomical object. It stores the flux value along with its corresponding error. This class can be used in scenarios where handling flux data and its uncertainty is required, such as in spectral analysis or photometric measurements.

    Methods:
    __init__: Initializes a new instance of the `BandFlux` class with the specified flux and error values.
    """
    
    def __init__(self, flux, flux_err) -> None:
        """
        Initializes a new instance of the `BandFlux` class.

        Parameters:
        flux (float): The observed flux value in the energy band.
        flux_err (float): The error associated with the observed flux value.

        This constructor method creates an instance of `BandFlux` with the provided flux and its error. 
        These values are expected to be numerical representations of the observed flux and its uncertainty.
        """
        
        self.flux = flux
        self.flux_err = flux_err


class SwiftData:
    """
    A class to represent and store the stacked flux data, its associated errors, 
    and observation times for astronomical observations, specifically from the Swift telescope.

    This class is tailored for handling time-series data from the Swift telescope, 
    commonly used in astrophysical studies. It stores stacked flux values, their corresponding 
    errors, and the observation times, facilitating data analysis and manipulation in 
    time-resolved studies.

    Attributes:
    stacked_flux (list of float): A list of stacked flux values observed by the Swift telescope.
    stacked_flux_err (list of float): A list of errors associated with the stacked flux values.
    stacked_times (list of float): A list of observation times corresponding to each flux value.

    Methods:
    __init__: Initializes a new instance of the `SwiftData` class with the specified data.
    """
    
    def __init__(self, stacked_flux, stacked_flux_err, stacked_times):
        """
        Initializes a new instance of the `SwiftData` class.

        Parameters:
        stacked_flux (list of float): A list of stacked flux values observed by the Swift telescope.
        stacked_flux_err (list of float): A list of errors associated with the stacked flux values.
        stacked_times (list of float): A list of observation times corresponding to each flux value.

        This constructor method creates an instance of `SwiftData` with the provided flux data, 
        errors, and times. These values are typically derived from observations conducted by the 
        Swift telescope and are crucial in time-series analyses of astronomical sources.

        """
        self.stacked_flux = stacked_flux
        self.stacked_flux_err = stacked_flux_err
        self.stacked_times = stacked_times
        

class Source:
    """
    A class to encapsulate various observational data of an astronomical source, 
    including flux measurements, band flux data, observation times, and additional 
    source-specific parameters.

    This class is designed to provide a structured way to store and manipulate 
    observational data from different catalogs, making it easier to perform various 
    analyses like calculating hardness ratios or handling Swift telescope data.

    Attributes:
    catalog (str): Name of the catalog from which the source data is obtained.
    iau_name (str): International Astronomical Union (IAU) designated name of the source.
    flux (list of float): List of flux values of the source.
    flux_err (list of float): Corresponding errors of the flux values.
    time_steps (list of float): Time steps of the observations.
    obs_ids (list of int): Observation IDs, if available.
    band_flux (list of float): Flux values in specific energy bands.
    band_flux_err (list of float): Errors associated with the band flux values.
    swift_data (SwiftData): Object containing Swift telescope specific data.
    xmm_offaxis (list): Off-axis angles for XMM-Newton observations, if available.
    short_term_var (list): Information about short-term variability, if available.

    Methods:
    __init__: Initializes the Source object with given data.
    hardness_ratio: Calculates the hardness ratio for the source based on catalog-specific parameters.
    swift_modif: Modifies Swift data attributes based on provided flux and flux error.

    """
    
    def __init__(self, catalog, iau_name, flux, flux_err, time_steps, band_flux_data, **kwargs) -> None:
        """
        Initializes a new instance of the Source class.

        Parameters:
        catalog (str): Name of the catalog from which the source data is obtained.
        iau_name (str): IAU designated name of the source.
        flux (list of float): List of flux values of the source.
        flux_err (list of float): Corresponding errors of the flux values.
        time_steps (list of float): Time steps of the observations.
        band_flux_data (BandFlux): Object containing flux and flux error data for specific energy bands.
        kwargs: Additional parameters such as observation IDs (obs_id), Swift data (swift_data), etc.

        This constructor initializes the Source object with the given data, converting some of the data 
        like observation times and observation IDs into appropriate formats. It also sets up default 
        values for some attributes based on the provided keyword arguments.

        """
        self.catalog = catalog
        self.iau_name = iau_name
        self.flux = flux
        self.flux_err = flux_err
        self.time_steps = [float(time) for time in time_steps]
        self.obs_ids = [int(obs_id) for obs_id in kwargs.get('obs_id', [])]

        self.band_flux = band_flux_data.flux
        self.band_flux_err = band_flux_data.flux_err

        self.swift_data = kwargs.get('swift_data', SwiftData([], [], []))
        self.xmm_offaxis = kwargs.get('xmm_offaxis', [])
        self.short_term_var = kwargs.get('short_term_var', [])

        self.hardness_ratio(catalog=catalog)
        self.swift_modif(flux=flux, flux_err=flux_err)

    
    def hardness_ratio(self, catalog:str) -> None:
        """
        Calculates the hardness ratio for a source based on the given catalog's parameters.

        The hardness ratio is a measure of the spectral shape of the source, typically
        used in X-ray astronomy. It is calculated using soft and hard detections which are 
        derived from band flux data, considering the specific energy bands defined for the 
        given catalog.

        Parameters:
        catalog (str): Name of the catalog which defines the energy bands and conversion factors.

        This method modifies the source object by calculating and setting the hardness ratio
        along with the associated errors.

        """

        hr_bandlimit_index = dict_cat.dictionary_catalog[catalog]["hr_bandlimit_index"]
        band_conv_factor_soft = dict_cat.dictionary_catalog[catalog]["band_conv_factor_soft"]
        band_conv_factor_hard = dict_cat.dictionary_catalog[catalog]["band_conv_factor_hard"]

        self.soft_dets = [np.sum(det[:hr_bandlimit_index]) * band_conv_factor_soft for det in self.band_flux]
        self.soft_errors = [[np.sum(err_neg[:hr_bandlimit_index]) * band_conv_factor_soft for err_neg in self.band_flux_err[0]],
                            [np.sum(err_pos[:hr_bandlimit_index]) * band_conv_factor_soft for err_pos in self.band_flux_err[1]]]

        if catalog != "RASS" and catalog != "WAGAT":
            self.hard_dets = [np.sum(det[hr_bandlimit_index:]) * band_conv_factor_hard for det in self.band_flux]
            self.hard_errors = [
                [np.sum(err_neg[hr_bandlimit_index:]) * band_conv_factor_hard for err_neg in
                self.band_flux_err[0]],
                [np.sum(err_pos[hr_bandlimit_index:]) * band_conv_factor_hard for err_pos in
                self.band_flux_err[1]]]
        else:
            self.hard_dets = [np.nan for det in self.flux]
            self.hard_errors = [[np.nan for det in self.flux], [np.nan for det in self.flux]]

        self.hardness_ratio = [(hard - soft)/(hard + soft) for (soft, hard) in zip(self.soft_dets, self.hard_dets)]
        low_soft = np.where(np.array(self.soft_dets) - np.array(self.soft_errors[0]) < 0, 0,
                            np.array(self.soft_dets) - np.array(self.soft_errors[0]))
        low_hard = np.where(np.array(self.hard_dets) - np.array(self.hard_errors[0]) < 0, 0,
                            np.array(self.hard_dets) - np.array(self.hard_errors[0]))
        up_soft = np.where(np.array(self.soft_dets) + np.array(self.soft_errors[1]) < 0, 0,
                           np.array(self.soft_dets) + np.array(self.soft_errors[1]))
        up_hard = np.where(np.array(self.hard_dets) + np.array(self.hard_errors[1]) < 0, 0,
                        np.array(self.hard_dets) + np.array(self.hard_errors[1]))
        self.hardness_err = [[hr - (hard - soft)/(hard + soft) for (soft, hard, hr) in zip(up_soft, low_hard, self.hardness_ratio)],
                            [(hard - soft)/(hard + soft) - hr for (soft, hard, hr) in zip(low_soft, up_hard, self.hardness_ratio)]]


    def swift_modif(self, flux:list, flux_err:list) -> None:
        """
        Modifies and updates the Swift data attributes of the source object.

        This method is specifically designed to handle data from the Swift telescope.
        It involves updating stacked flux, error values, and checking for variability 
        based on the given flux and error parameters.

        Parameters:
        flux (list of float): List of flux values of the source.
        flux_err (list of float): Corresponding errors of the flux values.

        The method updates attributes related to Swift telescope data, including checking
        for variability and adjusting minimum and maximum flux values.

        """

        self.swift_stacked_flux = self.swift_data.stacked_flux
        self.swift_stacked_flux_err = self.swift_data.stacked_flux_err
        self.swift_stacked_times = self.swift_data.stacked_times
        self.swift_stacked_variable = False
        
        self.min_upper, self.max_lower = 1, 0
        self.var = 1
        if len(flux) > 0:
            min_upper = min(np.array(flux) + np.array(flux_err[1]))
            max_lower = max(np.array(flux) - np.array(flux_err[0]))
        if self.swift_stacked_flux != []:
            stacked_min = min(np.array(self.swift_stacked_flux)+np.array(self.swift_stacked_flux_err[1]))
            if stacked_min<0.5*self.min_upper:
                self.swift_stacked_variable = True
            self.min_upper = min(self.min_upper, stacked_min)
        if len(flux) + len(self.swift_stacked_flux) > 1:
            self.var = max_lower/min_upper


class MasterSource:
    """
    A class representing a master source, consolidating data from various sources.

    This class aggregates and processes data related to astronomical sources from 
    different catalogs. It handles the removal of redundant data, calculation of various 
    properties like hardness ratio, variability ratios, and maintains a comprehensive 
    record of the source's observations across different telescopes.

    Attributes:
    src_id (int): Unique identifier for the master source.
    sources (dict): Dictionary storing Source objects from different catalogs.
    sources_flux (np.ndarray): Array of flux values from all sources.
    sources_error_bar (np.ndarray): Array of flux error values from all sources.
    sources_time_steps (list): List of time steps corresponding to each observation.
    sources_var (list): List of variability flags for each observation.
    tab_hr (list): List of hardness ratios.
    tab_hr_err (list): List of errors in hardness ratios.
    never_on_axis_xmm (bool): Indicates if source never appeared on-axis in XMM observations.
    has_short_term_var (bool): Flag for the presence of short-term variability.
    min_time (float): Minimum observation time across all sources.
    max_time (float): Maximum observation time across all sources.
    min_upper (float): Minimum upper limit of the source's flux.
    max_lower (float): Maximum lower limit of the source's flux.
    var_ratio (float): Variability ratio of the source.
    var_amplitude (float): Variability amplitude.
    var_significance (float): Significance of the variability.
    hr_min (float): Minimum hardness ratio.
    hr_max (float): Maximum hardness ratio.
    hr_var (float): Variability in hardness ratio.
    hr_var_signif (float): Significance of hardness ratio variability.
    xmm_ul (list): List of upper limits from XMM observations.
    xmm_ul_dates (list): Dates corresponding to XMM upper limits.
    xmm_ul_obsids (list): Observation IDs for XMM upper limits.
    slew_ul, slew_ul_dates, slew_ul_obsids, chandra_ul, chandra_ul_dates (lists): Similar attributes for other telescopes.
    ra (float): Right ascension of the source.
    dec (float): Declination of the source.
    pos_err (float): Positional error of the source.
    glade_distance (list): Distances from GLADE catalog.
    simbad_type (str): Source type from the SIMBAD database.
    has_sdss_widths (bool): Flag indicating the presence of SDSS widths.

    Methods:
    __init__(self, src_id, sources_table, ra, dec, poserr): Initializes the MasterSource object.

    The class primarily focuses on aggregating and processing the source data for 
    further analysis, particularly in the context of astronomical research.
    """
    
    def __init__(self, src_id, sources_table, ra, dec, poserr) -> None:
        self.src_id = src_id
        self.sources, self.sources_flux, self.sources_error_bar = {}, [], [[], []]
        self.sources_time_steps, self.sources_var = [], []
        self.tab_hr, self.tab_hr_err = [], [[], []]
        self.never_on_axis_xmm, self.has_short_term_var = False, False
        self.min_time, self.max_time = 60000, 0
        
        for source in sources_table:
            if ("XMM" in self.sources.keys()) and (source.catalog == "Stacked"):
                # We remove the Stacked detection that correspond to a clean XMM detection
                xmm_obs_id = self.sources["XMM"].obs_ids
                stacked_obs_id = source.obs_ids
                new_det_ind = [item for item in range(len(stacked_obs_id)) if stacked_obs_id[item] not in xmm_obs_id]
                
                source.flux = source.flux[new_det_ind]
                source.flux_err[0] = source.flux_err[0][new_det_ind]
                source.flux_err[1] = source.flux_err[1][new_det_ind]
                
                source.time_steps = np.array(source.time_steps)[new_det_ind]
                source.obs_ids = np.array(source.obs_ids)[new_det_ind]
                
                source.hardness_ratio = np.array(source.hardness_ratio)[new_det_ind]
                source.hardness_err[0] = np.array(source.hardness_err[0])[new_det_ind]
                source.hardness_err[1] = np.array(source.hardness_err[1])[new_det_ind]
                
                source.band_flux = source.band_flux[new_det_ind]
                source.band_flux_err[0] = source.band_flux_err[0][new_det_ind]
                source.band_flux_err[1] = source.band_flux_err[1][new_det_ind]
                
            source.master_source = self
            self.sources[source.catalog] = source
            
            for (flux, flux_err_neg, flux_err_pos, time_step) in zip(source.flux, source.flux_err[0], source.flux_err[1], source.time_steps):
                self.sources_flux.append(flux)
                self.sources_error_bar[0].append(flux_err_neg)
                self.sources_error_bar[1].append(flux_err_pos)
                self.sources_var.append(source.var)
                self.sources_time_steps.append(time_step)
            self.tab_hr += list(source.hardness_ratio)
            self.tab_hr_err[0] += list(source.hardness_err[0])
            self.tab_hr_err[1] += list(source.hardness_err[1])
            
            for (flux, flux_err_neg, flux_err_pos, start, stop) in zip(source.swift_stacked_flux, source.swift_stacked_flux_err[0], source.swift_stacked_flux_err[1], source.swift_stacked_times[0], source.swift_stacked_times[1]):
                self.sources_flux.append(flux)
                self.sources_error_bar[0].append(flux_err_neg)
                self.sources_error_bar[1].append(flux_err_pos)
                self.min_time = min(start, self.min_time)
                self.max_time = max(stop, self.max_time)
                self.sources_time_steps.append((start + stop)/2)
                
            if source.xmm_offaxis!=[]:
                if np.nanmin(source.xmm_offaxis)>1:
                    self.never_on_axis_xmm = True
            if source.time_steps!=[]:
                self.min_time = min(min(source.time_steps), self.min_time)
                self.max_time = max(max(source.time_steps), self.max_time)
            for var_flag in source.short_term_var:
                if var_flag>0:
                    self.has_short_term_var=True
        self.sources_flux = np.array(self.sources_flux)
        self.sources_error_bar = np.array(self.sources_error_bar)
        
        self.min_upper, self.max_lower, self.var_ratio = 1, 0, 1
        self.var_amplitude, self.var_significance = 0, 0

        if len(self.sources_flux)>0 and (not np.isnan(self.sources_flux).all()):
            min_upper_ind = np.argmin(self.sources_flux + self.sources_error_bar[1])
            self.min_upper = (self.sources_flux + self.sources_error_bar[1])[min_upper_ind]
            max_lower_tab = np.where(self.sources_flux - self.sources_error_bar[0]>0,
                                     self.sources_flux - self.sources_error_bar[0],
                                     self.sources_flux)
            max_lower_ind = np.argmax(max_lower_tab)
            self.max_lower = max_lower_tab[max_lower_ind]
            self.var_ratio = self.max_lower/self.min_upper
            self.var_amplitude = self.max_lower - self.min_upper
            self.var_optimistic = self.sources_flux[max_lower_ind]/self.sources_flux[min_upper_ind]
            self.var_significance = self.var_amplitude/np.sqrt(self.sources_error_bar[1][max_lower_ind]**2 + self.sources_error_bar[0][min_upper_ind]**2)
            #self.frac_var = np.sqrt((np.var(self.sources_flux, ddof=1)-np.mean(np.array(self.sources_error_bar)**2))/(np.mean(self.sources_flux)**2))
            
        self.hr_min, self.hr_max = np.nan, np.nan
        self.hr_var, self.hr_var_signif = np.nan, np.nan
        
        if len(self.tab_hr) > 1 and (not np.isnan(self.tab_hr).all()) and (not np.isnan(self.tab_hr_err).all()):
            index_hr_min = np.nanargmin(np.array(self.tab_hr) + np.array(self.tab_hr_err[1]))
            index_hr_max = np.nanargmax(np.array(self.tab_hr) - np.array(self.tab_hr_err[0]))
            self.hr_min = (np.array(self.tab_hr) + np.array(self.tab_hr_err[1]))[index_hr_min]
            self.hr_max = (np.array(self.tab_hr) - np.array(self.tab_hr_err[0]))[index_hr_max]
            self.hr_var = self.hr_max - self.hr_min
            if self.tab_hr_err[1][index_hr_min]**2 + self.tab_hr_err[0][index_hr_max]**2 > 0:
                self.hr_var_signif = self.hr_var/np.sqrt(self.tab_hr_err[1][index_hr_min]**2 + self.tab_hr_err[0][index_hr_max]**2)
            else:
                self.hr_var_signif = np.nan
            
        self.xmm_ul, self.xmm_ul_dates, self.xmm_ul_obsids = [], [], []

        self.slew_ul, self.slew_ul_dates, self.slew_ul_obsids = [], [], []

        self.chandra_ul, self.chandra_ul_dates = [], []

        self.ra, self.dec = float(ra), float(dec)
        self.pos_err = float(poserr)

        self.glade_distance=[]

        self.simbad_type = ''
        self.has_sdss_widths = False

  