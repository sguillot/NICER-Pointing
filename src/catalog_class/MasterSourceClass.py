# --------------- Packages --------------- #

import catalog_information as dict_cat
import numpy as np

# ---------------------------------------- #

# ---------- for documentation ---------- #

# import src.catalog_information as dict_cat

# --------------------------------------- #

class BandFlux:
    """
    Represents the observed flux and its error in a specific energy band for astronomical objects.

    The `BandFlux` class is tailored for handling and encapsulating data related to the observed flux in a distinct energy band. It's an essential component in spectral analysis and photometric measurements, where precise flux data and associated uncertainties are crucial.

    Note:
        - This class is particularly useful in scenarios where a clear and concise representation of flux data, along with its accuracy, is required.
        - The energy band specificity of the class makes it adaptable to different wavelength ranges, facilitating its use in various astronomical studies.
        
    Example:
        Creating a BandFlux instance for a hypothetical observation:
        
        >>> observed_flux = 2.3  # Flux value
        >>> observed_flux_err = 0.2  # Flux error
        >>> band_flux = BandFlux(observed_flux, observed_flux_err)

    This class is an integral tool for astronomers and researchers, enabling the effective management and analysis of flux data in specific energy bands.
    """
    
    def __init__(self, flux, flux_err) -> None:
        """
        Initializes a new instance of the `BandFlux` class.

        Args:
            flux (float): The observed flux value in the energy band.
            flux_err (float): The error associated with the observed flux value.
        
        Attributes:
        
        .. attribute:: flux
            :type: float
            :value: flux
            
        .. attribute:: flux_err
            :type: float
            :value: flux_err

        This constructor method creates an instance of `BandFlux` with the provided flux and its error. 
        These values are expected to be numerical representations of the observed flux and its uncertainty.
        """
        
        self.flux = flux
        self.flux_err = flux_err


class SwiftData:
    """
    The SwiftData class serves as a specialized container for handling time-series data obtained from the Swift telescope. 
    It is particularly designed to store and manage flux observations along with their corresponding errors and observation times.

    Important:
        - SwiftData is tailored for handling time-series data specifically from the Swift telescope, one of the most 
          versatile observatories for astrophysical research.
        - The class is crucial for astronomers and researchers focusing on time-resolved studies of celestial sources, 
          providing a structured approach to manage and analyze variable flux data over time.
        - By organizing flux values, their errors, and observation times, SwiftData facilitates various time-series 
          analysis techniques, essential for understanding transient or variable astronomical phenomena.

    Note:
        - The design of SwiftData underscores the importance of precision and accuracy in astrophysical data analysis. 
          It emphasizes careful handling of observational data and associated errors.
        - While specific to Swift telescope data, the principles and structure of the SwiftData class can be adapted 
          for use with time-series data from other telescopes or astronomical surveys.
        - Users of SwiftData are expected to have a fundamental understanding of observational astronomy and time-series 
          analysis to effectively utilize the class for research purposes.

    Example Usage:
        Example of initializing SwiftData with sample flux data, errors, and times:
        
        >>> sample_flux = [1.2, 1.3, 1.4]
        >>> sample_flux_err = [0.1, 0.1, 0.1]
        >>> sample_times = [10.5, 10.6, 10.7]
        >>> swift_data_instance = SwiftData(sample_flux, sample_flux_err, sample_times)

    SwiftData represents a critical step in organizing and preparing astrophysical time-series data for comprehensive analysis, 
    offering researchers a reliable and efficient way to manage observations from one of the leading space telescopes.
    """
    
    def __init__(self, stacked_flux, stacked_flux_err, stacked_times):
        """
        Initializes a new instance of the `SwiftData` class, which is designed for managing and analyzing time-series data from the Swift telescope.

        This constructor sets up the `SwiftData` object with stacked flux values, their associated errors, and corresponding observation times. These data are crucial for time-series analyses, offering insights into the behavior and characteristics of astronomical sources as observed by the Swift telescope.

        Args:
            stacked_flux (List[float]): A list of stacked flux values. Each value represents the integrated or cumulative flux measured by the Swift telescope over a specific time interval.
            stacked_flux_err (List[float]): A list of errors associated with the stacked flux values. These errors provide an estimate of the uncertainty in each flux measurement.
            stacked_times (List[float]): A list of observation times. Each time value corresponds to the observation time of the respective flux measurement.

        Attributes:
        
        .. attribute:: stacked_flux
            :type: List[float]
            :value: stacked_flux
            
        .. attribute:: stacked_flux_err
            :type: List[float]
            :value: stacked_flux_err
            
        .. attribute:: stacked_times
            :type: List[float]
            :value: stacked_times

        This constructor method creates an instance of `SwiftData` with the provided flux data, 
        errors, and times. These values are typically derived from observations conducted by the 
        Swift telescope and are crucial in time-series analyses of astronomical sources.
        
        Note:
            - The `SwiftData` class is specifically designed to work with time-series data from the Swift telescope. It may not be suitable for data from other telescopes without modifications.
            - Users are expected to have a foundational understanding of flux measurements and error analysis in astronomy for effective use of this class.
        """
        self.stacked_flux = stacked_flux
        self.stacked_flux_err = stacked_flux_err
        self.stacked_times = stacked_times
        

class Source:
    """
    ERWAN's QUINTIN code
    
    Manages and interprets observational data of an astronomical source, integrating various types of measurements.

    The `Source` class is a comprehensive tool designed for astronomers and astrophysicists to handle diverse observational data of celestial objects. It seamlessly integrates flux measurements, time series data, hardness ratios, and band-specific flux information, offering a unified approach to data analysis. This class proves invaluable in studies involving multi-wavelength observations, time variability analysis, and spectral properties of astronomical sources.

    Important:
        - The class is versatile, capable of handling data from different astronomical catalogs and telescopes, including Swift and XMM-Newton.
        - It offers advanced features such as hardness ratio calculations and handling Swift telescope data, essential for X-ray astronomy.
        - The class supports the inclusion of band flux data, enhancing its utility in spectral analysis and energy distribution studies.
        - Users can leverage this class to analyze short-term variability and off-axis observations, crucial for understanding dynamic celestial phenomena.

    Note:
        - `Source` class is a pivotal element in the toolkit of astronomers analyzing complex observational datasets.
        - It simplifies the process of correlating various data types, offering a holistic view of the observed astronomical source.
        - The class structure is adaptable, allowing for the integration of additional data types and parameters as astronomical research evolves.
        - Its design caters to both specific and broad analyses, making it suitable for detailed studies of individual sources and comparative studies across different catalogs.

    Example:
        Example of initializing a Source object with sample data:
        
        >>> catalog_name = "Sample Catalog"
        >>> iau_name = "NAME"
        >>> flux = [1.2, 1.4, 1.3]
        >>> flux_err = [0.1, 0.1, 0.1]
        >>> time_steps = [100, 200, 300]
        >>> band_flux_data = BandFlux([1.1, 1.2, 1.1], [0.05, 0.05, 0.05])
        >>> source = Source(catalog_name, iau_name, flux, flux_err, time_steps, band_flux_data)

    The `Source` class is an indispensable asset in the field of astronomy, facilitating the detailed analysis of sources and enhancing our understanding of the universe.
    """
    
    def __init__(self, catalog, iau_name, flux, flux_err, time_steps, band_flux_data, **kwargs) -> None:
        """
        Initializes a new instance of the Source class.

        Args:
            catalog (str): The name of the catalog from which the source data is obtained.
            iau_name (str): The IAU designated name of the source.
            flux (list[float]): A list of observed flux values for the source.
            flux_err (list[float]): A list of errors associated with the observed flux values.
            time_steps (list[float]): A list of observation times corresponding to each flux value.
            band_flux_data (BandFlux): An instance of the BandFlux class containing flux and flux error data for specific energy bands.
            kwargs: Additional optional parameters such as observation IDs (obs_id), Swift data (swift_data), etc.

        This constructor initializes the Source object with observational data, converting time steps and observation IDs into appropriate formats and setting up default values for attributes based on the provided keyword arguments.

        Attributes:
        
        .. attribute:: catalog
            :type: str
            :value: catalog
            
        .. attribute:: iau_name
            :type: str
            :value: iau_name
            
        .. attribute:: flux
            :type: float
            :value: flux
            
        .. attribute:: flux_err
            :type: float
            :value: flux_err
            
        .. attribute:: time_steps
            :type: List[float]
            
        .. attribute:: obs_ids
            :type: List[int]
            
        .. attribute:: band_flux
            :type: List[float]
            :value: band_flux_data.flux
            
        .. attribute:: band_flux_err
            :type: List[float]
            :value: band_flux_data.flux_err

        Note:
            - The catalog name helps to tailor the analysis methods to the specific format and conventions used in the catalog.
            - The IAU name is a standardized astronomical naming convention, ensuring consistency across different datasets and studies.
            - Flux, flux error, and time steps are fundamental observational data, crucial for any time-series or spectral analysis.
            - The band_flux_data argument allows for integration of spectral data specific to certain energy bands, enhancing the scope of analysis.
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
        Calculates and assigns the hardness ratio for the source based on catalog-specific parameters.

        Args:
            catalog (str): The name of the catalog, which defines energy bands and conversion factors for calculating the hardness ratio.

        The hardness ratio is a dimensionless value that represents the spectral shape of the source, typically used in X-ray and gamma-ray astronomy. This method calculates the hardness ratio using soft and hard detections derived from band flux data, taking into account the specific energy bands defined for the given catalog.

        Note:
            - The hardness ratio is a crucial parameter for understanding the spectral properties of astronomical sources.
            - Different catalogs may have different definitions and energy ranges for calculating the hardness ratio, hence the need for the catalog parameter.
            - This method modifies the source object by adding new attributes for the hardness ratio and associated errors.
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
        Updates the Swift telescope-specific data attributes of the source object.

        Args:
            flux (list[float]): A list of flux values observed by the Swift telescope.
            flux_err (list[float]): A list of errors associated with the observed flux values.

        This method is designed for handling and updating data specifically from the Swift telescope. It involves modifying stacked flux values, error margins, and checking for variability based on the provided flux and error parameters.

        Note:
            - The method is tailored for Swift telescope data but can be adapted for other telescopic data with similar structure.
            - It enhances the source object by incorporating telescope-specific variations and nuances into the analysis.
            - Variability checks are crucial for time-sensitive and transient astronomical phenomena.
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
    A class for consolidating and managing astronomical data from various sources into a single master source.

    The `MasterSource` class is instrumental in aggregating diverse observational data related to astronomical sources from multiple catalogs. It expertly manages the complexities of data integration, including the removal of redundant data, and ensures the maintenance of a comprehensive and unified record. This class is particularly valuable in multi-wavelength astronomical research, where data from different telescopes and catalogs need to be combined for a holistic analysis.

    Important:
        - The class is designed to handle a variety of data types, including flux measurements, time steps, observation IDs, and hardness ratios.
        - It provides functionalities for calculating various parameters such as variability ratios, minimum and maximum observational times, and flux limits.
        - The class is equipped to deal with specific data from telescopes like Swift, XMM-Newton, and Chandra, enhancing its utility in observational astronomy.

    Note:
        - `MasterSource` is a key component in studies that require cross-matching and analysis of data from different astronomical catalogs.
        - By offering a structured approach to data consolidation, it aids in the accurate and comprehensive analysis of astronomical sources.
        - This class serves as a foundational tool for researchers aiming to understand the complex nature of celestial objects by examining their multi-wavelength signatures.

    Example Usage:
        Creating a MasterSource instance with data from various Source objects:
        
        >>> master_source = MasterSource(src_id=12345, sources_table=[source1, source2], ra=180.0, dec=-45.0, poserr=0.1)
        
        Here, source1 and source2 are instances of the Source class, representing data from different catalogs.

    The MasterSource class stands at the forefront of modern astronomical data analysis, offering an integrated and efficient way to study the cosmos.
    """
    
    def __init__(self, src_id, sources_table, ra, dec, poserr) -> None:
        """
        Initializes a new instance of the MasterSource class.

        Args:
            src_id (int): A unique identifier for the master source.
            sources_table (list[Source]): A list of Source objects from different catalogs.
            ra (float): Right ascension of the master source.
            dec (float): Declination of the master source.
            poserr (float): Positional error of the master source's location.

        This constructor method creates an instance of `MasterSource` with the provided source data. 
        It processes each source in the sources_table, consolidates flux measurements, and calculates 
        various parameters like minimum and maximum times, flux limits, variability ratios, and hardness 
        ratios. The method also deals with data specific to certain telescopes like Swift, XMM-Newton, 
        and Chandra.
        
        Attributes:
        
        .. attribute:: src_id
            :type: int
            :value: Unique identifier for the master source.
        
        .. attribute:: sources
            :type: dict
            :value: Dictionary storing Source objects from different catalogs.
        
        .. attribute:: sources_flux
            :type: np.ndarray
            :value: Array of flux values from all sources.
        
        .. attribute:: sources_error_bar
            :type: np.ndarray
            :value: Array of flux error values from all sources.
        
        .. attribute:: sources_time_steps
            :type: list
            :value: List of time steps corresponding to each observation.
        
        .. attribute:: sources_var
            :type: list
            :value: List of variability flags for each observation.
        
        .. attribute:: tab_hr
            :type: list
            :value: List of hardness ratios.
        
        .. attribute:: tab_hr_err
            :type: list
            :value: List of errors in hardness ratios.
        
        .. attribute:: never_on_axis_xmm
            :type: bool
            :value: Indicates if source never appeared on-axis in XMM observations.
        
        .. attribute:: has_short_term_var
            :type: bool
            :value: Flag for the presence of short-term variability.
        
        .. attribute:: min_time
            :type: float
            :value: Minimum observation time across all sources.
        
        .. attribute:: max_time
            :type: float
            :value: Maximum observation time across all sources.
        
        .. attribute:: min_upper
            :type: float
            :value: Minimum upper limit of the source's flux.
        
        .. attribute:: max_lower
            :type: float
            :value: Maximum lower limit of the source's flux.
        
        .. attribute:: var_ratio
            :type: float
            :value: Variability ratio of the source.
        
        .. attribute:: var_amplitude
            :type: float
            :value: Variability amplitude.
        
        .. attribute:: var_significance
            :type: float
            :value: Significance of the variability.
        
        .. attribute:: hr_min
            :type: float
            :value: Minimum hardness ratio.
        
        .. attribute:: hr_max
            :type: float
            :value: Maximum hardness ratio.
        
        .. attribute:: hr_var
            :type: float
            :value: Variability in hardness ratio.
        
        .. attribute:: hr_var_signif
            :type: float
            :value: Significance of hardness ratio variability.
        
        .. attribute:: xmm_ul
            :type: list
            :value: List of upper limits from XMM observations.
        
        .. attribute:: xmm_ul_dates
            :type: list
            :value: Dates corresponding to XMM upper limits.
        
        .. attribute:: xmm_ul_obsids
            :type: list
            :value: Observation IDs for XMM upper limits.
        
        [additional attributes for other telescopes like slew_ul, slew_ul_dates, etc., as applicable]
        
        .. attribute:: ra
            :type: float
            :value: Right ascension of the source.
        
        .. attribute:: dec
            :type: float
            :value: Declination of the source.
        
        .. attribute:: pos_err
            :type: float
            :value: Positional error of the source.
        
        .. attribute:: glade_distance
            :type: list
            :value: Distances from GLADE catalog.
        
        .. attribute:: simbad_type
            :type: str
            :value: Source type from the SIMBAD database.
        
        .. attribute:: has_sdss_widths
            :type: bool
            :value: Flag indicating the presence of SDSS widths.

        Important:
            - The src_id serves as a unique identifier for the master source across different catalogs.
            - The sources_table contains Source objects, each representing an astronomical source from 
              a specific catalog. This table is crucial for aggregating and analyzing multi-wavelength data.
            - The method handles complexities such as redundant data removal and consolidation of 
              measurements from various telescopes.
            - The right ascension (ra) and declination (dec) along with the positional error (poserr) 
              provide accurate spatial information about the master source.

        Note:
            - This class is integral to studies involving cross-matching and combining data from different 
              astronomical catalogs.
            - It enhances the capacity to perform comprehensive and multi-faceted analyses of astronomical sources.
            - The constructor carefully processes the input data, ensuring accuracy and consistency in the 
              combined master source profile.
        """
        
        
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

  