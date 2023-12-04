import matplotlib
import numpy as np

catalogs = ["XMM", "Chandra", "Swift", "eRosita", "Slew", "Stacked", "RASS", "WGACAT"]
style = "bmh"
cmap_to_use = "turbo"

posErr_Names = {}
src_names = {}
colors = {}
for index, cat in enumerate(catalogs):
    posErr_Names[cat]=f"{cat}_PosErr"
    src_names[cat] = f"{cat}_IAUNAME"
    colors[cat] = matplotlib.cm.get_cmap(cmap_to_use)(index / len(catalogs))

dictionary_catalog = {
    "XMM":{"flux_obs":"SC_EP_8_FLUX",
           "flux_obs_err": ["SC_EP_8_FLUX_ERR", "SC_EP_8_FLUX_ERR"],
           "conv_factor": 1/0.999,
           "time_name": "MJD_START",
           "obsid_name": "OBS_ID",
           "band_flux_obs": [f"SC_EP_{item + 1}_FLUX" for item in range(5)],
           "band_flux_obs_err": [[f"SC_EP_{item + 1}_FLUX_ERR" for item in range(5)],
                                 [f"SC_EP_{item + 1}_FLUX_ERR" for item in range(5)]],
           "energy_band_center": [0.35, 0.75, 1.5, 3.25, 8.25],
           "energy_band_half_width": [0.15, 0.25, 0.5, 1.25, 3.75],
           "hr_bandlimit_index": 3,
           "band_conv_factor_soft": 0.35/0.35,
           "band_conv_factor_hard": 0.65/0.65,
           "hr_track_marker": "o"},
    "Chandra":{"flux_obs": "flux_powlaw_aper_b",
               "flux_obs_err": ["flux_powlaw_aper_b_negerr", "flux_powlaw_aper_b_poserr"],
               "conv_factor": 1/0.999,
               "time_name": "gti_mjd_obs",
               "obsid_name": np.nan,
               "band_flux_obs": [f"flux_powlaw_aper_{band}" for band in ["s", 'm', "h"]],
               "band_flux_obs_err": [[f"flux_powlaw_aper_{band}_negerr" for band in ["s", "m", "h"]],
                                     [f"flux_powlaw_aper_{band}_poserr" for band in ["s", "m", "h"]]],
               "energy_band_center": [0.85, 1.6, 4.5],
               "energy_band_half_width": [0.35, 0.4, 2.5],
               "hr_bandlimit_index": 2,
               "band_conv_factor_soft": 0.35/0.28,
               "band_conv_factor_hard": 0.65/0.41,
               "hr_track_marker": "v"},
    "CS_Chandra": {"flux_obs": "flux_powlaw_aper_b",
                   "flux_obs_err": ["flux_powlaw_aper_lolim_b", "flux_powlaw_aper_hilim_b"],
                   "conv_factor": 1/0.999,
                   "time_name": np.nan,
                   "obsid_name": np.nan,
                   "band_flux_obs": [f"flux_powlaw_aper_{band}" for band in ["s", 'm', "h"]],
                   "band_flux_obs_err": [[f"flux_powlaw_aper_lolim_{band}" for band in ['s', 'm', 'h']],
                                         [f"flux_powlaw_aper_hilim_{band}" for band in ['s', 'm', 'h']]],
                   "energy_band_center": [0.85, 1.6, 4.5],
                   "energy_band_half_width": [0.35, 0.4, 2.5],
                   "hr_bandlimit_index": 2,
                   "band_conv_factor_soft": 0.35/0.28,
                   "band_conv_factor_hard": 0.65/0.41,
                   "hr_track_marker": "v"},
    "Swift": {"flux_obs":"Flux",
              "flux_obs_err": ["FluxErr_neg", "FluxErr_pos"],
              "conv_factor": 1/0.69,
              "time_name": "MidTime_MJD",
              "obsid_name": "ObsID",
              "band_flux_obs": [f"Flux{item + 1}" for item in range(3)],
              "band_flux_obs_err":[[f"FluxErr{item + 1}_neg" for item in range(3)],
                                   [f"FluxErr{item + 1}_pos" for item in range(3)]],
              "energy_band_center": [0.65, 1.5, 6],
              "energy_band_half_width": [0.35, 0.5, 4],
              "hr_bandlimit_index": 2,
              "band_conv_factor_soft": 0.35/0.34,
              "band_conv_factor_hard": 0.65/0.56,
              "hr_track_marker": "s"},
    "eRosita": {"flux_obs": "ML_FLUX",
                "flux_obs_err": ["ML_FLUX_ERR", "ML_FLUX_ERR"],
                "conv_factor": 1/0.39,
                "time_name": "MJD_OBS",
                "obsid_name": np.nan,
                "band_flux_obs": [f"ML_FLUX_ERR_b{item +1}" for item in range(4)],
                "band_flux_obs_err": [[f"ML_FLUX_ERR_b{item + 1}" for item in range(4)],
                                      [f"ML_FLUX_ERR_b{item + 1}" for item in range(4)]],
                "energy_band_center": [0.35, 0.75, 1.5, 3.25],
                "energy_band_half_width": [0.15, 0.25, 0.5, 1.25],
                "hr_bandlimit_index": 3,
                "band_conv_factor_soft": 0.35/0.35,
                "band_conv_factor_hard": 0.65/0.24,
                "hr_track_marker": "^"},
    "Slew": {"flux_obs": "Flux",
             "flux_obs_err": ["FluxErr", "FluxErr"],
             "conv_factor": 1/0.999,
             "time_name": "DATE_OBS",
             "obsid_name": np.nan,
             "band_flux_obs": ["Flux6", "Flux7"],
             "band_flux_obs_err": [["Flux6Err", "Flux7Err"],
                                   ["Flux6Err", "Flux7Err"]],
             "energy_band_center": [1.1, 7],
             "energy_band_half_width": [0.9, 5],
             "hr_bandlimit_index": 1,
             "band_conv_factor_soft": 0.35/0.35,
             "band_conv_factor_hard": 0.65/0.65,
             "hr_track_marker": "P"},
    "Stacked": {"flux_obs": "EP_FLUX",
                "flux_obs_err": ["EP_FLUX_ERR", "EP_FLUX_ERR"],
                "conv_factor": 1/0.999,
                "time_name": "MJD_FIRST",
                "obsid_name": "OBS_ID",
                "band_flux_obs": [f"EP_{item + 1}_FLUX" for item in range(5)],
                "band_flux_obs_err": [[f"EP_{item + 1}_FLUX_ERR" for item in range(5)],
                                      [f"EP_{item + 1}_FLUX_ERR" for item in range(5)]],
                "energy_band_center": [0.35, 0.75, 1.5, 3.25, 8.25],
                "energy_band_half_width": [0.15,0.25,0.5,1.25,3.75],
                "hr_bandlimit_index": 3,
                "band_conv_factor_soft": 0.35/0.35,
                "band_conv_factor_hard": 0.65/0.65,
                "hr_track_marker": "*"},
    "RASS": {"flux_obs": "Flux",
             "flux_obs_err": ["FluxErr","FluxErr"],
             "conv_factor": 1/0.35,
             "time_name": "OBS_DATE_1",
             "obsid_name": np.nan,
             "band_flux_obs": ["Flux1","Flux3","Flux4"],
             "band_flux_obs_err": [["FluxErr1","FluxErr3","FluxErr4"],
                                   ["FluxErr1","FluxErr3","FluxErr4"]],
             "energy_band_center": [0.25,0.7,1.45],
             "energy_band_half_width": [0.15,0.2,0.55],
             "hr_bandlimit_index": 3,
             "band_conv_factor_soft": 0.35/0.35,
             "band_conv_factor_hard": np.nan,
             "hr_track_marker": "d"},
    "WGACAT": {"flux_obs": "Flux",
               "flux_obs_err": ["FluxErr","FluxErr"],
               "conv_factor": 1/0.35,
               "time_name": "StartDate",
               "obsid_name": np.nan,
               "band_flux_obs": [f"Flux{item + 1}" for item in range(3)],
               "band_flux_obs_err": [[f"FluxErr{item + 1}" for item in range(3)],
                                     [f"FluxErr{item + 1}" for item in range(3)]],
               "energy_band_center": [0.25,0.7,1.45],
               "energy_band_half_width": [0.15,0.2,0.55],
               "hr_bandlimit_index": 3,
               "band_conv_factor_soft": 0.35/0.35,
               "band_conv_factor_hard": np.nan,
               "hr_track_marker": "d"},
    }

band_edges = {}
for cat in catalogs:
    band_edges[cat] = [center-width for (center, width) in zip(dictionary_catalog[cat]["energy_band_center"], dictionary_catalog[cat]["energy_band_half_width"])]
    band_edges[cat].append(dictionary_catalog[cat]["energy_band_center"][-1] + dictionary_catalog[cat]["energy_band_half_width"][-1])

frequencies = {}
frequencies_half_width = {}
for cat in catalogs:
    frequencies[cat] = [2.41e17*center for center in dictionary_catalog[cat]["energy_band_center"]]
    frequencies_half_width[cat] = [2.41e17*width for width in dictionary_catalog[cat]["energy_band_half_width"]]

xband_average_frequency = 2 * 2.41e17 #2keV in Hz, to compute alpha_OX
xband_width = 11.9 * 2.41e17 #11.9 keV in Hz, to compute alpha_OX
