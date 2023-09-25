import Function as F
from CatalogClass import XmmCatalog as Xmm
from CatalogClass import Xmm2Athena as X2A

from astropy.table import Table

# -------------------------------------------------- #

            ## PULSARS TABLE
            
PSRfullnames   = ['PSR J0437-4715','PSR J2124-3358','PSR J0751+1807']
PSRshortnames  = ['PSR0437','PSR2124','PSR0751']
PSRcountrates  = [1.319,0.1,0.025]
PSRtable = Table([PSRfullnames, PSRshortnames, PSRcountrates], 
                 names=('FullName', 'ShortName', 'Count Rate'))

Pulsar = 'PSR J0751+1807'

# -------------------------------------------------- #
# -------------------------------------------------- #
# -------------------------------------------------- #
# -------------------------------------------------- #

# Import Nicer parameters
file_path = "Catalog/NICER_PSF.dat"
EffArea, OffAxisAngle = Xmm.NicerParameters(file_path)

                    # PARAMETRES SIMULATION

SIM_parameters = {"Pulsar": Pulsar, 
                  "PSRtable": PSRtable,
                  "PSRposition": F.GetCoordPSR(Pulsar),
                  "INSTbkgd": 0.2,
                  "EXPtime": 1e6,
                  "EffArea": EffArea,
                  "OffAxisAngle":OffAxisAngle
                  }

# -------------------------------------------------- #

                # Load Catalogs

XmmDR13_path = "Catalog/4XMM_slim_DR13cat_v1.0.fits"
X2A_path = 'Catalog/xmm2athena_D6.1_V3.fits'
XmmDR11_path = 'Catalog/4XMM_DR11cat_v1.0.fits'

Catalog = Xmm.open_catalog(XmmDR13_path)

Xmm2Ath_Table, XmmDR11_Table = X2A.open_catalog(X2A_path, XmmDR11_path)

# -------------------------------------------------- #


# -------------------------------------------------- #

                # FIND NEARBY SOURCES

SRCposition = Xmm.TransformCoord2SkyCoord(Catalog)

NearbySource = F.FindNearbySources(Catalog, SRCposition, Pulsar)

# -------------------------------------------------- #


# -------------------------------------------------- #

                # CREATION OF NEARBY SOURCES TABLE WITH NH AND PHOTON INDEX

NearbySource_Table, Nearby_SRCposition = Xmm.CreateNearbySourceTable(NearbySource, Catalog)
NearbySource_Table = X2A.NearbySourcesTable_With_X2M_data(NearbySource_Table, XmmDR11_Table, Xmm2Ath_Table)

# -------------------------------------------------- #


# -------------------------------------------------- #

                # VALUE OF COUNT RATES AND ADD COLUMNS IN NEARBY SOURCES TABLE

Count_Rate, NearbySource_Table = Xmm.CountRates(NearbySource_Table)
SIM_parameters['NearbySource_Table'] = NearbySource_Table

# -------------------------------------------------- #


# -------------------------------------------------- #

                # VISUALIZED DATA MATPLOTLIB WITHOUT S/N

F.VisualizeData(Pulsar, NearbySource_Table)

# -------------------------------------------------- #


# -------------------------------------------------- #

                # VISUALIZED DATA MATPLOTLIB WITH S/N AND OPTIMAL POINT

F.NominalPointingInfo(SIM_parameters, Nearby_SRCposition)
OptimalPointingIdx, SRCoptimalSEPAR, SRCoptimalRATES, Vector_Dictionary = F.CalculateOptiPoint(SIM_parameters, Nearby_SRCposition)
F.OptimalPointInfos(Vector_Dictionary, OptimalPointingIdx, SRCoptimalRATES)
F.DataMap(SIM_parameters, Vector_Dictionary, OptimalPointingIdx, Nearby_SRCposition)

# -------------------------------------------------- #