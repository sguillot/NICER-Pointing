import Function as F
from CatalogClass import XmmCatalog as Xmm
from astropy.table import Table

# -------------------------------------------------- #

            ## PULSARS TABLE
            
PSRfullnames   = ['PSR J0437-4715','PSR J2124-3358','PSR J0751+1807']
PSRshortnames  = ['PSR0437','PSR2124','PSR0751']
PSRcountrates  = [1.319,0.1,0.025]
PSRtable = Table([PSRfullnames, PSRshortnames, PSRcountrates], 
                 names=('FullName', 'ShortName', 'Count Rate'))

Pulsar = 'PSR J0437-4715'

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

# TEMPS COMILATION ~6 MIN 30

file_name = "Catalog/4XMM_slim_DR13cat_v1.0.fits"

Catalog = Xmm.open_catalog(file_name)

SRCposition = Xmm.TransformCoord2SkyCoord(Catalog)

NearbySource = F.FindNearbySources(Catalog, SRCposition, Pulsar)

NearbySource_Table, Nearby_SRCposition = Xmm.CreateNearbySourceTable(NearbySource, Catalog)

Count_Rate, NearbySource_Table = Xmm.CountRates(NearbySource_Table)

F.VisualizeData(Pulsar, NearbySource_Table)

SIM_parameters['NearbySource_Table'] = NearbySource_Table

F.NominalPointingInfo(SIM_parameters, Nearby_SRCposition)

OptimalPointingIdx, SRCoptimalSEPAR, SRCoptimalRATES, Vector_Dictionary = F.CalculateOptiPoint(SIM_parameters, Nearby_SRCposition)
F.OptimalPointInfos(Vector_Dictionary, OptimalPointingIdx, SRCoptimalRATES)

F.DataMap(SIM_parameters, Vector_Dictionary, OptimalPointingIdx, Nearby_SRCposition)