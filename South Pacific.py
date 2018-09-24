# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 15:45:52 2018

@author: lille

SOUTH PACIFIC

Aims:
    - Import MATLAB files related to South Pacific basin
    - Plot initial data
    - Perform Meanshift clustering
    - Calculate BIC scores
    - Perform GaussianMixture clustering
    - Perform BeyesianGaussianMixture clustering
    - Extract Silhoutte coefficient
    - Extract Calinski-Harabaz scores
    - Repeat steps producing plots for entire domain and specific selections
"""

# Importing modules
import time
from itertools import cycle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import scipy.io as sio
from sklearn.preprocessing import scale
from sklearn import mixture
from sklearn.cluster import MeanShift
from sklearn import metrics

# importing package
from mypackage import MS_plot, GMM_plot, BGMM_plot



# To close and replace any existing figures
plt.close("all")



# To record runtime
START_TIME = time.clock()

###############################################################################

# Data only needs to be loaded and shaped for first run of script
LOAD_OPTION = input("Load Argo data for South Pacific? (y/[n])\n")
if LOAD_OPTION == "y":
    
    ###########################################################################
    
    # Loading Argo Data
    
    # Loading all MATLAB files (saved to same directory as script)
    SAVE_DATE = sio.loadmat("save_date.mat")
    SAVE_LON = sio.loadmat("save_lon.mat")
    SAVE_LONG = sio.loadmat("save_long.mat")
    SAVE_LATG = sio.loadmat("save_latg.mat")
    SAVE_LAT = sio.loadmat("save_lat.mat")
    SAVE_PRESGRID = sio.loadmat("save_presgrid.mat")
    SAVE_DY = sio.loadmat("save_dy.mat")
    SAVE_PSAL_CLIM_PRES = sio.loadmat("save_psal_clim_pres.mat")
    SAVE_PSAL_MAP_PRES = sio.loadmat("save_psal_map_pres.mat")
    TESTFILE_1 = sio.loadmat("save_psal_map_sig_interp_pres.mat")
    """
    # contains potential salinity:
    #     - pressure (100)
    #     - longitude (56)
    #     - latitude (23)
    #     - time (496)
    """
    TESTFILE_2 = sio.loadmat("save_temp_map_sig_interp_pres.mat")
    """
    # contains temperature:
    #     - pressure (100)
    #     - longitude (56)
    #     - latitude (23)
    #     - time (496)
    """
    
    
    
    # Extracting the date using headers in the dictionary
    DATE = SAVE_DATE["date"]
    LON = SAVE_LON["lon"]
    LONG = SAVE_LONG["long"]
    LAT = SAVE_LAT["lat"]
    LATG = SAVE_LATG["latg"]
    PRES = SAVE_PRESGRID["presgrid"]
    DY = SAVE_DY["dy"]
    PSAL_CLIM_PRES = SAVE_PSAL_CLIM_PRES["psal_clim_pres"]
    PSAL_MAP_PRES = SAVE_PSAL_MAP_PRES["psal_map_pres"]
    
    SET_A = TESTFILE_1["psal_map_sig_interp_pres"]
    SET_B = TESTFILE_2["temp_map_sig_interp_pres"]
    
    ###########################################################################
    
    # Manipulating Argo data
    
    # Averaging SET_A (salinity) and SET_B (temperature) in longitude and time
    FEATURES_A = np.nanmean(SET_A, axis=(1, 3))
    FEATURES_B = np.nanmean(SET_B, axis=(1, 3))
    
    # Tile latitudes to same shape as FEATURES_A and FEATURES_B
    LATITUDE = np.tile(LATG, (PRES.shape[0], 1))
    # Tile pressures to same shape as FEATURES_A and FEATURES_B
    PRESSURE = np.tile(PRES, (1, LATG.shape[1]))
    
    # Reshaping FEATURES_A image to 1D array containing all salinities
    PSALS = np.reshape(FEATURES_A, ((PRES.shape[0]*LATG.shape[1]), ))
    # Reshaping FEATURES_B image to 1D array of all temperatures
    TEMPS = np.reshape(FEATURES_B, ((PRES.shape[0]*LATG.shape[1]), ))
    # Standardising both with inbuilt scikit-learn function
    PSALS = scale(PSALS)
    TEMPS = scale(TEMPS)
    
    # Putting both features into one array with shape (n_samples, n_features)
    FEATURES_C = np.column_stack((TEMPS, PSALS))
    
    ###########################################################################
    
    # Loading TEOS-10 data
    
    # Coefficient of thermal expansion for Argo data points
    ALPHA = sio.loadmat("alpha.mat")
    ALPHA = ALPHA["alpha"]
    ALPHA = np.nanmean(ALPHA, axis=(1, 3))
    ALPHAS = np.reshape(ALPHA, ((PRES.shape[0]*LATG.shape[1]), ))
    
    # Coefficient of haline concentration for Argo data points
    BETA = sio.loadmat("beta.mat")
    BETA = BETA["beta"]
    BETA = np.nanmean(BETA, axis=(1, 3))
    BETAS = np.reshape(BETA, ((PRES.shape[0]*LATG.shape[1]), ))
    
    # Potential temperature for Argo data points
    POTENTIAL_TEMPERATURE = sio.loadmat("potential_temperature.mat")
    POTENTIAL_TEMPERATURE = POTENTIAL_TEMPERATURE["PT"]
    POTENTIAL_TEMPERATURE = np.nanmean(POTENTIAL_TEMPERATURE, axis=(1, 3))
    POTENTIAL_TEMPS = np.reshape(POTENTIAL_TEMPERATURE,
                                 ((PRES.shape[0]*LATG.shape[1]), ))
    
    # Potential density (0dbar reference density) for all Argo data points
    SIGMA0 = sio.loadmat("sigma0.mat")
    SET_SIGMA0 = SIGMA0["sigma0"]
    SIGMA0 = np.nanmean(SET_SIGMA0, axis=(1, 3))
    SIGMA0S = np.reshape(SIGMA0, ((PRES.shape[0]*LATG.shape[1]), ))
    SIGMA0S = scale(SIGMA0S)
    
    # Potential density (1000dbar reference density) for all Argo data points
    SIGMA1 = sio.loadmat("sigma1.mat")
    SIGMA1 = SIGMA1["sigma1"]
    SIGMA1 = np.nanmean(SIGMA1, axis=(1, 3))
    
    # Adding potential density as new feature
    FEATURES_D = np.column_stack((TEMPS, PSALS, SIGMA0S))

###############################################################################

# Making initial plot of raw Argo data

# Finding nearest integer to minimum potential density provided
PD_MIN = int(np.amin(np.reshape(SIGMA0, ((PRES.shape[0]*LATG.shape[1]), ))))
# Adjusting to encompass all potential density values
if PD_MIN > np.amin(np.reshape(SIGMA0, ((PRES.shape[0]*LATG.shape[1]), ))):
    PD_MIN = PD_MIN-1
# Findinging nearest integer to maximum potential density provided
PD_MAX = int(np.amax(np.reshape(SIGMA0, ((PRES.shape[0]*LATG.shape[1]), ))))
# Adjusting to encompass asll potential density values
if PD_MAX < np.amax(np.reshape(SIGMA0, ((PRES.shape[0]*LATG.shape[1]), ))):
    PD_MAX = PD_MAX+1
# Defining potential density step size between levels
STEP = 0.45
# Calculating levels to plot potential density surfaces at
LEVELS = np.arange(PD_MIN, PD_MAX + STEP, STEP)
"""
# the un-scaled data is used here because actual values are plotted initially
# then the scaled data is used to produce less biased clustering
"""



# Plotting Argo zonally averaged potential salinity/ temperature (subplots)
plt.figure()
# Contour plot of salinity in the South Pacific
#plt.subplot(1, 2, 1)
plt.contourf(LATITUDE, PRESSURE, FEATURES_A)
CBAR = plt.colorbar()
CBAR.set_label("Salinity/ psu")
PDEN = plt.contour(LATITUDE, PRESSURE, SIGMA0, levels=LEVELS,
                   colors="w", linewidths=0.7)

# Formatting left subplot
plt.gca().invert_yaxis()
#plt.title("Salinity")
plt.xlabel("Latitude/ °N")
plt.ylabel("Pressure/ dbar")
plt.clabel(PDEN, inline=True, fmt = ("%0.2f" + " σ$_{0}$"),
           fontsize=9)



# Contour plot of temperature in the South Pacific
plt.figure()
#plt.subplot(1, 2, 2)
plt.contourf(LATITUDE, PRESSURE, FEATURES_B)
CBAR = plt.colorbar()
CBAR.set_label("Temperature/ °C")
PDEN = plt.contour(LATITUDE, PRESSURE, SIGMA0, levels=LEVELS,
                   colors="w", linewidths=0.7)

# Formatting right subplot
plt.gca().invert_yaxis()
#plt.title("Temperature")
plt.xlabel("Latitude/ °N")
plt.ylabel("Pressure/ dbar")
plt.clabel(PDEN, inline=True, fmt = ("%0.2f" + " σ$_{0}$"),
           fontsize=9)

# Formatting figure
#plt.suptitle("Zonal and Time Mean Temperature, Argo Interpolated Product (South Pacific)",
#             fontweight="semibold")
#plt.tight_layout()
#plt.subplots_adjust(wspace=0.9)

###############################################################################

# Meanshift Clustering

# Using only salinity as a feature to cluster using Meanshift
MS_plot(PSALS, "Salinity", "psu", False, FEATURES_A, LATITUDE, PRESSURE,
        LATG, PRES, "South Pacific")
# Defining variables from MS_plot function (salinity)
CLUSTNUM_A = MS_plot.CLUSTNUM
LABELS_PLT_A = MS_plot.LABELS_PLT
SILHOUETTE_A = MS_plot.SILHOUETTE
CALINSKI_HARABAZ_A = MS_plot.CALINSKI_HARABAZ

####################

# Using only temperature as a feature to cluster using Meanshift
MS_plot(TEMPS, "Temperature", " °C", False, FEATURES_B, LATITUDE, PRESSURE,
        LATG, PRES, "South Pacific")
# Defining variables from MS_plot function (temperature)
CLUSTNUM_B = MS_plot.CLUSTNUM
LABELS_PLT_B = MS_plot.LABELS_PLT
SILHOUETTE_B = MS_plot.SILHOUETTE
CALINSKI_HARABAZ_B = MS_plot.CALINSKI_HARABAZ

####################

# Using temperature and salinity to cluster data using Meanshift
MS_plot(FEATURES_C, "Temperature/ Salinity", " ", False, FEATURES_C,
        LATITUDE, PRESSURE, LATG, PRES, "South Pacific")
# Defining variables from MS_plot function (salinity, temperature)
CLUSTNUM_C = MS_plot.CLUSTNUM
LABELS_PLT_C = MS_plot.LABELS_PLT
SILHOUETTE_C = MS_plot.SILHOUETTE
CALINSKI_HARABAZ_C = MS_plot.CALINSKI_HARABAZ

####################

# Using temperature, salinity and density to cluster data
MS_plot(FEATURES_D, "Temperature/ Salinity/ Density", " ", False, FEATURES_D,
        LATITUDE, PRESSURE, LATG, PRES, "South Pacific")
# Defining variables from MS_plot function (salinity, temperature, density)
CLUSTNUM_D = MS_plot.CLUSTNUM
LABELS_PLT_D = MS_plot.LABELS_PLT

####################

# Plotting the 4 different clusterings performed on one subplot

# Definitions to iterate through in for loop
FIG = plt.figure()
CLUSTNUMS = np.array([CLUSTNUM_A, CLUSTNUM_B, CLUSTNUM_C, CLUSTNUM_D])
LABELS_PLTS = np.array([LABELS_PLT_A, LABELS_PLT_B, LABELS_PLT_C, LABELS_PLT_D])
TYPE = ["Salinity", "Temperature", "Temperature/ Salinity", "Temperature/ Salinity/ Density"]

# Defining extent of the subplots outside for loop for efficiency
EXTENT = [float(min(LATG.T)), float(max(LATG.T)),
          float(max(PRES)), float(min(PRES))]

# For loop replotting MS_plot results in a single figure
for i in range(len(TYPE)):
    
    # Each subplot will relate to one of the FEATURE arrays
    plt.subplot(2, 2, i+1)

    # Defining discrete colorbar to label each cluster
    CMAP = plt.cm.get_cmap("viridis", CLUSTNUMS[i])
    plt.imshow(LABELS_PLTS[i], cmap = CMAP, aspect="auto", extent=EXTENT)
    
    # Formatting subplots
    TITLE = TYPE[i] + " Clusters (South Pacific)\nEstimated Clusters: %d"
    plt.title(TITLE %CLUSTNUMS[i])
    plt.xlabel("Latitude/ °N")
    plt.ylabel("Pressure/ dbar")
    CBAR = plt.colorbar()
    CBAR_TITLE = TYPE[i] + " Clusters"
    CBAR.set_label(CBAR_TITLE)

# Manually adjusting space between subplots for neater figure
plt.subplots_adjust(hspace=0.4, wspace=0.4)

###############################################################################

# Calculating BIC Scores

# For only salinity as a feature
BIC_plot(PSALS, 1, 10, False, "Salinity", "Pacific")

# For only temperature as a feature
BIC_plot(TEMPS, 1, 10, False, "Temperature", "Pacific")

# For salinity and temperature as features
BIC_plot(FEATURES_C, 1, 10, False, "Temperature/ Salinity", "Pacific")

# For salinity, temperature and density as features
BIC_plot(FEATURES_D, 1, 10, False, "Temperature/ Salinity/ Density", "Pacific")

###############################################################################

# GMM Clustering

# Using salinity and temperature as features to cluster using GaussianMixture
GMM_plot(FEATURES_C, False, 2, 3, 5, 10, 1, LATITUDE, LATG, PRES)
# Defining scoring metrics from GMM_plot function (salinity, temperature)
GMM_S_SCORES_C = GMM_plot.SILHOUETTE_SCORES
GMM_CH_SCORES_C = GMM_plot.CALINSKI_HARABAZ_SCORES

# Using salinity, temperature and density as features to cluster using GaussianMixture
GMM_plot(FEATURES_D, False, 2, 3, 5, 10, 1, LATITUDE, LATG, PRES)
# Defining scoring metrics from GMM_plot function (salinity, temperature, density)
GMM_S_SCORES_D = GMM_plot.SILHOUETTE_SCORES
GMM_CH_SCORES_D = GMM_plot.CALINSKI_HARABAZ_SCORES

####################

# BGMM Clustering

# Using salinity and temperature as features to cluster using BayesianGaussianMixture
BGMM_plot(FEATURES_C, False, 2, 3, 5, 10, 1, LATITUDE, LATG, PRES)
# Defining scoring metrics from BGMM_plot function (salinity, temperature)
BGMM_S_SCORES_C = BGMM_plot.SILHOUETTE_SCORES
BGMM_CH_SCORES_C = BGMM_plot.CALINSKI_HARABAZ_SCORES

# Using salinity, temperature and density as features to cluster using BayesianGaussianMixture
BGMM_plot(FEATURES_D, False, 2, 3, 5, 10, 1, LATITUDE, LATG, PRES)
# Defining scoring metrics from BGMM_plot function (salinity, temperature, density)
BGMM_S_SCORES_D = BGMM_plot.SILHOUETTE_SCORES
BGMM_CH_SCORES_D = BGMM_plot.CALINSKI_HARABAZ_SCORES

###############################################################################

# Silhouette and Calinski-Harabaz scores with increasing components (GMM, BGMM)

# Defining range of components to plot scoring metrics over
COMPONENT_RANGE = np.arange(2, 21, 1)

# Defining empty arrays to assign Silhouette and Calinski-Harabaz scores to
GMM_SILHOUETTE_SCORES = np.empty(((max(COMPONENT_RANGE)-min(COMPONENT_RANGE)+1),))
GMM_CALINSKI_HARABAZ_SCORES = np.empty(((max(COMPONENT_RANGE)-min(COMPONENT_RANGE)+1),))

BGMM_SILHOUETTE_SCORES = np.empty(((max(COMPONENT_RANGE)-min(COMPONENT_RANGE)+1),))
BGMM_CALINSKI_HARABAZ_SCORES = np.empty(((max(COMPONENT_RANGE)-min(COMPONENT_RANGE)+1),))



# For loop to determine GMM Silhouette and Calinski-Harabaz scores for coponent range defined
for i in range(len(COMPONENT_RANGE)):
    
    # GMM
    
    # Perform clustering using GaussianMixture function
    GMM = mixture.GaussianMixture(n_components=COMPONENT_RANGE[i], covariance_type="full").fit(FEATURES_D)
    
    # 1D array containing GMM cluster labels corresponsing to each sample
    LABELS = GMM.predict(FEATURES_D)
    
    # Defining Silhouette Coefficient from cluster labels
    GMM_SILHOUETTE_SCORES[i] = metrics.silhouette_score(FEATURES_D, LABELS)
    # Defining Calinski-Harabz Score from cluster labels
    GMM_CALINSKI_HARABAZ_SCORES[i] = metrics.calinski_harabaz_score(FEATURES_D, LABELS)
    
    
    # BGMM
    
    # Perform clustering using BayesianGaussianMixture function
    BGMM = mixture.BayesianGaussianMixture(n_components=COMPONENT_RANGE[i], covariance_type="full").fit(FEATURES_D)
    
    # 1D array containing BGMM cluster labels corresponsing to each sample
    LABELS = BGMM.predict(FEATURES_D)
    
    # Defining Silhouette Coefficient from cluster labels
    BGMM_SILHOUETTE_SCORES[i] = metrics.silhouette_score(FEATURES_D, LABELS)
    # Defining Calinski-Harabz Score from cluster labels
    BGMM_CALINSKI_HARABAZ_SCORES[i] = metrics.calinski_harabaz_score(FEATURES_D, LABELS)

# Polynomial fit line as a very basic initial indication of trend
GMM_SILHOUETTE_POLYNOMIAL = np.poly1d(np.polyfit(COMPONENT_RANGE, GMM_SILHOUETTE_SCORES, 2))
GMM_CALINSKI_HARABAZ_POLYNOMIAL = np.poly1d(np.polyfit(COMPONENT_RANGE, GMM_CALINSKI_HARABAZ_SCORES, 2))
BGMM_SILHOUETTE_POLYNOMIAL = np.poly1d(np.polyfit(COMPONENT_RANGE, BGMM_SILHOUETTE_SCORES, 2))
BGMM_CALINSKI_HARABAZ_POLYNOMIAL = np.poly1d(np.polyfit(COMPONENT_RANGE, BGMM_CALINSKI_HARABAZ_SCORES, 2))

###############

# Plotting Silhouette, Calinski-Harabaz outputs (GMM, BGMM)

# Plotting GMM Silhouette Coefficient
plt.figure()
plt.plot(COMPONENT_RANGE, GMM_SILHOUETTE_SCORES, "r", label="GMM Silhouette Scores")
plt.plot(COMPONENT_RANGE, GMM_SILHOUETTE_POLYNOMIAL(COMPONENT_RANGE), "r--", linewidth=1, label="GMM 2$^n$$^d$ Order Polynomial", alpha=0.3)
# Plotting BGMM Silhouette Coefficient
plt.plot(COMPONENT_RANGE, BGMM_SILHOUETTE_SCORES, "b", label="BGMM Silhouette Scores")
plt.plot(COMPONENT_RANGE, BGMM_SILHOUETTE_POLYNOMIAL(COMPONENT_RANGE), "b--", linewidth=1, label="BGMM 2$^n$$^d$ Order Polynomial", alpha=0.3)

# Formatting Silhouette Coefficient plot
#plt.title("Silhouette Scores for GMM and BGMM Clustering with Polynomial Fit\n(Features: temperature, salinity, potential density)")
plt.xlabel("Number of Components")
plt.ylabel("Silhouette score")
plt.legend()
# Fixing number of component ticks to whole numbers
plt.xticks(COMPONENT_RANGE.tolist())



# Plotting GMM Calinski-Harbaz Score
plt.figure()
plt.plot(COMPONENT_RANGE, GMM_CALINSKI_HARABAZ_SCORES, "r", label="GMM Calinski-Harabaz Scores")
plt.plot(COMPONENT_RANGE, GMM_CALINSKI_HARABAZ_POLYNOMIAL(COMPONENT_RANGE), "r--", linewidth=1, label="GMM 2$^n$$^d$ Order Polynomial", alpha=0.3)
# Plotting BGMM Calinski-Harbaz Score
plt.plot(COMPONENT_RANGE, BGMM_CALINSKI_HARABAZ_SCORES, "b", label="BGMM Calinski-Harabaz Scores")
plt.plot(COMPONENT_RANGE, BGMM_CALINSKI_HARABAZ_POLYNOMIAL(COMPONENT_RANGE), "b--", linewidth = 1, label="BGMM 2$^n$$^d$ Order Polynomial", alpha=0.3)

# Formatting Calinski-Harbaz Score plot
#plt.title("Calinski-Harabaz Scores for GMM and BGMM Clustering with Polynomial Fit\n(Features: temperature, salinity, potential density)")
plt.xlabel("Number of Components")
plt.ylabel("Calinski-Harabaz score")
plt.legend()
# Fixing number of component ticks to whole numbers
plt.xticks(COMPONENT_RANGE.tolist())

###############################################################################

# Looking at specific latitude ranges within the original Argo domain

# Slices of the domain I want to look at
DOMAIN_SECTIONS = {}
DOMAIN_SECTIONS["north of -45"] = [None, None, (np.abs(LATG - (-45))).argmin(),
                                   None]
DOMAIN_SECTIONS["south of -45"] = [None, None, None,
                                   (np.abs(LATG - (-45))).argmin()]
DOMAIN_SECTIONS["north of -45, below 500"] = [(np.abs(PRES - (500))).argmin(),
                                              None,
                                              (np.abs(LATG - (-45))).argmin(),
                                              None]
DOMAIN_SECTIONS["north of -45, above 500"] = [(np.abs(LATG - (-45))).argmin(),
                                              None, None,
                                              (np.abs(PRES - (500))).argmin()]
DOMAIN_SECTIONS["north of -50, south of -40"] = [None, None,
                                                 (np.abs(LATG - (-50))).argmin(),
                                                 (np.abs(LATG - (-40))).argmin()]
"""
# list of arguments corresponding to:
#     - pressure minimum
#     - pressure maximum
#     - latitude minimum
#     - latitude maximum
"""

DOMAIN_SECTION_NAMES = ["north of -45", "south of -45",
                        "north of -45, below 500", "north of -45, above 500",
                        "north of -50, south of -40"]

# For loop to slice and reshape original Argo data so that sections can be analysed
for i in range(len(DOMAIN_SECTION_NAMES)):

    # New FEATURES_A and FEATURES_B containing data in relevent section only
    SECTION_FEATURES_A = FEATURES_A[DOMAIN_SECTIONS[DOMAIN_SECTION_NAMES[i]][0]:DOMAIN_SECTIONS[DOMAIN_SECTION_NAMES[i]][1],
                                    DOMAIN_SECTIONS[DOMAIN_SECTION_NAMES[i]][2]:DOMAIN_SECTIONS[DOMAIN_SECTION_NAMES[i]][3]]
    SECTION_FEATURES_B = FEATURES_B[DOMAIN_SECTIONS[DOMAIN_SECTION_NAMES[i]][0]:DOMAIN_SECTIONS[DOMAIN_SECTION_NAMES[i]][1],
                                    DOMAIN_SECTIONS[DOMAIN_SECTION_NAMES[i]][2]:DOMAIN_SECTIONS[DOMAIN_SECTION_NAMES[i]][3]]
    SECTION_SIGMA0 = SIGMA0[DOMAIN_SECTIONS[DOMAIN_SECTION_NAMES[i]][0]:DOMAIN_SECTIONS[DOMAIN_SECTION_NAMES[i]][1],
                            DOMAIN_SECTIONS[DOMAIN_SECTION_NAMES[i]][2]:DOMAIN_SECTIONS[DOMAIN_SECTION_NAMES[i]][3]]
        
    # Reshaping FEATURES_A image to 1D array containing relevent salinities
    SECTION_PSALS = np.reshape(SECTION_FEATURES_A,
                               ((SECTION_FEATURES_A.shape[0]*SECTION_FEATURES_A.shape[1]),
                                1))
    # Reshaping FEATURES_B image to 1D array containing relevent temperatures
    SECTION_TEMPS = np.reshape(SECTION_FEATURES_B,
                               ((SECTION_FEATURES_B.shape[0]*SECTION_FEATURES_B.shape[1]),
                                1))
    # Reshaping SIGMA0 image to 1D array containing relevent densities
    SECTION_SIGMA0S = np.reshape(SECTION_SIGMA0,
                               ((SECTION_SIGMA0.shape[0]*SECTION_SIGMA0.shape[1]),
                                1))
    
    # Standardising feature quantities
    SECTION_PSALS = scale(SECTION_PSALS)
    SECTION_TEMPS = scale(SECTION_TEMPS)
    SECTION_SIGMA0S = scale(SECTION_SIGMA0S)
    
    # Arrays containing features in shape (n_samples, n_features)
    SECTION_FEATURES_C = np.column_stack((SECTION_TEMPS, SECTION_PSALS))
    SECTION_FEATURES_D = np.column_stack((SECTION_TEMPS, SECTION_PSALS, SECTION_SIGMA0S))
    
    # Slicing to relevent latitudes and pressures as 1D arrays
    SECTION_LATG = LATG[:, DOMAIN_SECTIONS[DOMAIN_SECTION_NAMES[i]][2]:DOMAIN_SECTIONS[DOMAIN_SECTION_NAMES[i]][3]]
    SECTION_PRES = PRES[DOMAIN_SECTIONS[DOMAIN_SECTION_NAMES[i]][0]:DOMAIN_SECTIONS[DOMAIN_SECTION_NAMES[i]][1], :]
    
    # Tiling latitude and pressure arrays to correct shape to plot as image
    SECTION_LATITUDE = np.tile(SECTION_LATG, (SECTION_PRES.shape[0], 1))
    SECTION_PRESSURE = np.tile(SECTION_PRES, (1, SECTION_LATG.shape[1]))
    
    ###########################################################################
    
    # Making initial plot of section of Argo data
    
    # Finding nearest integer to minimum potential density provided
    PD_MIN = int(np.amin(np.reshape(SECTION_SIGMA0, ((SECTION_PRES.shape[0]*SECTION_LATG.shape[1]), ))))
    # Adjusting to encompass all potential density values
    if PD_MIN > np.amin(np.reshape(SECTION_SIGMA0, ((SECTION_PRES.shape[0]*SECTION_LATG.shape[1]), ))):
        PD_MIN = PD_MIN-1
    # Findinging nearest integer to maximum potential density provided
    PD_MAX = int(np.amax(np.reshape(SECTION_SIGMA0, ((SECTION_PRES.shape[0]*SECTION_LATG.shape[1]), ))))
    # Adjusting to encompass asll potential density values
    if PD_MAX < np.amax(np.reshape(SECTION_SIGMA0, ((SECTION_PRES.shape[0]*SECTION_LATG.shape[1]), ))):
        PD_MAX = PD_MAX+1
    # Defining potential density step size between levels
    STEP = 0.45
    # Calculating levels to plot potential density surfaces at
    LEVELS = np.arange(PD_MIN, PD_MAX + STEP, STEP)
    """
    # the un-scaled data is used here because actual values are plotted initially
    # then the scaled data is used to produce less biased clustering
    """
    
    
    
    # Plotting Argo zonally averaged potential salinity/ temperature (subplots)
    plt.figure()
    # Contour plot of salinity in the South Pacific
    #plt.subplot(1, 2, 1)
    plt.contourf(SECTION_LATITUDE, SECTION_PRESSURE, SECTION_FEATURES_A)
    CBAR = plt.colorbar()
    CBAR.set_label("Salinity/ psu")
    PDEN = plt.contour(SECTION_LATITUDE, SECTION_PRESSURE, SECTION_SIGMA0,
                       levels=LEVELS, colors="w", linewidths=0.7)
    
    # Formatting left subplot
    plt.gca().invert_yaxis()
    #plt.title("Salinity")
    plt.xlabel("Latitude/ °N")
    plt.ylabel("Pressure/ dbar")
    plt.clabel(PDEN, inline=True, fmt = ("%0.2f" + " σ$_{0}$"),
               fontsize="smaller")
    
    
    
    # Contour plot of temperature in the South Pacific
    plt.figure()
    #plt.subplot(1, 2, 2)
    plt.contourf(SECTION_LATITUDE, SECTION_PRESSURE, SECTION_FEATURES_B)
    CBAR = plt.colorbar()
    CBAR.set_label("Temperature/ °C")
    PDEN = plt.contour(SECTION_LATITUDE, SECTION_PRESSURE, SECTION_SIGMA0,
                       levels=LEVELS, colors="w", linewidths=0.7)
    
    # Formatting right subplot
    plt.gca().invert_yaxis()
    #plt.title("Temperature")
    plt.xlabel("Latitude/ °N")
    plt.ylabel("Pressure/ dbar")
    plt.clabel(PDEN, inline=True, fmt = ("%0.2f" + " σ$_{0}$"),
               fontsize="smaller")
    
    # Formatting figure
#    plt.suptitle("Zonal and Time Mean Temperature, Argo Interpolated Product (South Pacific)",
#                 fontweight="semibold")
#    plt.tight_layout()
    
    ###########################################################################
    
    # GMM Clustering
    
    # Using salinity and temperature as features to cluster using GaussianMixture
    GMM_plot(SECTION_FEATURES_C, False, 2, 3, 5, 10, 1, SECTION_LATITUDE,
             SECTION_LATG, SECTION_PRES)
    
    # Using salinity, temperature and density as features to cluster using GaussianMixture
    GMM_plot(SECTION_FEATURES_D, False, 2, 3, 5, 10, 1, SECTION_LATITUDE,
             SECTION_LATG, SECTION_PRES)
    
    ####################
    
    # BGMM Clustering
    
    # Using salinity and temperature as features to cluster using BayesianGaussianMixture
    BGMM_plot(SECTION_FEATURES_C, False, 2, 3, 5, 10, 1, SECTION_LATITUDE,
              SECTION_LATG, SECTION_PRES)
    
    # Using salinity, temperature and density as features to cluster using BayesianGaussianMixture
    BGMM_plot(FEATURES_D, False, 2, 3, 5, 10, 1, LATITUDE, LATG, PRES)

###############################################################################

# Seasonal datasets

# Defining blank list to append seasons (int labels) to in same shape as DY
SEASONS = np.empty(DY.shape)
"""
# 1 = Austral Summer
# 2 = Austral Autumn
# 3 = Austral Winter
# 4 = Austral Spring
"""

# Defining empty array to assign indicies used to label seasons in DY
INDXX = np.empty((252, 1), dtype = int)
COUNT = 0

# For loop sorting dates into labelled seasons
for i in range(DY.shape[1]):
    
    if ((DY[:, i] - abs(int(DY[:, i]))) < (2*(1/12))):
        SEASONS[:, i] = 1   # Austral Summer
        INDXX[COUNT] = i
        COUNT += 1
    elif ((DY[:, i] - abs(int(DY[:, i]))) > (11*(1/12))):
        SEASONS[:, i] = 1   # Austral Summer
        INDXX[COUNT] = i
        COUNT += 1
        
    elif ((2*(1/12)) < (DY[:, i] - abs(int(DY[:, i]))) < (5*(1/12))):
        SEASONS[:, i] = 2   # Austral Autumn
        INDXX[COUNT] = i
        COUNT += 1
        
    elif ((5*(1/12)) < (DY[:, i] - abs(int(DY[:, i]))) < (8*(1/12))):
        SEASONS[:, i] = 3   # Austral Winter
        
    else:
        SEASONS[:, i] = 4   # Austral Spring

# Redefining seasons array shape for easier data manipulation
SEAS_COL = np.reshape(SEASONS, (DY.shape[1], 1))

####################

# Redefining initial data to contain specific seasonal months only

# Austral summer
WINSET_A = np.empty((SET_A.shape[0], SET_A.shape[1], SET_A.shape[2],
                      int(sum(SEAS_COL == 1))))
WINSET_B = np.empty((SET_B.shape[0], SET_B.shape[1], SET_B.shape[2],
                     int(sum(SEAS_COL == 1))))
WINSET_SIGMA0 = np.empty((SET_SIGMA0.shape[0], SET_SIGMA0.shape[1],
                          SET_SIGMA0.shape[2], int(sum(SEAS_COL == 1))))

# Austral autumn
SPRSET_A = np.empty((SET_A.shape[0], SET_A.shape[1], SET_A.shape[2],
                      int(sum(SEAS_COL == 2))))
SPRSET_B = np.empty((SET_B.shape[0], SET_B.shape[1], SET_B.shape[2],
                     int(sum(SEAS_COL == 2))))
SPRSET_SIGMA0 = np.empty((SET_SIGMA0.shape[0], SET_SIGMA0.shape[1],
                          SET_SIGMA0.shape[2], int(sum(SEAS_COL == 2))))

# Austral winter
SUMSET_A = np.empty((SET_A.shape[0], SET_A.shape[1], SET_A.shape[2],
                      int(sum(SEAS_COL == 3))))
SUMSET_B = np.empty((SET_B.shape[0], SET_B.shape[1], SET_B.shape[2],
                     int(sum(SEAS_COL == 3))))
SUMSET_SIGMA0 = np.empty((SET_SIGMA0.shape[0], SET_SIGMA0.shape[1],
                          SET_SIGMA0.shape[2], int(sum(SEAS_COL == 3))))

# Austral spring
AUTSET_A = np.empty((SET_A.shape[0], SET_A.shape[1], SET_A.shape[2],
                      int(sum(SEAS_COL == 4))))
AUTSET_B = np.empty((SET_B.shape[0], SET_B.shape[1], SET_B.shape[2],
                     int(sum(SEAS_COL == 4))))
AUTSET_SIGMA0 = np.empty((SET_SIGMA0.shape[0], SET_SIGMA0.shape[1],
                          SET_SIGMA0.shape[2], int(sum(SEAS_COL == 4))))

####################

# Defining empty arrays for indices of elements relevent to each season

# Austral summer
WININDX_A = np.empty((WINSET_A.shape[3], 1), dtype = int)
WININDX_B = np.empty((WINSET_B.shape[3], 1), dtype = int)
WININDX_SIGMA0 = np.empty((WINSET_SIGMA0.shape[3], 1), dtype = int)

# Austral autumn
SPRINDX_A = np.empty((SPRSET_A.shape[3], 1), dtype = int)
SPRINDX_B = np.empty((SPRSET_B.shape[3], 1), dtype = int)
SPRINDX_SIGMA0 = np.empty((SPRSET_SIGMA0.shape[3], 1), dtype = int)

# Austral summer
SUMINDX_A = np.empty((SUMSET_A.shape[3], 1), dtype = int)
SUMINDX_B = np.empty((SUMSET_B.shape[3], 1), dtype = int)
SUMINDX_SIGMA0 = np.empty((SUMSET_SIGMA0.shape[3], 1), dtype = int)

# Austral spring
AUTINDX_A = np.empty((AUTSET_A.shape[3], 1), dtype = int)
AUTINDX_B = np.empty((AUTSET_B.shape[3], 1), dtype = int)
AUTINDX_SIGMA0 = np.empty((AUTSET_SIGMA0.shape[3], 1), dtype = int)

# For loop assigning indices of elements relevent to each season
COUNT1 = COUNT2 = COUNT3 = COUNT4 = 0
for i in range(len(SEAS_COL)):
            if SEAS_COL[i] == 1:
                WININDX_A[COUNT1] = WININDX_B[COUNT1] = WININDX_SIGMA0[COUNT1] = i
                COUNT1 += 1
            if SEAS_COL[i] == 2:
                SPRINDX_A[COUNT2] = SPRINDX_B[COUNT2] = SPRINDX_SIGMA0[COUNT2] = i
                COUNT2 += 1
            if SEAS_COL[i] == 3:
                SUMINDX_A[COUNT3] = SUMINDX_B[COUNT3] = SUMINDX_SIGMA0[COUNT3] = i
                COUNT3 += 1
            if SEAS_COL[i] == 4:
                AUTINDX_A[COUNT4] = AUTINDX_B[COUNT4] = AUTINDX_SIGMA0[COUNT4] = i
                COUNT4 += 1
                
####################

# redefinging feature arrays to contain only seasonal data

# Austral summer
WINSET_A = SET_A[..., WININDX_A][..., 0]
WINFEATURES_A = np.nanmean(WINSET_A, axis = (1,3))
WINSET_B = SET_B[..., WININDX_B][..., 0]
WINFEATURES_B = np.nanmean(WINSET_B, axis = (1,3))
WINSET_SIGMA0 = SET_SIGMA0[..., WININDX_SIGMA0][..., 0]
WIN_SIGMA0 = np.nanmean(WINSET_SIGMA0, axis = (1,3))

# Austral autumn
SPRSET_A = SET_A[..., SPRINDX_A][..., 0]
SPRFEATURES_A = np.nanmean(SPRSET_A, axis = (1,3))
SPRSET_B = SET_B[..., SPRINDX_B][..., 0]
SPRFEATURES_B = np.nanmean(SPRSET_B, axis = (1,3))
SPRSET_SIGMA0 = SET_SIGMA0[..., SPRINDX_SIGMA0][..., 0]
SPR_SIGMA0 = np.nanmean(SPRSET_SIGMA0, axis = (1,3))

# Austral winter
SUMSET_A = SET_A[..., SUMINDX_A][..., 0]
SUMFEATURES_A = np.nanmean(SUMSET_A, axis = (1,3))
SUMSET_B = SET_B[..., SUMINDX_B][..., 0]
SUMFEATURES_B = np.nanmean(SUMSET_B, axis = (1,3))
SUMSET_SIGMA0 = SET_SIGMA0[..., SUMINDX_SIGMA0][..., 0]
SUM_SIGMA0 = np.nanmean(SUMSET_SIGMA0, axis = (1,3))

# Austral spring
AUTSET_A = SET_A[..., AUTINDX_A][..., 0]
AUTFEATURES_A = np.nanmean(AUTSET_A, axis = (1,3))
AUTSET_B = SET_B[..., AUTINDX_B][..., 0]
AUTFEATURES_B = np.nanmean(AUTSET_B, axis = (1,3))
AUTSET_SIGMA0 = SET_SIGMA0[..., AUTINDX_SIGMA0][..., 0]
AUT_SIGMA0 = np.nanmean(AUTSET_SIGMA0, axis = (1,3))

###############################################################################

# Plotting unclustered seasonal Argo data in same way as before

# Defining lists to iterate through temperature, salinity and density profiles
SEASONS_FEATURES_A = [WINFEATURES_A, SPRFEATURES_A, SUMFEATURES_A, AUTFEATURES_A]
SEASONS_FEATURES_B = [WINFEATURES_B, SPRFEATURES_B, SUMFEATURES_B, AUTFEATURES_B]
SEASONS_SIGMA0 = [WIN_SIGMA0, SPR_SIGMA0, SUM_SIGMA0, AUT_SIGMA0]
SEASON_NAMES = ["Austral Summer", "Austral Autumn",
                "Austral Winter", "Austral Spring"]

# Plotting potential salinity with constant density surfaces seasonally
#FIG1 = plt.figure()
#GRID1 = plt.GridSpec(2, 2, hspace=0.5, wspace=0.4)
INDX1 = np.repeat(np.arange(0, 2, 1), 2)
INDX2 = np.tile(np.arange(0, 2, 1), 2)
COUNT = 0
for i in range(len(SEASONS_FEATURES_A)):
    
    plt.figure()
    
    # Finding nearest integer to minimum potential density provided
    PD_MIN = int(np.nanmin(np.reshape(SEASONS_SIGMA0[i],
                ((SEASONS_SIGMA0[i].shape[0]*SEASONS_SIGMA0[i].shape[1]), ))))
    # Adjusting to encompass all potential density values
    if PD_MIN > np.nanmin(np.reshape(SEASONS_SIGMA0[i],
                        ((SEASONS_SIGMA0[i].shape[0]*SEASONS_SIGMA0[i].shape[1]), ))):
        PD_MIN = PD_MIN-1
    # Findinging nearest integer to maximum potential density provided
    PD_MAX = int(np.nanmax(np.reshape(SEASONS_SIGMA0[i],
                ((SEASONS_SIGMA0[i].shape[0]*SEASONS_SIGMA0[i].shape[1]), ))))
    # Adjusting to encompass asll potential density values
    if PD_MAX < np.nanmax(np.reshape(SEASONS_SIGMA0[i],
                        ((SEASONS_SIGMA0[i].shape[0]*SEASONS_SIGMA0[i].shape[1]), ))):
        PD_MAX = PD_MAX+1
    # Defining potential density step size between levels
    STEP = 0.45
    # Calculating levels to plot potential density surfaces at
    LEVELS = np.arange(PD_MIN, PD_MAX + STEP, STEP)
    
    #plt.subplot(GRID1[INDX1[COUNT], INDX2[COUNT]])
    PDEN = plt.contour(LATITUDE, PRESSURE, SEASONS_SIGMA0[i], levels=LEVELS,
                       colors="w", linewidths=0.7)
    plt.contourf(LATITUDE, PRESSURE, SEASONS_FEATURES_A[i])
    
    # Formatting figure
    plt.gca().invert_yaxis()
    #plt.title(SEASON_NAMES[i])
    plt.xlabel("Latitude/ °N")
    plt.ylabel("Pressure/ dbar")
    plt.clabel(PDEN, inline=True, fmt = ("%0.2f" + " σ$_{0}$"),
           fontsize=6)
    plt.colorbar(label = "Salinity/ psu")
    
    COUNT += 1

# Adjusting salinity subplots to add colorbar
#FIG1.subplots_adjust(right=0.8)
#plt.colorbar(label = "Salinity", 
#             cax = FIG1.add_axes([0.85, 0.1, 0.015, 0.8]))

# Plotting temperature with constant density surfaces seasonally
FIG2 = plt.figure()
GRID2 = plt.GridSpec(2, 2, hspace=0.5, wspace=0.4)
COUNT = 0
for i in range(len(SEASONS_FEATURES_B)):
    
    plt.figure()
    
    # Finding nearest integer to minimum potential density provided
    PD_MIN = int(np.nanmin(np.reshape(SEASONS_SIGMA0[i],
                ((SEASONS_SIGMA0[i].shape[0]*SEASONS_SIGMA0[i].shape[1]), ))))
    # Adjusting to encompass all potential density values
    if PD_MIN > np.nanmin(np.reshape(SEASONS_SIGMA0[i],
                        ((SEASONS_SIGMA0[i].shape[0]*SEASONS_SIGMA0[i].shape[1]), ))):
        PD_MIN = PD_MIN-1
    # Findinging nearest integer to maximum potential density provided
    PD_MAX = int(np.nanmax(np.reshape(SEASONS_SIGMA0[i],
                ((SEASONS_SIGMA0[i].shape[0]*SEASONS_SIGMA0[i].shape[1]), ))))
    # Adjusting to encompass asll potential density values
    if PD_MAX < np.nanmax(np.reshape(SEASONS_SIGMA0[i],
                        ((SEASONS_SIGMA0[i].shape[0]*SEASONS_SIGMA0[i].shape[1]), ))):
        PD_MAX = PD_MAX+1
    # Defining potential density step size between levels
    STEP = 0.45
    # Calculating levels to plot potential density surfaces at
    LEVELS = np.arange(PD_MIN, PD_MAX + STEP, STEP)
    
    #plt.subplot(GRID2[INDX1[COUNT], INDX2[COUNT]])
    PDEN = plt.contour(LATITUDE, PRESSURE, SEASONS_SIGMA0[i], levels=LEVELS,
                       colors="w", linewidths=0.7)
    plt.contourf(LATITUDE, PRESSURE, SEASONS_FEATURES_B[i])
    
    # Formatting figure
    plt.gca().invert_yaxis()
    #plt.title(SEASON_NAMES[i])
    plt.xlabel("Latitude/ °N")
    plt.ylabel("Pressure/ dbar")
    plt.clabel(PDEN, inline=True, fmt = ("%0.2f" + " σ$_{0}$"),
           fontsize=6)
    plt.colorbar(label = "Temperature/ °C")
    
    COUNT += 1

# Adjusting temperature subplots to add colorbar
#FIG2.subplots_adjust(right=0.8)
#plt.colorbar(label = "Temperature", 
#             cax = FIG2.add_axes([0.85, 0.1, 0.015, 0.8]))

###############################################################################

# Clustering seasonal data

# Define features with sliced seasonal data (temperature, salinity, density)
SUMFEATURES_D = np.column_stack((np.reshape(SUMFEATURES_B,
                                            (SUMFEATURES_B.shape[0]*SUMFEATURES_B.shape[1],
                                             1)),
                                 np.reshape(SUMFEATURES_A,
                                            (SUMFEATURES_A.shape[0]*SUMFEATURES_A.shape[1],
                                             1)),
                                 np.reshape(SEASONS_SIGMA0[2],
                                            (SEASONS_SIGMA0[2].shape[0]*SEASONS_SIGMA0[2].shape[1],
                                             1))))

AUTFEATURES_D = np.column_stack((np.reshape(AUTFEATURES_B,
                                            (AUTFEATURES_B.shape[0]*AUTFEATURES_B.shape[1],
                                             1)),
                                 np.reshape(AUTFEATURES_A,
                                            (AUTFEATURES_A.shape[0]*AUTFEATURES_A.shape[1],
                                             1)),
                                 np.reshape(SEASONS_SIGMA0[3],
                                            (SEASONS_SIGMA0[3].shape[0]*SEASONS_SIGMA0[3].shape[1],
                                             1))))
"""
# only performing clustering on austral winter and spring because of the
# erroneous nan values seen in the other 2 seasons
"""

# Putting seasonal features into a dictionary to iterate through
SEASONAL_FEATURES = {}
SEASONAL_FEATURES["Austral winter"] = SUMFEATURES_D
SEASONAL_FEATURES["Austral spring"] = AUTFEATURES_D
SEASONAL_FEATURE_NAMES = ["Austral winter", "Austral spring"]

# For loop clustering seasons provided for all features
for i in range(len(SEASONAL_FEATURE_NAMES)):
    
    # Meashift clustering
    
    MS_plot(SEASONAL_FEATURES[SEASONAL_FEATURE_NAMES[i]],
            "Temperature/ Salinity/ Density", " ", False,
            SEASONAL_FEATURES[SEASONAL_FEATURE_NAMES[i]], LATITUDE, PRESSURE,
            LATG, SEASONAL_FEATURES[SEASONAL_FEATURE_NAMES[i]][0],
            "Pacific")
    
    # GMM clustering
    GMM_plot(SEASONAL_FEATURES[SEASONAL_FEATURE_NAMES[i]],
             False, 1, 3, 4, 8, 2, LATITUDE, LATG, PRES)
    plt.subplots_adjust(wspace=0.4)
    
    # BGMM clustering
    BGMM_plot(SEASONAL_FEATURES[SEASONAL_FEATURE_NAMES[i]],
             False, 1, 3, 4, 8, 2, LATITUDE, LATG, PRES)
    plt.subplots_adjust(wspace=0.4)

###############################################################################

# Output runtime
print("\nruntime = ", time.clock() - START_TIME, " s\n")
