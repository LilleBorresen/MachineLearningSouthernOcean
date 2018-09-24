# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 21:04:41 2018

@author: lille

Aims:
    -
"""

# Importing modules
import time
import numpy as np
import scipy.io as sio
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

# Importing package
from mypackage import MS_plot, GMM_plot, BGMM_plot



# To close and replace any existing figures
plt.close("all")



# To record runtime
START_TIME = time.clock()

###############################################################################

# Loading matlab files (saved to same directory as script)
#ACCESS1 = pickle.load(open("CMIP5_HIST_ACCESS1-0_1976_2005_mean.pck", "rb"), encoding = "latin1")
#bcc_csm1_1 = pickle.load(open("CMIP5_HIST_bcc-csm1-1_1976_2005_mean.pck", "rb"), encoding = "latin1")
#bcc_csm1_1_m = pickle.load(open("CMIP5_HIST_bcc-csm1-1-m_1976_2005_mean.pck", "rb"), encoding = "latin1")
#CanESM2 = pickle.load(open("CMIP5_HIST_CanESM2_1976_2005_mean.pck", "rb"), encoding = "latin1")
#CESM1_BGC = pickle.load(open("CMIP5_HIST_CESM1-BGC_1976_2005_mean.pck", "rb"), encoding = "latin1")
#CESM1_CAM5 = pickle.load(open("CMIP5_HIST_CESM1-CAM5_1976_2005_mean.pck", "rb"), encoding = "latin1")
#CESM1_FASTCHEM = pickle.load(open("CMIP5_HIST_CESM1-FASTCHEM_1976_2005_mean.pck", "rb"), encoding = "latin1")
#CESM1_WACCM = pickle.load(open("CMIP5_HIST_CESM1-WACCM_1976_2005_mean.pck", "rb"), encoding = "latin1")
#CMCC_CESM = pickle.load(open("CMIP5_HIST_CMCC-CESM_1976_2005_mean.pck", "rb"), encoding = "latin1")
#CMCC_CM = pickle.load(open("CMIP5_HIST_CMCC-CM_1976_2005_mean.pck", "rb"), encoding = "latin1")
#CMCC_CMS = pickle.load(open("CMIP5_HIST_CMCC-CMS_1976_2005_mean.pck", "rb"), encoding = "latin1")
#CNRM_CM5 = pickle.load(open("CMIP5_HIST_CNRM-CM5_1976_2005_mean.pck", "rb"), encoding = "latin1")
#CNRM_CM5_2 = pickle.load(open("CMIP5_HIST_CNRM-CM5-2_1976_2005_mean.pck", "rb"), encoding = "latin1")
#CSIRO_Mk3_6_0 = pickle.load(open("CMIP5_HIST_CSIRO-Mk3-6-0_1976_2005_mean.pck", "rb"), encoding = "latin1")
#EC_EARTH = pickle.load(open("CMIP5_HIST_EC-EARTH_1976_2005_mean.pck", "rb"), encoding = "latin1")
#GFDL_CM2p1 = pickle.load(open("CMIP5_HIST_GFDL-CM2p1_1976_2005_mean.pck", "rb"), encoding = "latin1")
GFDL_ESM2G = pickle.load(open("CMIP5_HIST_GFDL-ESM2G_1976_2005_mean.pck", "rb"), encoding = "latin1")
#GFDL_ESM2M = pickle.load(open("CMIP5_HIST_GFDL-ESM2M_1976_2005_mean.pck", "rb"), encoding = "latin1")
#GISS_E2_H = pickle.load(open("CMIP5_HIST_GISS-E2-H_1976_2005_mean.pck", "rb"), encoding = "latin1")
#GISS_E2_H_CC = pickle.load(open("CMIP5_HIST_GISS-E2-H-CC_1976_2005_mean.pck", "rb"), encoding = "latin1")
#GISS_E2_R = pickle.load(open("CMIP5_HIST_GISS-E2-R_1976_2005_mean.pck", "rb"), encoding = "latin1")
#GISS_E2_R_CC = pickle.load(open("CMIP5_HIST_GISS-E2-R-CC_1976_2005_mean.pck", "rb"), encoding = "latin1")
HadCM3 = pickle.load(open("CMIP5_HIST_HadCM3_1976_2005_mean.pck", "rb"), encoding = "latin1")
#HadGEM2_AO = pickle.load(open("CMIP5_HIST_HadGEM2-AO_1976_2005_mean.pck", "rb"), encoding = "latin1")
HadGEM2_CC = pickle.load(open("CMIP5_HIST_HadGEM2-CC_1976_2005_mean.pck", "rb"), encoding = "latin1")
HadGEM2_ES = pickle.load(open("CMIP5_HIST_HadGEM2-ES_1976_2005_mean.pck", "rb"), encoding = "latin1")
#inmcm4 = pickle.load(open("CMIP5_HIST_inmcm4_1976_2005_mean.pck", "rb"), encoding = "latin1")
#IPSL_CM5A_LR = pickle.load(open("CMIP5_HIST_IPSL-CM5A-LR_1976_2005_mean.pck", "rb"), encoding = "latin1")
#IPSL_CM5A_MR = pickle.load(open("CMIP5_HIST_IPSL-CM5A-MR_1976_2005_mean.pck", "rb"), encoding = "latin1")
#IPSL_CM5B_LR = pickle.load(open("CMIP5_HIST_IPSL-CM5B-LR_1976_2005_mean.pck", "rb"), encoding = "latin1")
#MRI_CGCM3 = pickle.load(open("CMIP5_HIST_MRI-CGCM3_1976_2005_mean.pck", "rb"), encoding = "latin1")
#MRI_ESM1 = pickle.load(open("CMIP5_HIST_MRI-ESM1_1976_2005_mean.pck", "rb"), encoding = "latin1")
#NorESM1_M = pickle.load(open("CMIP5_HIST_NorESM1-M_1976_2005_mean.pck", "rb"), encoding = "latin1")
#NorESM1_ME = pickle.load(open("CMIP5_HIST_NorESM1-ME_1976_2005_mean.pck", "rb"), encoding = "latin1")
#ACCESS1_3 = pickle.load(open("CMIP5_HIST_ACCESS1.3_1976_2005_mean.pck", "rb"), encoding = "latin1")

#NAMES = [ACCESS1, bcc_csm1_1, bcc_csm1_1_m, CanESM2, CESM1_BGC, CESM1_CAM5,
#         CESM1_FASTCHEM, CESM1_WACCM, CMCC_CESM, CMCC_CM, CMCC_CMS, CNRM_CM5,
#         CNRM_CM5_2 , CSIRO_Mk3_6_0, EC_EARTH, GFDL_CM2p1, GFDL_ESM2G,
#         GFDL_ESM2M, GISS_E2_H, GISS_E2_H_CC, GISS_E2_R, GISS_E2_R_CC, HadCM3,
#         HadGEM2_AO, HadGEM2_CC, HadGEM2_ES, inmcm4, IPSL_CM5A_LR,
#         IPSL_CM5A_MR, IPSL_CM5B_LR, MRI_CGCM3, MRI_ESM1, NorESM1_M,
#         NorESM1_ME, ACCESS1_3]

###############################################################################

# Repeating analysis methods from ARGO data for selected models

# List of selected models so not every model is considered each run
NAMES = [GFDL_ESM2G, HadCM3, HadGEM2_CC, HadGEM2_ES]
NAMES_STR = ["GFDL_ESM2G", "HadCM3", "HadGEM2_CC", "HadGEM2_ES"]

COUNT = 0
# For loop iterating through each model to perform relevant analysis
for i in range(len(NAMES)):
    
    # Extracting data in 3D
    DEPTH = NAMES[i]["DEPTH"]
    LAT = NAMES[i]["LAT"]
    PDENS0 = NAMES[i]["PDENS0"]
    PT = NAMES[i]["PT"]
    SO = NAMES[i]["SO"]
    
    # Averaging longitude (axis 2) for depth/lat space
    DEPTH = np.nanmean(DEPTH, axis=(2))
    LAT = np.nanmean(LAT, axis=(2))
    PDENS0 = np.nanmean(PDENS0, axis=(2))
    PT = np.nanmean(PT, axis=(2)) 
    SO = np.nanmean(SO, axis=(2))
    
    # Finding argument of nearest value to domain limits in Argo data
    LAT_LIMIT = (np.abs(LAT - (-64))).argmin()
    DEPTH_LIMIT = (np.abs(DEPTH.T - (1990))).argmin()
    
    # Slicing to only analyse same domain as Argo data using nearest arguments
    DEPTH = DEPTH[:DEPTH_LIMIT, LAT_LIMIT:]
    LAT = LAT[:DEPTH_LIMIT, LAT_LIMIT:]
    PDENS0 = PDENS0[:DEPTH_LIMIT, LAT_LIMIT:]
    PT = PT[:DEPTH_LIMIT, LAT_LIMIT:]
    SO = SO[:DEPTH_LIMIT, LAT_LIMIT:]
    
    # Reshaping latitude and depth values into columns
    LATS = LAT[0, :].T
    DEPTHS = DEPTH[:, 0]
    
    #Scaling features for standardised clustering (must account for NaNs, need cols)
    PDENS0S = PDENS0[~np.isnan(PDENS0)]
    PTS = PT[~np.isnan(PT)]
    SOS = SO[~np.isnan(SO)]
    
    # In same order as Argo to use same package for clustering
    FEATURES = np.column_stack((PTS, SOS, PDENS0S))

    ###########################################################################
    
    # Plotting initial model temperature and salinity with potential density
    
    # Finding nearest integer to minimum potential density provided
    PD_MIN = int(np.amin(np.reshape(PDENS0, ((DEPTH.shape[0]*LAT.shape[1]), ))))
    # Adjusting to encompass all potential density values
    if PD_MIN > np.amin(np.reshape(PDENS0, ((DEPTH.shape[0]*LAT.shape[1]), ))):
        PD_MIN = PD_MIN-1
    # Findinging nearest integer to maximum potential density provided
    PD_MAX = int(np.amax(np.reshape(PDENS0, ((DEPTH.shape[0]*LAT.shape[1]), ))))
    # Adjusting to encompass asll potential density values
    if PD_MAX < np.amax(np.reshape(PDENS0, ((DEPTH.shape[0]*LAT.shape[1]), ))):
        PD_MAX = PD_MAX+1
    # Defining potential density step size between levels
    STEP = 0.45
    # Calculating levels to plot potential density surfaces at
    LEVELS = np.arange(PD_MIN, PD_MAX + STEP, STEP)
    
    plt.figure()
    #plt.subplot(1, 2, 1)
    plt.contourf(LAT, DEPTH, PT)
    CBAR = plt.colorbar()
    CBAR.set_label("Potential Temperature")
    plt.contour(LAT, DEPTH, PT)
    PDEN = plt.contour(LATS, DEPTHS, PDENS0, levels=LEVELS,
                   colors="w", linewidths=0.7)
    
    # Formatting left subplot
    plt.gca().invert_yaxis()
#    plt.title("POTENTIAL TEMPERATURE")
    plt.xlabel("Latitude/ °N")
    plt.ylabel("Pressure/ dbar")
    plt.clabel(PDEN, inline=True, fmt = ("%0.2f" + " σ$_{0}$"),
           fontsize=9)
    
    plt.figure()
    #plt.subplot(1, 2, 2)
    plt.contourf(LAT, DEPTH, SO)
    CBAR = plt.colorbar()
    CBAR.set_label("Salinity")
    plt.contour(LAT, DEPTH, SO)
    PDEN = plt.contour(LATS, DEPTHS, PDENS0, levels=LEVELS,
                   colors="w", linewidths=0.7)
    
    # Formatting left subplot
    plt.gca().invert_yaxis()
#    plt.title("SALINTIY")
    plt.xlabel("Latitude/ °N")
    plt.ylabel("Pressure/ dbar")
    plt.clabel(PDEN, inline=True, fmt = ("%0.2f" + " σ$_{0}$"),
           fontsize=9)
    
    # Adding model name as main title to distinguish plots easily
#    plt.suptitle(NAMES_STR[i], fontweight = "semibold")
#    plt.subplots_adjust(wspace = 0.4)
    
    ###########################################################################
    """
    # Applying MeanShift clustering using package definition
    MS_plot(FEATURES, "Temperature/ Salinity/ Density", " ", False, FEATURES, LAT, DEPTH,
        LATS, DEPTHS, "CMIP5")
    """
    # Applying GaussianMixture clustering using package definition
    GMM_plot(FEATURES, False, 2, 3, 5, 10, 1, LAT, LATS, DEPTHS)
    GMM_WEIGHTS = GMM_plot.WEIGHTS
    GMM_MEANS = GMM_plot.MEANS
    GMM_COVARIANCES = GMM_plot.COVARIANCES
    
    # Applying BayesianGaussianMixture clustering using package definition
    BGMM_plot(FEATURES, False, 2, 3, 5, 10, 1, LAT, LATS, DEPTHS)
    BGMM_WEIGHTS = BGMM_plot.WEIGHTS
    BGMM_MEANS = BGMM_plot.MEANS
    BGMM_COVARIANCES = BGMM_plot.COVARIANCES
    
    ##########################################################################
        
    # Posterior Probabilities for GMM clustering with different num comps
    GMM_5_COMPONENTS = GMM_plot.POSTERIOR_PROBABILITIES["5"]
    GMM_5_COMPONENTS_MAXS = np.amax(GMM_5_COMPONENTS, axis=1)
    GMM_6_COMPONENTS = GMM_plot.POSTERIOR_PROBABILITIES["6"]
    GMM_6_COMPONENTS_MAXS = np.amax(GMM_6_COMPONENTS, axis=1)
    GMM_7_COMPONENTS = GMM_plot.POSTERIOR_PROBABILITIES["7"]
    GMM_7_COMPONENTS_MAXS = np.amax(GMM_7_COMPONENTS, axis=1)
    GMM_8_COMPONENTS = GMM_plot.POSTERIOR_PROBABILITIES["8"]
    GMM_8_COMPONENTS_MAXS = np.amax(GMM_8_COMPONENTS, axis=1)
    GMM_9_COMPONENTS = GMM_plot.POSTERIOR_PROBABILITIES["9"]
    GMM_9_COMPONENTS_MAXS = np.amax(GMM_9_COMPONENTS, axis=1)
    GMM_10_COMPONENTS = GMM_plot.POSTERIOR_PROBABILITIES["10"]
    GMM_10_COMPONENTS_MAXS = np.amax(GMM_10_COMPONENTS, axis=1)
    # 
    GMM_COMPONENTS_MAXS = [GMM_5_COMPONENTS_MAXS, GMM_6_COMPONENTS_MAXS, GMM_7_COMPONENTS_MAXS, GMM_8_COMPONENTS_MAXS, GMM_9_COMPONENTS_MAXS, GMM_10_COMPONENTS_MAXS]
    
    # Plotting boxplot for GMM clustering at each cluster number
    plt.figure()
    # Adding red line to show where 1 (max) is
    plt.plot(np.linspace(4, 11, 5), np.ones(5), "c--", linewidth=0.75)
    plt.boxplot(GMM_COMPONENTS_MAXS, showfliers=False, positions=np.arange(5, 11, 1))
    # Formatting
#    TITLE = "Posterior Probability Distributions for GMM Clustering (" + NAMES_STR[i] + ")"
#    plt.title(TITLE)
    plt.xlabel("Number of Classes")
    plt.ylabel("Posterior Probability Distribution")
    
    # Posterior Probabilities for BGMM clustering with different num comps
    BGMM_5_COMPONENTS = BGMM_plot.POSTERIOR_PROBABILITIES["5"]
    BGMM_5_COMPONENTS_MAXS = np.amax(BGMM_5_COMPONENTS, axis=1)
    BGMM_6_COMPONENTS = BGMM_plot.POSTERIOR_PROBABILITIES["6"]
    BGMM_6_COMPONENTS_MAXS = np.amax(BGMM_6_COMPONENTS, axis=1)
    BGMM_7_COMPONENTS = BGMM_plot.POSTERIOR_PROBABILITIES["7"]
    BGMM_7_COMPONENTS_MAXS = np.amax(BGMM_7_COMPONENTS, axis=1)
    BGMM_8_COMPONENTS = BGMM_plot.POSTERIOR_PROBABILITIES["8"]
    BGMM_8_COMPONENTS_MAXS = np.amax(BGMM_8_COMPONENTS, axis=1)
    BGMM_9_COMPONENTS = BGMM_plot.POSTERIOR_PROBABILITIES["9"]
    BGMM_9_COMPONENTS_MAXS = np.amax(BGMM_9_COMPONENTS, axis=1)
    BGMM_10_COMPONENTS = BGMM_plot.POSTERIOR_PROBABILITIES["10"]
    BGMM_10_COMPONENTS_MAXS = np.amax(BGMM_10_COMPONENTS, axis=1)
    # 
    BGMM_COMPONENTS_MAXS = [BGMM_5_COMPONENTS_MAXS, BGMM_6_COMPONENTS_MAXS, BGMM_7_COMPONENTS_MAXS, BGMM_8_COMPONENTS_MAXS, BGMM_9_COMPONENTS_MAXS, BGMM_10_COMPONENTS_MAXS]
    
    # Plotting boxplot for BGMM clustering at each cluster number
    plt.figure()
    # Adding red line to show where 1 (max) is
    plt.plot(np.linspace(4, 11, 5), np.ones(5), "c--", linewidth=0.75)
    plt.boxplot(BGMM_COMPONENTS_MAXS, showfliers=False, positions=np.arange(5, 11, 1))
    # Formatting
#    TITLE = "Posterior Probability Distributions for BGMM Clustering (" + NAMES_STR[i] + ")"
#    plt.title(TITLE)
    plt.xlabel("Maximum Number of Classes")
    plt.ylabel("Posterior Probability Distribution")
        
    COUNT += 1

###############################################################################

# Output runtime
print("\nruntime = ", time.clock() - START_TIME, " s\n")
