"""
Created on Sat Aug 18 22:11:01 2018

@author: Lille Borresen

Aims:
    - Define MeanShift routine to cluster and plot results
    - Define BIC score routine
    - Define GaussianMixture and BayesianGaussianMixture routines
"""

# Importing modules
import time
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from sklearn import mixture
from sklearn.cluster import MeanShift
from sklearn import metrics
from itertools import cycle
from scipy import linalg



# To record runtime
START_TIME = time.clock()

###############################################################################

# Plotting MeanShift clusterings
def MS_plot(X, datatype, unit, ref, AVG, xx, yy, x, y, OCEAN):
    """
    Returns plots of MeanShift clusters\n
    ----------\n
    `X` : (array) Features, shape (n_samples, n_features)\n
    `datatype` : (str) Physical quantity being clustered\n
    `unit` : (str) Units of physical quantity\n
    `ref` : (bool) Either True or False (original plot as reference if True)\n
    `AVG` : (array) All datatype data in multidimensional array\n
    `xx` : (array) x-axis quantity in shape of AVG (e.g latitude)\n
    `yy` : (array) y-axis quantity in shape of AVG (e.g pressure)\n
    `x` : (array) xx reshaped as col\n
    `y` : (array) yy reshaped as col\n
    `OCEAN`: (str) Name of ocean basin data is taken from 
    """
    
    # Defining MeanShift function
    MS = MeanShift()
    
    # If X is 1D then it must be reshaped in order to continue with clustering
    if X.ndim==1:
        X = np.reshape(X, (-1, 1))
    
    # Defining labels relating to cluster numbers assigned to each point
    LABELS = MS.fit_predict(X)
    LABELS_UNQ = np.unique(LABELS)
    MS_plot.CLUSTNUM = len(LABELS_UNQ)
    print("\nEstimated", datatype, "clusters : %d" % MS_plot.CLUSTNUM)
    
    # Returning goodness of fit metrics when there is more than one cluster
    if sum(LABELS==0)!=len(LABELS):
        # Defining Silhouette Coefficient for use outside function
        MS_plot.SILHOUETTE = metrics.silhouette_score(X, LABELS)
        MS_plot.CALINSKI_HARABAZ = metrics.calinski_harabaz_score(X, LABELS)
    
    # Plotting clustering
    plt.figure()
    # Plotting as subplots if the refernce will be included (single feature)
    if ref == True:
        plt.subplot(1, 2, 1)

    # Defining segemented image to plot cluster extents
    MS_plot.LABELS_PLT = np.reshape(LABELS, (xx.shape))
    # Defining discrete colorbar to label each cluster
    CMAP = plt.cm.get_cmap("viridis", MS_plot.CLUSTNUM)
    CS = plt.imshow(MS_plot.LABELS_PLT, cmap=CMAP, aspect="auto",
               extent=[float(min(x.T)), float(max(x.T)),
                       float(max(y)), float(min(y))])
    
    # Formatting plot of clustering
#    TITLE = datatype + " Clusters (" + OCEAN + ")\nEstimated Clusters: %d"
#    plt.title(TITLE %MS_plot.CLUSTNUM)
    plt.xlabel("Latitude/ °N")
    plt.ylabel("Pressure/ dbar")
    # Colorbar with discrete levels
    CBAR_TITLE = datatype + " Clusters"
    CBAR = plt.colorbar(ticks = range(MS_plot.CLUSTNUM), label=CBAR_TITLE)
    plt.clim(-0.5, MS_plot.CLUSTNUM-0.5)
    CBAR_LABELS = np.arange(0, MS_plot.CLUSTNUM, 1)
    CBLOC = np.arange((- 0.5), (MS_plot.CLUSTNUM + 0.5), 1)
    
    # Including original plot as a reference (single feature)
    if ref == True:
        
        # Subplot showing original contour plot for comparison
        plt.subplot(1, 2, 2)
        
        # Replotting original salinity data
        plt.contourf(xx, yy, AVG)
        CBAR = plt.colorbar()
        CBAR_TITLE = datatype + " / " + unit
        CBAR.set_label(CBAR_TITLE)
        plt.contour(xx, yy, AVG, colors="k")
        
        # Formatting right subplot
        plt.gca().invert_yaxis()
        plt.xlabel("Latitude/ °N")
        plt.ylabel("Pressure/ dbar")
        plt.title("Original Plot as Reference")
    
    # If 2 features are used, plot clustering relative to those features
    if X.shape[1] == 2:
        
        # Plotting figure
        plt.figure()
        # Subplot showing temperature versus salinity
        plt.subplot(1, 2, 1)
        plt.plot(X[:,0], X[:,1], "r.", markersize=1)
        
        # Formatting left subplot
        TITLE = datatype + " Unclustered (South " + OCEAN + ")"
        plt.title(TITLE)
        plt.xlabel("Scaled Temperature")
        plt.ylabel("Scaled Salinity")
        
        # Subplot showing clustered temperature and salinity with cluster centers
        plt.subplot(1, 2, 2)
        
        # Plotting clusters and centers with discrete colors
        CLUSTER_CENTERS = MS.cluster_centers_
        COLORS = cycle("rbmgkcyrbmgkcyrbmgkcy")
        for i, col in zip(range(MS_plot.CLUSTNUM), COLORS):
            MEMBER = LABELS == i
            CLUSTER_CENTER = CLUSTER_CENTERS[i]
            plt.plot(AVG[MEMBER, 0], AVG[MEMBER, 1], col + ".",
                     markersize=1)
            plt.plot(CLUSTER_CENTER[0], CLUSTER_CENTER[1], "o", markerfacecolor=col,
                     markeredgecolor="k", markersize=8)
        
        # Formatting right subplot
#        TITLE = datatype + " Clustered (South " + OCEAN + ")"
#        plt.title(TITLE)
        plt.xlabel("Temperature/ °C")
        plt.ylabel("Salinity/ ")

###############################################################################

# Plotting BIC score bar chart and components in TS space if specified
def BIC_plot(X, MIN_COMP, MAX_COMP, ref, DATATYPE, OCEAN):
    """
    Returns bar chart of BIC scores and components as ellipse in TS space\n
    ----------\n
    `X` : (array) Shape (n_samples, n_features)\n
    `MIN_COMP` : (int) Minimum number of components to calculate BIC for\n
    `MAX_COMP` : (int) Maximum number of components to calculate BIC for\n
    "ref" : (bool) If true plot TS space reference of optimal clustering\n
    `DATATYPE` : (str) Name of physical quantity/s being clustered\n
    `OCEAN` : (str) Name of ocean basin being used
    """
    
    # If X is 1D then it must be reshaped in order to continue with clustering
    if len(X.shape) == 1:
        X = np.reshape(X, (-1, 1))
        
    # Blank list to append BIC scores to
    BIC = []
    # Number of components to calculate BIC scores for at each covariance
    N_COMP_RNG = range(MIN_COMP, MAX_COMP)
    COV_TYPES = ["spherical", "tied", "diag", "full"]
    # Define lowest BIC as infinity to be redefined in for loop (dummy value)
    LOW_BIC = np.infty
    for cov_type in COV_TYPES:
        for n_components in N_COMP_RNG:
            # Fit using Gaussian Mixture
            GMM = mixture.GaussianMixture(covariance_type = cov_type)
            GMM.fit(X)
            BIC.append(GMM.bic(X))
            
            # To determine covariance and number of components of best BIC score
            if BIC[-1] < LOW_BIC:
                LOW_BIC = BIC[-1]
    
    # Determining best covariance/components from best_gmm
    PARAMS = GMM.get_params()
    BEST_COV = PARAMS["covariance_type"]
    BEST_COMP = PARAMS["n_components"]
    print("\nbest covariance type for temperature/ salinity : \"",
          BEST_COV, "\"")
    print("( when components = ",
          BEST_COMP, ")")

    # Convert BIC list to array in order to plot
    BIC = np.array(BIC)
    COLORS = cycle(["mediumseagreen", "orchid",
    "crimson", "mediumslateblue"])
    BARS = []
    
    # Plotting BIC scores
    plt.figure()
    # Only need suplot if there are at least 2 features
    if ref == True:
        if X.shape[1] >= 2:
            plt.subplot(2, 1, 1)
    # Plotting bar chart to compare temperature BIC scores
    for i, (cov_type, color) in enumerate(zip(COV_TYPES, COLORS)):
        XPOS = np.array(N_COMP_RNG) + 0.2 * (i - 2)
        BARS.append(plt.bar(XPOS, BIC[i * len(N_COMP_RNG):
            (i + 1) * len(N_COMP_RNG)],
            width=0.2, color=color))
            
    # Formatting figure
    plt.ylim([- 0.02 * BIC.min(), 1.02 * BIC.max()])
#    TITLE = "BIC Score per Model for " + DATATYPE
#    plt.title(TITLE)
    plt.xlabel("Number of Components")
    plt.ylabel("BIC Score")
    plt.legend([b[0] for b in BARS], COV_TYPES)
    
    if ref == True:
        if X.shape[1] >= 2:    
            # Plotting best covariance type over clusters
            subplot = plt.subplot(2, 1, 2)
            
            PDCT = GMM.predict(X)
            for i, (mean, cov, color) in enumerate(zip(GMM.means_, GMM.covariances_,
                                                       COLORS)):
                
                # Plotting points in a colour corresponding to a specific cluster
                O, P = linalg.eigh(cov)
                if not np.any(PDCT == i):
                    continue
                plt.scatter(X[PDCT == i, 0], X[PDCT == i, 1], 0.8, color=color)
            
                # Define ellipse parameters for Gaussian component
                RADANG = np.arctan2(P[0][1], P[0][0])
                DEGANG = np.rad2deg(RADANG)
                V = 2.0 * np.sqrt(2.0) * np.sqrt(O)
                # Defining ellipse to plot to represent each cluster
                ELL = Ellipse(mean, V[0], V[1], 180.0 + DEGANG, color=color)
                ELL.set_clip_box(subplot.bbox)
                ELL.set_alpha(1/2)
                subplot.add_artist(ELL)
            
            # Formatting figure
#            TITLE = "Temperature versus Salinity with " + str(BEST_COMP) + " Components (" + str(BEST_COV) + ")"
#            plt.title(TITLE)
            plt.xlabel("Temperature / °C")
            plt.ylabel("Salinity / ")
            plt.subplots_adjust(hspace = 0.4)

###############################################################################

# Plotting GMM clusterings
def GMM_plot(X, ref, sub1, sub2, COMP_MIN, COMP_MAX, STEP, xx, x, y):
    """
    Returns plots of Gaussian Mixture Model clusters\n
    ----------\n
    `X` : (array) Shape (n_samples, n_features)\n
    `ref` : (bool) Either True or False (plot clusters in feature space if True)\n
    `sub1` : (int) First subplots arguments\n
    `sub2` : (int) Second subplots arguments\n
    `COMP_MIN` : (int) Minimum number of components to use\n
    `COMP_MAX` : (int) Maximum number of components to use\n
    `STEP` : (int) Step size for number of components between min and max\n
    `xx` : (array) x-axis quantity in shape of AVG (e.g latitude)\n
    `x` : (array) xx reshaped as col\n
    `y` : (array) yy reshaped as col\n
    """
    
    # Defining dictionaries to append clustering attributes to
    GMM_plot.WEIGHTS = {}
    GMM_plot.MEANS = {}
    GMM_plot.COVARIANCES = {}
    GMM_plot.POSTERIOR_PROBABILITIES = {}
    # Creating headers (strings) for each dictionary item
    COMP_NUMS = [str(a) for a in np.arange(COMP_MIN, COMP_MAX+1, 1)]
    
    # Defining one colourmap to plot all clusters to
    CMAP = plt.cm.get_cmap("viridis", COMP_MAX)
    # Blank array to append cluster labels to
    CLUSTS = np.empty(((xx.shape[0]*xx.shape[1]), ))
    
    # Plotting clustering onto latitude/ pressure space
    #FIG1 = plt.figure()
    #GRID = plt.GridSpec(sub1, sub2, hspace=0.4, wspace=0.4)
    
    # Defining indicies to iterate through when assigning subplots
    indx1 = np.repeat(np.arange(0, sub1, 1), sub2)
    indx2 = np.tile(np.arange(0, sub2, 1), sub1)
    
    # Defining empty arrays to assign goodness of fit scores to
    GMM_plot.SILHOUETTE_SCORES = np.empty(((COMP_MAX-COMP_MIN+1),))
    GMM_plot.CALINSKI_HARABAZ_SCORES = np.empty(((COMP_MAX-COMP_MIN+1),))
    
    # For loop to plot GaussianMixture clusters
    count1 = 0
    for i in np.arange(COMP_MIN, COMP_MAX+1, STEP):

        # Fit a Gaussian mixture model
        GMM = mixture.GaussianMixture(n_components=i, covariance_type="full").fit(X)
        
        # Assigning attribiutes to dictionaries
        GMM_plot.WEIGHTS[COMP_NUMS[count1]] = GMM.weights_
        GMM_plot.MEANS[COMP_NUMS[count1]] = GMM.means_
        GMM_plot.COVARIANCES[COMP_NUMS[count1]] = GMM.covariances_
        GMM_plot.POSTERIOR_PROBABILITIES[COMP_NUMS[count1]] = GMM.predict_proba(X)
        
        # New subplot for each number of components used
        plt.figure()
        #plt.subplot(GRID[indx1[count1], indx2[count1]])
        # Defining segemented image to plot cluster extents
        CLUST_IMG = np.reshape(GMM.predict(X), (xx.shape))
        # Defining discrete colorbar to label each cluster
        plt.imshow(CLUST_IMG, cmap=CMAP, aspect="auto",
                   extent=[float(min(x.T)), float(max(x.T)),
                           float(max(y)), float(min(y))])
        # Formatting each subplot
        #TITLE = str(i) + " Components"
        #plt.title(TITLE)
        plt.xlabel("Latitude/ °N")
        plt.ylabel("Pressure/ dbar")
        
        # Defining labels in order to calculate goodness of fit metrics
        LABELS = np.reshape(CLUST_IMG, (len(X[:, 0]), ))
        
        # Recording goodness of fit scores
        GMM_plot.SILHOUETTE_SCORES[count1] = metrics.silhouette_score(X, LABELS)
        GMM_plot.CALINSKI_HARABAZ_SCORES[count1] = metrics.calinski_harabaz_score(X, LABELS)
        
        # Adding 1 to count in order to loop through subplots
        count1 +=1
    
    # Adding single discrete colorbar to the figure
#    plt.colorbar(ticks = range(COMP_MAX), label = "Temperature, Salinity, Density Clusters", 
#                 cax = FIG1.add_axes([0.93, 0.1, 0.015, 0.8]))
#    plt.clim(-0.5, COMP_MAX-0.5)
    
    # If ref is True then the routine will plot clusters in feature-space
    if ref == True:
        
        # When there are not 3 features, plot in temperature/ salinity space
        if X.shape[1]!=3:
            
            count2 = 1
            FIG2 = plt.figure()
            
            for i in np.arange(COMP_MIN, COMP_MAX+1, STEP):

                # Fit a Gaussian mixture model
                GMM = mixture.GaussianMixture(n_components=i, covariance_type="full").fit(X)
                
                # Defining colors to cycle through when plotting clusters
                COLORS = cycle(["navy", "c", "gray", "palevioletred", "yellowgreen", "r",
                        "orange", "b", "mediumseagreen", "k", "peru", "y"])
                
                # Plotting each cluster in temperature/ salintity space
                for j, (mean, covar, color) in enumerate(zip(GMM.means_,
                       GMM.covariances_, COLORS)):
            
                    SPLOT = plt.subplot(sub1, sub2, count2, aspect = "auto")
                    
                    # Eigen values and eigen vectors of covariances
                    V, W = linalg.eigh(covar)
                    V = 2.0 * np.sqrt(2.0) * np.sqrt(V)
                    U = W[0] / linalg.norm(W[0])
                
                    if not np.any(GMM.predict(X) == j):
                        continue
                    plt.scatter(X[GMM.predict(X) == j, 0],
                        X[GMM.predict(X) == j, 1], 0.8, color=color)
          
                    # Plot an ellipse to show the Gaussian component
                    ANGLE = np.arctan(U[1] / U[0])
                    ANGLE = np.rad2deg(ANGLE)
                    ELL = Ellipse(mean, V[0], V[1], 180.0 + ANGLE, color=color)
                    ELL.set_clip_box(SPLOT.bbox)
                    ELL.set_alpha(0.5)
                    SPLOT.add_artist(ELL)
                    
                    # Labelling subplot
                    TITLE = str(i) + " Components"
                    plt.title(TITLE)
                    plt.xlabel("Temperature/ °C")
                    plt.ylabel("Salinity/ ")
                    
                # Adding 1 to count in order to loop through subplots
                count2 += 1
            
            # If 3 features given , can plot ellipsoids in 3D feature-space
            if X.shape[1] == 3:
                
                count2 = 1
                # Plotting clustering into temperature/ salinity space
                FIG2 = plt.figure()
                
                for i in np.arange(COMP_MIN, COMP_MAX+1, STEP):
    
                    # Fit a Gaussian mixture model
                    GMM = mixture.GaussianMixture(n_components=i, covariance_type="full").fit(X)
                    
                    # Defining colors to cycle through when plotting clusters
                    COLORS = cycle(["navy", "c", "gray", "palevioletred", "yellowgreen", "r",
                            "orange", "b", "mediumseagreen", "k", "peru", "y"])
                    
                    # Plotting each cluster in temperature/ salintity space
                    for j, (mean, covar, color) in enumerate(zip(GMM.means_,
                           GMM.covariances_, COLORS)):
                
                        SPLOT = plt.subplot(sub1, sub2, count2,
                                            aspect="auto", projection="3d")
                        
                        # Eigen values and eigen vectors of covariances
                        V, W = linalg.eigh(covar)
                        V = 2.0 * np.sqrt(2.0) * np.sqrt(V)
                        U = W[0] / linalg.norm(W[0])
                    
                        if not np.any(GMM.predict(X) == j):
                            continue
                        plt.scatter(X[GMM.predict(X) == j, 0],
                            X[GMM.predict(X) == j, 1], X[GMM.predict(X) == j, 2],
                            0.8, color=color)
              
#                        # Plot an ellipse to show the Gaussian component
#                        ANGLE = np.arctan(U[1] / U[0])
#                        ANGLE = np.rad2deg(ANGLE)
#                        ELL = Ellipse(mean, V[0], V[1], 180.0 + ANGLE, color=color)
#                        ELL.set_clip_box(SPLOT.bbox)
#                        ELL.set_alpha(0.5)
#                        SPLOT.add_artist(ELL)
                        
                        # Labelling subplot
                        TITLE = str(i) + " Components"
                        plt.title(TITLE)
                        plt.xlabel("Temperature/ °C")
                        plt.ylabel("Salinity/ ")
                        
                    # Adding 1 to count in order to loop through subplots
                    count2 += 1

###############################################################################

# Plotting BGMM clusterings
def BGMM_plot(X, ref, sub1, sub2, COMP_MIN, COMP_MAX, STEP, xx, x, y):
    """
    Returns plots of Gaussian Mixture Model clusters\n
    ----------\n
    `X` : (array) Shape (n_samples, n_features)\n
    `ref` : (bool) Either True or False (plot clusters in feature space if True)\n
    `sub1` : (int) First subplots arguments\n
    `sub2` : (int) Second subplots arguments\n
    `COMP_MIN` : (int) Minimum number of components to use\n
    `COMP_MAX` : (int) Maximum number of components to use\n
    `STEP` : (int) Step size for number of components between min and max\n
    `xx` : (array) x-axis quantity in shape of AVG (e.g latitude)\n
    `x` : (array) xx reshaped as col\n
    `y` : (array) yy reshaped as col\n
    """
    
    # Defining dictionaries to append clustering attributes to
    BGMM_plot.WEIGHTS = {}
    BGMM_plot.MEANS = {}
    BGMM_plot.COVARIANCES = {}
    BGMM_plot.POSTERIOR_PROBABILITIES = {}
    # Creating headers (strings) for each dictionary item
    COMP_NUMS = [str(a) for a in np.arange(COMP_MIN, COMP_MAX+1, 1)]
    
    # Defining one colourmap to plot all clusters to
    CMAP = plt.cm.get_cmap("viridis", COMP_MAX)
    # Blank array to append cluster labels to
    CLUSTS = np.empty(((xx.shape[0]*xx.shape[1]), ))
    
    # Plotting clustering onto latitude/ pressure space
    #FIG1 = plt.figure()
    #GRID = plt.GridSpec(sub1, sub2, hspace=0.4, wspace=0.4)
    
    # Defining indicies to iterate through when assigning subplots
    indx1 = np.repeat(np.arange(0, sub1, 1), sub2)
    indx2 = np.tile(np.arange(0, sub2, 1), sub1)
    
    # Defining empty arrays to assign goodness of fit scores to
    BGMM_plot.SILHOUETTE_SCORES = np.empty(((COMP_MAX-COMP_MIN+1),))
    BGMM_plot.CALINSKI_HARABAZ_SCORES = np.empty(((COMP_MAX-COMP_MIN+1),))
    
    # For loop to plot BayesianGaussianMixture clusters
    count1 = 0
    for i in np.arange(COMP_MIN, COMP_MAX+1, STEP):

        # Fit a variational Gaussian mixture model
        BGMM = mixture.BayesianGaussianMixture(n_components=i, covariance_type="full").fit(X)
        
        # Assigning attribiutes to dictionaries
        BGMM_plot.WEIGHTS[COMP_NUMS[count1]] = BGMM.weights_
        BGMM_plot.MEANS[COMP_NUMS[count1]] = BGMM.means_
        BGMM_plot.COVARIANCES[COMP_NUMS[count1]] = BGMM.covariances_
        BGMM_plot.POSTERIOR_PROBABILITIES[COMP_NUMS[count1]] = BGMM.predict_proba(X)
        
        # New subplot for each number of components used
        plt.figure()
        #plt.subplot(GRID[indx1[count1], indx2[count1]])
        # Defining segemented image to plot cluster extents
        CLUST_IMG = np.reshape(BGMM.predict(X), (xx.shape))
        # Defining discrete colorbar to label each cluster
        plt.imshow(CLUST_IMG, cmap=CMAP, aspect="auto",
                   extent=[float(min(x.T)), float(max(x.T)),
                           float(max(y)), float(min(y))])
        # Formatting each subplot
        #TITLE = str(i) + " Components"
        #plt.title(TITLE)
        plt.xlabel("Latitude/ °N")
        plt.ylabel("Pressure/ dbar")
        
        # Defining labels in order to calculate goodness of fit metrics
        LABELS = np.reshape(CLUST_IMG, (len(X[:, 0]), ))
        
        # Recording goodness of fit scores
        BGMM_plot.SILHOUETTE_SCORES[count1] = metrics.silhouette_score(X, LABELS)
        BGMM_plot.CALINSKI_HARABAZ_SCORES[count1] = metrics.calinski_harabaz_score(X, LABELS)
        
        # Adding 1 to count in order to loop through subplots
        count1 +=1
    
    # Adding single discrete colorbar to the figure
#    plt.colorbar(ticks = range(COMP_MAX), label = "Temperature, Salinity, Density Clusters", 
#                 cax = FIG1.add_axes([0.93, 0.1, 0.015, 0.8]))
#    plt.clim(-0.5, COMP_MAX-0.5)
    
    # If ref is True then the routine will plot clusters in feature-space
    if ref == True:
        
        # When there are not 3 features, plot in temperature/ salinity space
        if X.shape[1]!=3:
            
            count2 = 1
            FIG2 = plt.figure()
            
            for i in np.arange(COMP_MIN, COMP_MAX+1, STEP):

                # Fit a variational Gaussian mixture model
                BGMM = mixture.BayesianGaussianMixture(n_components=i, covariance_type="full").fit(X)
                
                # Defining colors to cycle through when plotting clusters
                COLORS = cycle(["navy", "c", "gray", "palevioletred", "yellowgreen", "r",
                        "orange", "b", "mediumseagreen", "k", "peru", "y"])
                
                # Plotting each cluster in temperature/ salintity space
                for j, (mean, covar, color) in enumerate(zip(BGMM.means_,
                       BGMM.covariances_, COLORS)):
            
                    SPLOT = plt.subplot(sub1, sub2, count2, aspect = "auto")
                    
                    # Eigen values and eigen vectors of covariances
                    V, W = linalg.eigh(covar)
                    V = 2.0 * np.sqrt(2.0) * np.sqrt(V)
                    U = W[0] / linalg.norm(W[0])
                
                    if not np.any(BGMM.predict(X) == j):
                        continue
                    plt.scatter(X[BGMM.predict(X) == j, 0],
                        X[BGMM.predict(X) == j, 1], 0.8, color=color)
          
                    # Plot an ellipse to show the Gaussian component
                    ANGLE = np.arctan(U[1] / U[0])
                    ANGLE = np.rad2deg(ANGLE)
                    ELL = Ellipse(mean, V[0], V[1], 180.0 + ANGLE, color=color)
                    ELL.set_clip_box(SPLOT.bbox)
                    ELL.set_alpha(0.5)
                    SPLOT.add_artist(ELL)
                    
                    # Labelling subplot
                    TITLE = str(i) + " Components"
                    plt.title(TITLE)
                    plt.xlabel("Temperature/ °C")
                    plt.ylabel("Salinity/ ")
                    
                # Adding 1 to count in order to loop through subplots
                count2 += 1
            
            # If 3 features given , can plot ellipsoids in 3D feature-space
            if X.shape[1] == 3:
                
                count2 = 1
                # Plotting clustering into temperature/ salinity space
                FIG2 = plt.figure()
                
                for i in np.arange(COMP_MIN, COMP_MAX+1, STEP):
    
                    # Fit a variational Gaussian mixture model
                    BGMM = mixture.BayesianGaussianMixture(n_components=i, covariance_type="full").fit(X)
                    
                    # Defining colors to cycle through when plotting clusters
                    COLORS = cycle(["navy", "c", "gray", "palevioletred", "yellowgreen", "r",
                            "orange", "b", "mediumseagreen", "k", "peru", "y"])
                    
                    # Plotting each cluster in temperature/ salintity space
                    for j, (mean, covar, color) in enumerate(zip(BGMM.means_,
                           BGMM.covariances_, COLORS)):
                
                        SPLOT = plt.subplot(sub1, sub2, count2,
                                            aspect="auto", projection="3d")
                        
                        # Eigen values and eigen vectors of covariances
                        V, W = linalg.eigh(covar)
                        V = 2.0 * np.sqrt(2.0) * np.sqrt(V)
                        U = W[0] / linalg.norm(W[0])
                    
                        if not np.any(BGMM.predict(X) == j):
                            continue
                        plt.scatter(X[BGMM.predict(X) == j, 0],
                            X[BGMM.predict(X) == j, 1], X[BGMM.predict(X) == j, 2],
                            0.8, color=color)
              
#                        # Plot an ellipse to show the Gaussian component
#                        ANGLE = np.arctan(U[1] / U[0])
#                        ANGLE = np.rad2deg(ANGLE)
#                        ELL = Ellipse(mean, V[0], V[1], 180.0 + ANGLE, color=color)
#                        ELL.set_clip_box(SPLOT.bbox)
#                        ELL.set_alpha(0.5)
#                        SPLOT.add_artist(ELL)
                        
                        # Labelling subplot
                        TITLE = str(i) + " Components"
                        plt.title(TITLE)
                        plt.xlabel("Temperature/ °C")
                        plt.ylabel("Salinity/ ")
                        
                    # Adding 1 to count in order to loop through subplots
                    count2 += 1

###############################################################################            

# Output runtime
print("\nruntime = ", time.clock() - START_TIME, " s\n")
