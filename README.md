# MachineLearningSouthernOcean
Applying unsupervised clustering functions available in the Python Scikit-learn library (Meanshift, BIC, GaussianMixture, BayesianGaussianMixture) to oceanographic data to study water mass structures.
There are 4 scripts included as the culmination of a 10 week project Summer Research Placement with British Antarctic Survey (2018):
* South Atlantic.py
* South Pacific.py 
* CMIP5 load.py
* mypackage.py

It is assumed that these scripts will be saved to the same directory as the data used.

## South Atlantic.py
### Library Requirements
* Python 3.5.2
* Scikit-learn 0.18.1
* numpy 1.11.3
* scipy 0.19.0
* matplotlib 1.5.3
* itertools (part of the standard python library)
* time (part of the standard python library)
### Overview
Includes option to load and split Argo gridded interpolated product into domain sections/ seasons in the South Atlantic basin. Produces initial plots of salinity and temperature (with potential density levels), performs Meanshift clustering on increasing numbers of features included, BIC scores for 1 - 10 classes with increasing numbers of features included, GaussianMixture clustering (GMM), BayesianGaussianMixture clustering (BGMM), Silhouette Coefficient for 5 - 10 classes for GMM and BGMM approaches, Calinski-Harabasz Score for 5 - 10 classes for GMM and BGMM approach, GMM and BGMM clustering on specific sections of the domain, initial plots of sections' salinity and temperature (with potential density levels), initial plots of salinity and temperature (with potential density levels) for seasonally sliced data, GMM and BGMM clustering of austral autumn and austral winter.

## South Pacific.py
### Library Requirements
* Python 3.5.2
* Scikit-learn 0.18.1
* numpy 1.11.3
* scipy 0.19.0
* matplotlib 1.5.3
* itertools (part of the standard python library)
* time (part of the standard python library)
### Overview
Includes option to load and split Argo gridded interpolated product into domain sections/ seasons in the South Pacific basin. Produces initial plots of salinity and temperature (with potential density levels), performs Meanshift clustering on increasing numbers of features included, BIC scores for 1 - 10 classes with increasing numbers of features included, GaussianMixture clustering (GMM), BayesianGaussianMixture clustering (BGMM), Silhouette Coefficient for 5 - 10 classes for GMM and BGMM approaches, Calinski-Harabasz Score for 5 - 10 classes for GMM and BGMM approach, GMM and BGMM clustering on specific sections of the domain, initial plots of sections' salinity and temperature (with potential density levels), initial plots of salinity and temperature (with potential density levels) for seasonally sliced data, GMM and BGMM clustering of austral autumn and austral winter.

## CMIP5 load.py
### Library Requirements
* Python 3.5.2
* Scikit-learn 0.18.1
* numpy 1.11.3
* scipy 0.19.0
* matplotlib 1.5.3
* pickle (part of the standard python library)
* time (part of the standard python library)
### Overview
Loads specific models from the CMIP5 suite, iterates through these to produce initial plots of salinity and temperature (with potential density levels), Gaussian Mixture clustering for 5 - 10 classes, variational Gaussian Mixture clustering for 5 - 10 components and boxplots for the maximum posterior probabilities for these methods (5 - 10 classes).

## mypackage.py
### Library Requirements
* Python 3.5.2
* Scikit-learn 0.18.1
* numpy 1.11.3
* scipy 0.19.0
* matplotlib 1.5.3
* itertools (part of the standard python library)
### Overview
Included definition to cluster and produce plots and scoring metrics using  either MeanShift, GaussianMixture, BayesianGaussianMixture.
