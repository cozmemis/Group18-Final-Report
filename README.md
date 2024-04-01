# Patient Length of Stay Prediction with Initial Information (Group 18)

Authors: Cagri Ozmemis, Hardik Dhaval Patel, Varun Ramakrishnan

## Introduction

### Background

The ability to forecast patient length of stay with initial information (PLOSWI) upon patient intake allows hospitals to make informed operational decisions and provide optimal patient care,
ultimately increasing resource allocation and decreasing costs.
In this project, we focus on developing a machine learning solution tailored to address the challenge of predicting patient length of stay based on data collected at the time of admission using both supervised and unsupervised learning methods.

### Dataset

Our dataset will be [real-world patient records acquired and made public by New York state hospitals in 2021](https://data.world/johnsnowlabs/hospital-inpatient-treatment-discharges-2021).
The dataset includes over one million data points, with 32 features.
Some of the notable features are basic patient information (age, gender, ethnicity) as well as diagnosis information (a code and description).


### Literature Review

Most LOS prediction literature focus on a specific illness and rely on advanced medical information by assuming an ongoing stay [[1]](#1) [[2]](#2) [[3]](#3). Additionally, there exists a limited literature that is parallel to ours. For example, Turgeman et al. [[4]](#4) created a "Cubist" rule-based regression model to predict PLOWSI using data from Veterans Health Administration in Pittsburgh, PA. Turgeman et al. used tree-based regression models with the intention of understanding the factors governing length of stay. They also describe prior research to predict length of stay including a C5.0 tree, Naive-Bayes model, K-nearest neighbours, a Multi-Layer Neural Network, and support-vector regression.

Barnes et al. [[5]](#5) worked on a slightly different approach of predicting length of stay in real-time, constantly updating the predictions at specified time intervals.
Our project aims to make inferences at the time of intake, which we believe would be more feasible to integrate into healthcare systems.

## Problem Definition

Our problem centers on predicting patient length of stay with initial information (PLOSWI) in a hospital setting.
Our primary aim is to assist managerial decision-making processes within a single hospital or any healthcare facility providing inpatient care.
Specifically, we seek to forecast the duration of a patient's stay upon admission, relying solely on data collected on the admission day, such as demographics, diagnosis, severity of illness, treatment procedure decision, and risk of mortality.
Our objective is to predict the length of stay for newly admitted patients.
This predictive capability is vital for the hospital's capacity planning and resource allocation, ultimately leading to improved utilization and reduced costs.

![inpatient image](https://intermountainhealthcare.org/-/media/images/images-sc9/medical-specialties/behavioral-health/hospital-patient-16x9.ashx?mw=500)

<span style="font-size:small; color:grey; font-style:italic;">Source:[Intermountain Healthcare](https://intermountainhealthcare.org/medical-specialties/behavioral-health/)</span>

## Data Preprocessing
We start data preprocessing by discarding the unneccessary features and keeping only the useful ones. Among 32 features, we find 19 to be suitable, which are 'Age_Group', 'Gender', 'Race', 'Ethnicity', 'Type_Of_Admission', 'Diagnosis_Description', 'Procedure_Description', 'APR_DRG_Description', 'APR_MDC_Description', 'APR_Severity_Of_Illness_Description', 'APR_Risk_Of_Mortality', 'APR_Medical_Surgical_Description', 'Payment_Typology_1', 'Is_Emergency_Department_Indicator', 'Length_Of_Stay'. Here, 'Length_Of_Stay' stay denotes the target value that we are trying to predict with ML algorithms. Afterwards, we clean the data from missing information. As our dataset contains a vast amount of information, totaling 1,046,218 entries, we opted to eliminate any data points that possessed at least one missing feature. This process resulted in a remaining dataset of 764,953 data points.

Later, we plotted the frequency of each unique entry for each data feature separately to identify any abnormal behavior. Initially, we observed that the 'Gender' feature had rarely occurring entries marked as 'U' for unknown. These unknown gender entries were discarded. Similarly, entries with 'Not Available' for the 'Type_Of_Admission' feature were also discarded.

Subsequently, in the 'Race' feature, we noticed that although there were only a few entries labeled as 'Multi-racial', their average length of stay (LOS) values were remarkably close to those labeled as 'Other race'. As these two unique values are semantically similar, we merged them to enhance the data's homogeneity. Additionally, we eliminated the 'Ethnicity' feature due to its low entropy and redundant information compared to the 'Race' feature at a different level.

Furthermore, upon examining the 'Payment_Typology_1' column, we found that values such as 'Miscellaneous/Other', 'Federal/State/Local/VA', and 'Department of Corrections' carried negligible contextual information and were infrequent. Hence, entries with these values were deducted.

After these data cleaning steps, the number of data points was reduced to 749,815. For further details and visualization, we direct the reader to the 'Data_Cleaning.ipynb' file, which contains all frequency plots. To conserve space in our report, we provide only two examples of frequency plots before and after the aforementioned processing steps. Below are the before and after frequency plots for the features.

<img src="./images/frequency_race_before.png" width="310" height="180"> <img src="./images/frequency_race_after.png" width="310" height="180">

<img src="./images/frequency_payment_before.png" width="310" height="180"> <img src="./images/frequency_payment_after.png" width="310" height="180">

Since our data consists entirely of categorical variables, we employ frequency-based methods for outlier detection. In this process, we specifically target the 'Diagnosis_Description', 'Procedure_Description', and 'APR_DRG_Description' features due to their notably high number of unique values (467, 319, and 332, respectively). To identify outliers, we employ the percentile technique, considering values as outliers if their frequency falls below the 25th percentile. Consequently, we detect values with frequencies below this threshold and exclude them from our dataset. Following this procedure, the number of entries is reduced to 728,589.

Our target value, Length of Stay (LOS), encompasses numerical data ranging from 1 day to 119 days. However, considering the possibility of utilizing a classification method as one of our supervised learning approaches, we opt to introduce another target value: a discretized version of the numerical LOS feature. To achieve this, we discretize the LOS such that each category contains a similar frequency distribution. Through experimentation, we observe that frequencies become notably consistent when we categorize LOS values of 1 and 2 days as category 1, those between 3 and 6 days as category 2, and durations exceeding one week as category 3. The graph illustrating the frequency distribution of each category is provided below.

<img src="./images/categorical_LOS.png" width="375" height="325">

We previously mentioned that we ended up with 728,589 data points in the final version of our dataset after preprocessing steps. However, this quantity still poses significant challenges in terms of speed and memory usage during algorithm development. To address this, we create three random samples representing 5%, 10%, and 15% of our data while maintaining the distribution of values in the target feature 'Length_Of_Stay'. For instance, in the 5% sample, we extract 5% of each unique value in the 'Length_Of_Stay' column. This ensures that the distribution of unique 'Length_Of_Stay' values in the original dataset is preserved.

For further details on our preprocessing efforts and frequency plots, we direct the reader to the 'Data_Cleaning.ipynb' file.


## Methods

### Unsupervised Learning

For the unsupervised learning algorithm, we decided to start with K-means method, and proceed to more complex algorithms if K-means alone cannot account for the sufficient explanation of the data dynamics. To be on the safe side, along with SKLearn's built-in function, we developed our own K-means algorithm, and analyze the results produced by both. 

Moreover, we tune parameters and configurations for our algorithms. Namely, we try different numbers of clusters, initilazition methods, and feature selections. Below, we provide how we proceed with each of those.

* Number of clusters: We try eight different values for number of clusters between 3 and 10.
* Intialization method: We employ both random and K-means++ methods to initialize the cluster centroids.
* Feature selection: Due to the categorical nature and abundance of unique values in the 'Diagnosis_Description', 'Procedure_Description', and 'APR_DRG_Description' features, using them could lead to issues of over-complexity. Hence, we opt to discard them initially. However, we remain open to reconsidering their inclusion if our algorithm fails to adequately explain the data without them. Additionally, since we are conducting an unsupervised experiment, we eliminate our target features. Thus, we proceed with the remaining ten features for algorithm development: 'Age_Group', 'Gender', 'Race', 'Type_Of_Admission', 'Payment_Typology_1', 'Is_Emergency_Department_Indicator', 'APR_MDC_Description', 'APR_Severity_Of_Illness_Description', 'APR_Risk_Of_Mortality', 'APR_Medical_Surgical_Description'. For these features, we find it suitable to apply Forward Feature Selection (FFS). However, considering all possible combinations of ten features would result in 1023 feature selection configurations, consuming substantial CPU time. To mitigate this, we fix six attributes in our configurations, selecting the simplest ones with the least medical information. These fixed features are: 'Age_Group', 'Gender', 'Race', 'Type_Of_Admission', 'Payment_Typology_1', and 'Is_Emergency_Department_Indicator'. Furthermore, we apply FFS to explore different combinations of the four advanced medical features: 'APR_MDC_Description', 'APR_Severity_Of_Illness_Description', 'APR_Risk_Of_Mortality', and 'APR_Medical_Surgical_Description'. This process results in 15 feature selection configurations.

We initially conducted experiments with varying numbers of iterations, starting with the default of 300 provided by SKlearn's built-in function. However, increasing the number of iterations did not yield different results. Therefore, to be efficient in terms of CPU time, we stick to using 300 iterations for our algorithms.

Next, we execute both our developed K-means algorithm and the built-in function a total of 240 times. This is achieved by considering 8 clusters, 2 initialization methods, and 15 feature selection configurations. We analyze the results using two metrics: the Silhouette Coefficient and the Beta-CV Measure. Since we lack true label information, we resort to internal measures rather than external ones like Precision, Recall, and F1. For the Beta-CV Measure, we develop our own algorithm, while for the Silhouette Coefficient, we utilize SKlearn's built-in function. 

For a detailed examination of our efforts in this section, we direct the reader to the 'kmeans_develop.ipynb' file.

### Supervised Learning


## Results and Discussion

### Unsupervised Learning

Initially, we note that employing different feature selection configurations does not alter the Silhouette Coefficient and Beta-CV Measure significantly. This observation indicates that our decision to utilize basic admission information, comprising 'Age_Group', 'Gender', 'Race', 'Type_Of_Admission', 'Payment_Typology_1', and 'Is_Emergency_Department_Indicator', is justified, and these features alone are adequate for effective clustering. Below, we present two tables listing the Silhouette Coefficient (in descending order) and Beta-CV Measure (in ascending order) for each algorithm configuration, initialization method, and number of clusters.

<img src="./images/sil_table.png" width="310" height="600">   <img src="./images/beta_table.png" width="310" height="600">

As indicated in the table, SKlearn's built-in function with 7 clusters and K-means++ initialization yields the highest Silhouette Coefficient, while our developed algorithm with 10 clusters and K-means++ initialization produces the best result in terms of Beta-CV Measure. Considering that achieving effective clustering with fewer clusters is more practical, we designate SKlearn's built-in function with 7 clusters and K-means++ initialization as the optimal configuration. Consequently, we utilize its outcomes to investigate clustering dynamics.

The most significant discovery from our unsupervised learning endeavor is that our algorithm effectively clusters the data points based on the target values. That is, without any explicit guidance regarding the target value, the unsupervised algorithm organizes the data points according to increasing Length of Stay (LOS) values. Below is the assignment of each cluster to a specific LOS interval:

* Cluster 0: LOS between 1 and 3
* Cluster 4: LOS between 2 and 6
* Cluster 2: LOS between 7 and 12
* Cluster 5: LOS between 13 and 22
* Cluster 1: LOS between 23 and 38
* Cluster 3: LOS between 39 and 65
* Cluster 6: LOS between 66 and 116

These findings suggest that through our unsupervised learning algorithm, utilizing only 'Age_Group', 'Gender', 'Race', 'Type_Of_Admission', 'Payment_Typology_1', and 'Is_Emergency_Department_Indicator' information, we can predict the duration of a patient's hospital stay. Furthermore, we observe distinct boundaries in terms of Length of Stay (LOS) when transitioning from one cluster to another, except for the transition from Cluster 0 to Cluster 4. In this case, there is an overlap in LOS values of 2 and 3 days. This overlap occurs because these clusters are grouped based on diagnosis descriptions and other key feature values, which may result in similar LOS outcomes.

To explain the motivation behind all clusterings concerning the feature values, we scrutinize how the clusters are structured based on the output generated by our selected algorithm configuration. We examine each feature for each cluster by extracting the following information:
* What is the most frequent categorical value?
* What is the frequency of the most frequent categorical value?
The table below provides answers to these two questions, aiding us in uncovering and comprehending the underlying patterns within the clusters.

<img src="./images/unsupervised_results_table.png" width="820" height="500"> 

During the data collection period coinciding with the COVID-19 pandemic, we observe the profound impact of the disease on the demographics of admitted patients. Clusters 1 to 6 predominantly consist of various severities of COVID-19 admissions. For instance, in Clusters 2 and 4, moderate and major cases of COVID-19 admissions prevail, where patients undergo standard COVID-19 treatments and typically stay in the hospital for 2 to 12 days. In Cluster 4, primarily representing moderate severity cases, the most frequent Length of Stay (LOS) is 4 days, whereas for major severity cases, it extends to 7 days.

Clusters 1, 3, and 5 highlight extreme COVID-19 cases in terms of severity and mortality risk. Patients in these clusters often require mechanical ventilation and exhibit LOS ranging from 13 to 65 days. Patients in Clusters 1 and 5 share similar features and belong to the All Patients Refined Diagnosis Related Groups (APR-DRG) category of SEPTICEMIA AND DISSEMINATED INFECTIONS. However, while Cluster 5 patients are typically aged 70 or older and stay for around 13 days, Cluster 1 patients are typically aged between 50 and 69 and have an LOS of around 23 days. Cluster 3 differs from Clusters 1 and 5 in terms of APR-DRG, resulting in a longer hospital stay with a most frequent LOS of 39 days.

Cluster 6 encompasses the most severe cases of COVID-19 patients requiring TRACHEOSTOMY surgery and an extended hospital stay for recovery, with patients staying for a most frequent duration of 67 days.

Lastly, Cluster 0 denotes a distinct yet prevalent admission type: childbirth. Predominantly female patients aged between 30 and 49 years are admitted for childbirth. Due to the relatively lower severity and mortality risk associated with childbirth, patients typically stay in the hospital for 1 to 3 days, with 2 days being the most frequent LOS.

Our unsupervised learning efforts demonstrate that hospital managers can obtain a preliminary estimate for the Length of Stay (LOS) and take quick actions for planning and resource allocation if necessary. To achieve this, managers must initially differentiate between childbirth and COVID-19 admissions. Subsequently, they can distinguish among various types of COVID-19 admissions based on severity of illness, risk of mortality, treatment procedure, and APR-DRG description. However, for a more accurate prediction of LOS, encompassing all types of admissions—not just childbirth and COVID-19—managers can employ our supervised learning algorithm.

We refer the reader to the 'kmeans_analysis.ipynb' to examine our efforts described in this section.

### Supervised Learning


### Next Steps
Throughout the rest of the project, we will create additional unsupervised and supervised learning algorithms and assess their effectiveness through comparison.

## References

<a id="1">[1]</a>
T.A. Daghistani, R. Elshawi, S. Sakr, A.M. Ahmed, A. Al-Thwayee, M.H. Al-Mallah,
"Predictors of in-hospital length of stay among cardiac patients: A machine learning approach,"
*International Journal of Cardiology*,
vol. 288,
August 2019,
Pages 140-147,
https://doi.org/10.1016/j.eswa.2017.02.023](https://doi.org/10.1016/j.ijcard.2019.01.046

<a id="2">[2]</a>
R. Houthooft, J. Ruyssinck, J. van der Herten, S. Stijven, I. Couckuyt, B. Gadeyne, F. Ongenae, K. Colpaert, J. Decruyenaere, T. Dhaene, F. De Turck,
"Predictive modelling of survival and length of stay in critically ill patients using sequential organ failure scores,"
*Artificial Intelligence in Medicine,
vol. 63,
March 2015,
Pages 191-207,
https://doi.org/10.1016/j.artmed.2014.12.009

<a id="3">[3]</a>
B. Alsinglawi, O. Alshari, M. Alorjani, O. Mubin, F. Alnajjar, M. Novoa, O. Darwish,
"An explainable machine learning framework for lung cancer hospital length of stay prediction,"
*Scientific Reports*,
vol. 12,
January 2022,
https://doi.org/10.1016/j.eswa.2017.02.023](https://doi.org/10.1038/s41598-021-04608-7


<a id="4">[4]</a>
L. Turgeman, J. May, and R. Sciulli,
"Insights from a machine learning model for predicting the hospital Length of Stay (LOS) at the time of admission,"
*Expert Systems with Applications*,
vol. 78,
July 2017,
Pages 376-385,
https://doi.org/10.1016/j.eswa.2017.02.023


<a id="5">[5]</a>
S. Barnes, E. Hamrock, M. Toerper, S. Siddiqui, S. Levin,
"Real-time prediction of inpatient length of stay for discharge prioritization",
*Journal of the American Medical Informatics Association*,
vol. 23,
April 2016,
Pages e2–e10,
https://doi.org/10.1093/jamia/ocv106




## Appendix

### Contribution Table

| Name               | Contribution                                                                                             |
|--------------------|----------------------------------------------------------------------------------------------------------|
| Cagri Ozmemis      | Data Preprocessing, Unsupervised Learning algorithm development                                          |
| Hardik D. Patel    | Data Preprocessing, Supervised Learning algorithm development                                            |
| Varun Ramakrishnan | Data Visualization, Supervised Learning algorithm development                                            |
| Max T. Pan         | -                                                                                                        |

### Gantt Chart

![image](https://github.com/cozmemis/Group18-Proposal/assets/156548803/237b9bae-be66-46cb-8f39-05ebbe852b35)


