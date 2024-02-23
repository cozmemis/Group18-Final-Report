# Patient Length of Stay Prediction with Initial Information (Group 18)

Authors: Cagri Ozmemis, Hardik Dhaval Patel, Varun Ramakrishnan

## Introduction

### Background

The ability to forecast patient length of stay with initial information (PLOSWI) upon patient intake allows hospitals to make informed operational decisions and provide optimal patient care,
ultimately increasing resource allocation and decreasing costs.
In this project, we focus on developing a machine learning solution tailored to address the challenge of predicting patient length of stay based on data collected at the time of admission using both supervised and unsupervised learning methods.

### Dataset

Our dataset will be [real-world patient records acquired and made public by New York state hospitals in 2021](https://data.world/johnsnowlabs/hospital-inpatient-treatment-discharges-2021).
The dataset includes over one million data points, with 19 key features.
Some of the notable features are basic patient information (age, gender, ethnicity) as well as diagnosis information (a code and description).


### Literature Review

Turgeman et al [[1]](#1). created a "Cubist" rule-based regression model to predict PLOWSI using data from Veterans Health Administration in Pittsburgh, PA.
Turgeman et al. used tree-based regression models with the intention of understanding the factors governing length of stay.
They also describe prior research to predict length of stay including a C5.0 tree, Naive-Bayes model, K-nearest neighbours, a Multi-Layer Neural Network, and support-vector regression.

Barnes et al. [[2]](#2) worked on a slightly different approach of predicting length of stay in real-time, constantly updating the predictions at specified time intervals.
Our project aims to make inferences at the time of intake, which we believe would be more feasible to integrate into healthcare systems.

## Problem Definition

Our problem centers on predicting patient length of stay with initial information (PLOSWI) in a hospital setting. Our primary aim is to assist managerial decision-making processes within a single hospital or any healthcare facility providing inpatient care. Specifically, we seek to forecast the duration of a patient's stay upon admission, relying solely on data collected on the admission day, such as demographics, diagnosis, severity of illness, treatment procedure decision, and risk of mortality. Utilizing these simple yet essential pieces of information, our objective is to predict the length of stay for newly admitted patients. This predictive capability is vital for the hospital's capacity planning and resource allocation, ultimately leading to improved utilization and reduced costs.

![inpatient image](https://intermountainhealthcare.org/-/media/images/images-sc9/medical-specialties/behavioral-health/hospital-patient-16x9.ashx?mw=500)

<span style="font-size:small; color:grey; font-style:italic;">Source:[Intermountain Healthcare](https://intermountainhealthcare.org/medical-specialties/behavioral-health/)</span>


## Method

We will start by removing irrelevant or redundant data as they can harm the accuracy. Variables such as Total_Charges are calculated after days in the hospital and therefore would not be relevant as we are trying to predict the days in the hospital. Categorical variables can be encoded using OneHotEncoder to convert them into numerical format to facilitate a more programmable and interpretable way. Several supervised learning models are commonly utilized in LOS prediction for their robustness in handling complex relationships in the data such as Support Vector Regression, Gradient Boosting Regression and Random Forest. Among these, we will select two and implement them for the prediction. Moreover, unsupervised models, including DBSCAN, GMM and K-Means, are used for patient grouping, aiding in pattern discovery. We will also select two of these algorithms based on their effectiveness in dealing with high-dimensional healthcare data and their interpretability in providing insights.

## (Potential) Results and Discussion

For evaluation, Mean Percentage Error (MPE) and K-fold cross validation can be used to evaluate prediction quality. In terms of metrics, the project aims to achieve an MPE of near 0. Also, high matching based measures (e.g., accuracy, precision, recall, F-measure) and AUC scores are expexted which demonstrates the model's advanced prediction abilities. Comparing these metrics among the models to be employed, we will be able to demonstrate the strengths and weaknesses of each model and decide on the best performing ones.

## References

<a id="1">[1]</a>
L. Turgeman, J. May, and R. Sciulli,
"Insights from a machine learning model for predicting the hospital Length of Stay (LOS) at the time of admission,"
*Expert Systems with Applications*,
vol. 78,
July 2017,
Pages 376-385,
https://doi.org/10.1016/j.eswa.2017.02.023


<a id="2">[2]</a>
S. Barnes, E. Hamrock, M. Toerper, S. Siddiqui, S. Levin,
"Real-time prediction of inpatient length of stay for discharge prioritization",
*Journal of the American Medical Informatics Association*,
vol. 23,
April 2016,
Pages e2–e10,
https://doi.org/10.1093/jamia/ocv106




## Appendix

### Contribution Table

| Name      | Contrubtion |
| ----------- | ----------- |
| Cagri Ozmemis      | finding dataset, literature review, writing Problem Definition, GitHub setup, video creation & recording       |
| Hardik D. Patel   | writing Methods, writing Results & Discussion        |
| Varun Ramakrishnan   | literature review, writing Introduction/Background,  GitHub setup, video creation & recording        |


### Gantt Chart

![image](https://github.com/cozmemis/Group18-Proposal/assets/156548803/237b9bae-be66-46cb-8f39-05ebbe852b35)


