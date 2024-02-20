# Patient Length of Stay Prediction with Initial Information (Group 18)

Authors: Cagri Ozmemis, Hardik Dhaval Patel, Max Teng Pan, Varun Ramakrishnan

## Introduction

### Background

In the field of healthcare management, efficient allocation of resources and accurate capacity planning are crucial factors for ensuring quality patient care.
One of the most important aspects of this problem is predicting the length of stay for patients admitted to hospitals.
The ability to forecast patient length of stay with initial information (PLOSWI) upon patient intake allows hospitals to make informed operational decisions and provide optimal patient care.
Some of the many benefits associated with the ability to predict patient length of stay include increasing resource utilization, decreasing costs, and being prepared for emergencies.

In this project, we focus on developing a machine learning solution tailored to address the challenge of predicting patient length of stay based on data collected at the time of admission using both supervised and unsupervised learning methods.

### Dataset

Our dataset will be [real-world patient records acquired and made public by New York state hospitals in 2021](https://data.world/johnsnowlabs/hospital-inpatient-treatment-discharges-2021).
The dataset includes over one million data points, with 19 key features representing information about the patient collected at the time of patient intake (upon an initial diagnosis by a physician).
The features include information about the patient - such as their age, gender, and ethnicity - as well as information about their diagnosis, including a code representing the diagnosis, a description of the diagnosis, and the severity of the diagnosis.
The features also include other potentially relevant information such as insurance information and a predicted risk of mortality.


### Literature Review

There have been some related endeavours in the field of length of stay prediction.
Turgeman et al [1](#1). created a "Cubist" rule-based regression model to predict length of stay at the time of admission using data from Veterans Health Administration in Pittsburgh, PA.

## Problem Definition

Our problem centers on predicting patient length of stay with initial information (PLOSWI) in a hospital setting. Our primary aim is to assist managerial decision-making processes within a single hospital or any healthcare facility providing inpatient care. Specifically, we seek to forecast the duration of a patient's stay upon admission, relying solely on data collected on the admission day, such as demographics, diagnosis, severity of illness, treatment procedure decision, and risk of mortality. Utilizing these simple yet essential pieces of information, our objective is to predict the length of stay for newly admitted patients. This predictive capability is vital for the hospital's capacity planning and resource allocation, ultimately leading to improved utilization and reduced costs.

![inpatient image](https://intermountainhealthcare.org/-/media/images/images-sc9/medical-specialties/behavioral-health/hospital-patient-16x9.ashx?mw=500)
<span style="font-size:small; color:grey; font-style:italic;">Source:[Intermountain Healthcare](https://intermountainhealthcare.org/medical-specialties/behavioral-health/)</span>


## Method


## (Potential) Results and Discussion


## References

<a id="1">[1]</a>
L. Turgeman, J. May, and R. Sciulli,
"Insights from a machine learning model for predicting the hospital Length of Stay (LOS) at the time of admission,"
*Expert Systems with Applications*,
vol. 78,
July 2017


## Appendix

### Gantt Chart

### Contribution Table
