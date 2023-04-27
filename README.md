# Gaussian Process Regression on Parkinson

This repository contains an implementation of Gaussian Process Regression (GPR) for the estimation of total Unified Parkinson's Disease Rating Scale (UPDRS) from patient voice recordings. The results are compared with a linear regression model to demonstrate the superiority of GPR. You can find the detailed information on the `Gaussian Process on Parkinson.pdf` file.

## Introduction

Patients affected by Parkinson’s disease experience difficulties in controlling their muscles, leading to tremors, walking difficulties, and problems with starting movements. Many cannot speak correctly due to difficulties in controlling their vocal cords and vocal tract. Levodopa is prescribed to patients, but the treatment amount should be increased as the illness progresses and provided at the right time during the day to prevent the freezing phenomenon. An automatic way to measure total UPDRS must be developed using simple techniques easily managed by the patient or their caregiver. One possibility is to use patient voice recordings (easily obtained several times during the day through a smartphone) to generate vocal features that can be then used to regress total UPDRS. GPR is used on the public dataset [1] to estimate total UPDRS, and the results are compared to those obtained with linear regression.

## Dataset

The dataset used in this project is publicly available at [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring). It contains 5875 voice recordings from 42 Parkinson's disease patients. The dataset contains 22 features, including subject ID, test time, motor UPDRS, total UPDRS, and 18 vocal features.

## Packages Installation

To run the code in this repository, you need to install required packages using:

```
pip install -r requirements.txt
```

## Usage

1. Clone this repository.
2. Download the dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring).
3. Run the code using the downloaded dataset and run `python main.py` in the directory of the repository.

## Results

Gaussian Process Regression (GPR) significantly outperforms Linear Regression based on Linear Least Squares (LLS) in terms of both mean squared error and R2 score on the test set. GPR has three times less error on the test set in comparison with LLS, which is promising.

## References

[1] https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring

[2] Global, regional, and national burden of Parkinson’s disease, 1990–2016: a systematic analysis for the Global Burden of Disease Study 2016 https://www.thelancet.com/journals/laneur/article/PIIS1474-4422(18)30295-3

[3] Gaussian Processes for Machine Learning, Carl Edward Rasmussen and Christopher K. I. Williams, The MIT Press, 2006. ISBN 0-262-18253-X. http://www.gaussianprocess.org/gpml/

