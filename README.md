# Sepsis Prediction
## Early Prediction of Sepsis from Clinical Data
Authors: Omer Madmon, Omer Nahum

Sepsis is a life-threatening medical condition caused by an immune response to a serious infection. Every year about 7.1 million people in the United States develop sepsis and about 2,000,000 of them die. Early prediction of sepsis and starting medical treatment accordingly are critical and can save many lives. In this assignment we are asked to suggest and evaluate multiple approaches and algorithms for solving the early prediction task, given a time series data of 20k patients recorded while in ICU, containing medical and demographic features.

We will start by exploratory data analysis to better understand the dataset used for the prediction task, including understanding the features distributions, correlations and missing data, including hypothesis testing regarding differences between patients who developed sepsis and healthy patients.

Then, we decide on imputation policy for different features based on the EDA, and create statistical features based on the raw features in order to summarize the time dimension for each patient while still preserving as much information as possible. This way we represent each patient in a fixed dimension vector space.

We then decide on four algorithms to optimize and evaluate for the prediction task: logistic regression, kernel SVM, random forest and XG-boost. We have trained and optimized hyper parameters on the given train set and evaluated on the given test set. Our main result was obtained by XG-boost, having ~72% F1 on the test set. We also used feature importance techniques to better understand decisions made by learned models, which will be visualized in this report as well.

Full project report is available [here](https://github.com/omer6nahum/SepsisPrediction/blob/main/Report.pdf).

## Getting Started

```
git clone https://github.com/omer6nahum/SepsisPrediction.git
cd SepsisPrediction
conda env create -f environment.yml
conda activate hw1_env
python predict.py directory_name
```

Where `directory_name` should contain patient psv files.
