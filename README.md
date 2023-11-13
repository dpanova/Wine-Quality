# Wine Quality Report Overall Documentation 

## Task Overview:
- Kaufland wants to improve the quality of the red wine selection the store has.
- Kaufland Analytics examined ~1600 samples of red wine ratings and the 11 chemical substances.
- The goal is to identify the good-quality-wine drivers (chemical elements) and provide  them as a recommendation to the team.
- Assumption - good wine is considered one with above 5 rating (quality ranges from 1 to 10), splitting the data into high-low quality buckets. 

## Data Files: 
- requirements.txt - python packages requirements 
- data_types.csv - manually created csv file to ensure any upcoming new input data will have the same data types and columns 
- wine+quality/winequality-red.csv - input data for the report (https://archive.ics.uci.edu/ml/datasets/wine+quality)
- wine.log - log file for WineQuality.py
- top_drivers_statistics.csv - statistical analysis after the WineQualityReport is conducted 

# Python Files: 
1. WineQuality.py - a python class is created to automate the analysis of the data. Documentation is available at https://dpanova.github.io/Wine-Quality/ 
2. WineQualityReport.py - a python file which uses WineQuality.py to produce an automated pdf report. The report steps are as follows:
- read the data from a csv
- validate if the data is as expected 
- create binary variable for quality 
- split the data into test and train 
- conduct correlation analysis 
- run separate logistic regressions against each independent feature
- conduct lasso regression analysis for main drivers identification
- conduct random forest analysis for main drivers identification

## Resulting Files:
1. wine_quality_report.pdf - is teh output from WineQualityReport.py 

## Analysis based on the wine_quality_report.pdf
### Dependant variable 
The original dependant variable is highly skewed. Additionally, it is observed that not all quality ratings are present (for, example 1,2 are absent). A new variable , quality bucket, is created which splits the data into:
- high quality, if quality > 5
- low quality, if quality <= 5

Confusion between quality levels 5 and 6 is anticipated, as quality is subjective and determined by individuals rather than a scientific method. Furthermore, both make up the majority of the data.

The distribution of the new variable shows that the data is now represented equally (54% and 46%) between the two new classes, therefore, a stratified sample is not needed.

### Correlation of the independent variables 
The Pearson correlation is utilized, revealing a strong correlation (above 0.5) between different acidity measures and pH, as well as between density - alcohol, total - free sulfur dioxide and density-acidity. Other pairs also exhibit a correlation of 0.3. To mitigate any potential impact on results, it is advisable to employ regularization techniques.

Principal component analysis (PCA) is a technique used to find underlying correlations that exist in a set of variables. The objective of the analysis is to take a set of n variables and to find correlations. The first component has explained variance 0.95, which is also suggests highly correlated features. 

### Individual regressions 
The goal of this visual analysis is to see if there is any evidence if a dependent variable (quality bucket) and the independent have strong linear relationship. The following variables exhibit linear relation:
1. fixed acidity 
2. residual sugar
3. free sulfur dioxide 
4. total sulfur dioxide 
5. pH

From the correlation analysis, it is known that the following variables are highly correlated (above 0.5):
- 1 and 5
- 3 and 4

### Lasso Regression Results
Due to the correlated features, regularized logistic regression is utilized. L1 (Lasso) encourages the model to have sparse coefficients, which leads to most of the coefficients to have a value zero.

Lasso accuracy after hyperparameter tuning: 
- accuracy is 0.74 accuracy
- precision and recall are similar to accuracy 
- as expected majority of the misclassified quality bucket actually has 5 and 6 as quality ratings 

Variables with non-zero coefficients are: 
- fixed acidity
- volatile acidity
- residual sugar (closer to zero)
- free sulfur dioxide (closer to zero)
- total sulfur dioxide (closer to zero)
- pH
- sulphates 
- alcohol

It's important to note a significant correlation between fixed acidity and pH, while other variables also demonstrate some degree of correlation.

Based on the coefficient values, the top 3 drivers are volatile acidity, pH and sulphates, however, the volatile acidity may be inflated due to correlation. 
### Random Forest Results

To address correlated features, a random forest model is employed. It exhibits superior accuracy (0.83) in comparison to Lasso, suggesting that the dependent variable is not linearly dependent. Recall and precision values are comparable. Notably, there is a tendency for quality levels 5 and 6 to be frequently misclassified.. 

According to the feature importance analysis, the top three variables are identified as the primary drivers for the high-quality category.: 
- alcohol
- sulphates
- volatile acidity 

Examining the confusion matrix, it is evident that these variables do not exhibit a high level of correlation.

### Conclusion 
Opting for the random forest model due to its higher accuracy, we decide to focus on the top 3 drivers identified by that model. As the next step, to examine the actual relationship with quality, as opposed to the quality bucket, we conduct statistical analyses together with domain knowledge expertise from the following article https://www.ncbi.nlm.nih.gov/books/NBK531662/.
