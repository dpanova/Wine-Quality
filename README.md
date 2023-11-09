# Wine-Quality

Notes from the task: 
- how to further increase the quality of the red wine assortment
- what leads to a good red wine rating
- input data - chemical properties
- output data - wine rating

# Aim:
- find out the drivers of the ratings
- make a recommendation of the selection?
  (here I need to think how to do it)

# Game Plan
1. read in the data from csv **done**
2. split test and train (stratified sample) **done**
3. feature selection techniques
4. create a pipeline to take a model and test it 
5. suggested models - Random Forest, Logistic Regression, PCA, SVM, Neural Network, samething else? 
6. align on the evaluation metrics - precision? https://towardsdatascience.com/performance-metrics-confusion-matrix-precision-recall-and-f1-score-a8fe076a2262
7. what if we use clustering? 
8. create an automated package ?
9. it should be easy to understand from the model which are the drivers 
10. I can automate the data download from the web https://archive.ics.uci.edu/dataset/186/wine+quality 
11. note that this is an ordered output
12. k-fold cross validation? - we can use from here for the cross-validation https://www.kaggle.com/code/dawood619/logisticregression-with-lasso-rfe
13. check for feature correlation **done**
14. for the regression, we can play around with the L1 and L2 regularizations https://medium.com/@chandradip93/all-about-feature-selec-e6e88e8ccd46
15. add requirements file for packages - if not, install
16. maybe we need to stratify by quality 
17. it would be cool if we generate a pdf file out of the automated frame
