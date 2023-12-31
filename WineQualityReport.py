from WineQuality import WineQualityClass
from fpdf import FPDF
from datetime import datetime

# Generate the wine quality report

csv_path = 'wine+quality/winequality-red.csv'
dtypes_path ='data_types.csv'

# initiate class
wine=WineQualityClass(csv_path=csv_path, dtype_path=dtypes_path)
# read the data from a csv
wine.get_data_csv()
# validate if the data is as expected
wine.validate_data()
# create binary variable for quality
wine.binary_variable_creation()
# split the data into test and train
wine.data_split()
# conduct correlation analysis
wine.correlation_analysis()
pca_results = wine.pca_analysis()
# run separate logistic regressions against each independent feature
wine.independent_logistic_regression()
# conduct lasso regression analysis for main drivers identification
acc_log, precision_log, recall_log, misclassified_log, log_reg_coef_non_zero =  wine.log_reg_results()
# conduct random forest analysis for main drivers identification
acc_rf, precision_rf, recall_rf, misclassified_rf = wine.random_forest_results()

# Generate the PDF report and save it
pdf = FPDF('P', 'mm', 'A4')

# page 1
# Report title, author, data of the report and disclaimer and dependant variable distribution
pdf.add_page()
pdf.set_margins(10, 10, 10)
pdf.set_font('Arial', 'B', 24)
pdf.cell(w = 180, h = 10, txt='Wine Quality Automated Report', align= 'C')
pdf.ln(20)
pdf.set_font('Arial', size= 10)
pdf.cell(w = 40, h = 10, txt='Author: Denitsa Panova')
pdf.set_font('Arial', size= 10,style='I')
pdf.ln(5)
pdf.cell(w = 40, h = 10, txt='Date: '+datetime.today().strftime('%Y-%m-%d') )
pdf.ln(5)
pdf.cell(w = 40, h = 10, txt='Disclaimer: The objective of this report is to present the outcomes generated by four distinct machine learning' )
pdf.ln(5)
pdf.cell(w = 40, h = 10, txt='learning techniques: correlation analysis, principal components analysis (PCA), random forest, and logistic' )
pdf.ln(5)
pdf.cell(w = 40, h = 10, txt='regression. A specialized interpretation is essential to derive accurate conclusions regarding the chemical')
pdf.ln(5)
pdf.cell(w = 40, h = 10, txt='factors influencing the high-quality wine.')

# Dependant variable analysis
pdf.ln(15)
pdf.set_font('Arial', size= 16, style='B')
pdf.cell(w = 40, h = 10, txt='Dependant variable distribution')
#df = wine.data

pdf.ln(10)
pdf.set_font('Arial', size= 12)
pdf.cell(w = 40, h = 10, txt='Original Variable')
pdf.ln(5)
pdf.image('dependant_original.png', x = 10, y = 90, w = 130, h = 0)

pdf.ln(85)
pdf.cell(w = 40, h = 10, txt='Created Variable')
pdf.image('dependant_created.png', x = 10, y = 180, w = 130, h = 0)

# add footer
pdf.set_font('Arial', 'I', 8)
pdf.set_y(260)
pdf.cell(0, 10, 'Page ' + str(pdf.page_no()), 0, 0, 'C')

# page 2
# correlation analysis
pdf.add_page()
pdf.set_margins(10, 10, 10)
pdf.set_font('Arial', size= 16, style='B')
pdf.cell(w = 40, h = 10, txt='Correlation Analysis')
pdf.image('feature_correlation.png', x = 10, y = 20, w = 200, h = 0)

# PCA analysis
pdf.ln(120)
pdf.cell(w = 40, h = 10, txt='PCA')
pdf.ln(10)
pdf.set_font('Arial', size= 12)
pdf.cell(w = 40, h = 10, txt='The explained variance for the first principal component is ' + str(round(pca_results[0],2)))

# add footer
pdf.set_font('Arial', 'I', 8)
pdf.set_y(260)
pdf.cell(0, 10, 'Page ' + str(pdf.page_no()), 0, 0, 'C')

# page 3
pdf.add_page()
pdf.set_margins(10, 10, 10)
# Independent Logistic Regressions
pdf.set_font('Arial', size= 16, style='B')
pdf.cell(w = 40, h = 10, txt='Independent Logistic Regressions')
pdf.image('logistic_regression.png', x = 10, y = 20, w = 160, h = 0)

# add footer
pdf.set_font('Arial', 'I', 8)
pdf.set_y(260)
pdf.cell(0, 10, 'Page ' + str(pdf.page_no()), 0, 0, 'C')

# page 4
pdf.add_page()
pdf.set_margins(10, 10, 10)
# logistic regression results
pdf.set_font('Arial', size= 16, style='B')
pdf.cell(w = 40, h = 10, txt='Lasso Logistic Regression Results')
pdf.ln(10)
pdf.set_font('Arial', size= 12)
pdf.cell(w = 40, h = 10, txt='The model accuracy is ' + str(round(acc_log,3)))
pdf.ln(7)
pdf.cell(w = 40, h = 10, txt='The model precision is ' + str(round(precision_log,3)))
pdf.ln(7)
pdf.cell(w = 40, h = 10, txt='The model precision is ' + str(round(recall_log,3)))
pdf.ln(10)

pdf.cell(w = 40, h = 10, txt='Below is a graph representing the distribution of the misclassified quality classes' )
pdf.ln(10)
pdf.image('misclassified_log.png', x = 10, y = 55, w = 130, h = 0)

pdf.ln(70)
pdf.cell(w = 40, h = 10, txt='Below is a graph representing the confusion matrix' )
pdf.image('lasso_confusion_matrix.png', x = 20, y = 150, w = 130, h = 0)

# add footer
pdf.set_font('Arial', 'I', 8)
pdf.set_y(260)
pdf.cell(0, 10, 'Page ' + str(pdf.page_no()), 0, 0, 'C')

# page 5
pdf.add_page()
pdf.set_margins(10, 10, 10)
pdf.set_font('Arial', size= 14, style='B')
pdf.cell(w = 40, h = 10, txt='Lasso Logistic Regression Drivers')
pdf.ln(10)
pdf.set_font('Arial', size= 12)
for f in log_reg_coef_non_zero:
    pdf.cell(w=40, h=10, txt='Feature ' + str(f[1]) + ' has a coefficient ' + str(round(f[0],3)))
    pdf.ln(10)

# add footer
pdf.set_font('Arial', 'I', 8)
pdf.set_y(260)
pdf.cell(0, 10, 'Page ' + str(pdf.page_no()), 0, 0, 'C')

# page 6
pdf.add_page()
pdf.set_margins(10, 10, 10)
# logistic regression results
pdf.set_font('Arial', size= 16, style='B')
pdf.cell(w = 40, h = 10, txt='Random Forest Results')
pdf.ln(10)
pdf.set_font('Arial', size= 12)
pdf.cell(w = 40, h = 10, txt='The model accuracy is ' + str(round(acc_rf,3)))
pdf.ln(7)
pdf.cell(w = 40, h = 10, txt='The model precision is ' + str(round(precision_rf,3)))
pdf.ln(7)
pdf.cell(w = 40, h = 10, txt='The model precision is ' + str(round(recall_rf,3)))
pdf.ln(10)

pdf.cell(w = 40, h = 10, txt='Below is a graph representing the distribution of the misclassified quality classes' )
pdf.ln(10)
pdf.image('misclassified_rf.png', x = 10, y = 55, w = 130, h = 0)

pdf.ln(80)
pdf.cell(w = 40, h = 10, txt='Below is a graph representing the confusion matrix' )
pdf.image('rf_confusion_matrix.png', x = 20, y = 170, w = 100, h = 0)

# add footer
pdf.set_font('Arial', 'I', 8)
pdf.set_y(260)
pdf.cell(0, 10, 'Page ' + str(pdf.page_no()), 0, 0, 'C')

# page 7
pdf.add_page()
pdf.set_margins(10, 10, 10)
pdf.set_font('Arial', size= 14, style='B')
pdf.cell(w = 40, h = 10, txt='Random Forest Drivers')
pdf.ln(10)
pdf.image('rf_feature_importance.png', x = 20, y = 20, w = 130, h = 0)

# add footer
pdf.set_font('Arial', 'I', 8)
pdf.set_y(260)
pdf.cell(0, 10, 'Page ' + str(pdf.page_no()), 0, 0, 'C')

pdf.output('wine_quality_report_above.pdf', 'F')
