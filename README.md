Two Files have been submitted for this project:

1)One is our own implementation of Random Forests in the file RForest.py
2)The second file is called ScikitComparison.py which utilizes Scikit Learn’s inbuilt libraries to implement Random Forests and KNN for classification. A grid search was performed to get the maximum results for various tuned parameters.

RandomForest.py:
Breast Cancer classification has been done using the Random Forests ensemble method. A collection of trees created using the ID3 algorithm, and using bootstrapping to create variations between the decision trees has been implemented in RForests.py. 
data.csv is the Breast Cancer Wisconsin dataset can be downloaded  from the kaggle website at the following link: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data. This file is kept at the same location as the Random Forests implementation at RandomForests.py. All of these files are zipped together and are hence readily runnable from any location, since they are in the same folder. The last two lines of this file runs the code and tests the random forest. The last before line is used to create the random forest model. The number of trees has been varied from 5 to 50 and results shown in the report.


ScikitComparison.py:
This file contains the implementation available from scikit learn library. These are supposed to be more optimised and hence has been used to compare our implementation of Random Forests with this. Sklearn contains this same dataset inbuilt and has the dataset is loaded using sklearn. This same dataset is then used to train sklearn’s Random Forest model as well as sklearn’s KNN model. To run this file, just Run the file in Python terminal, and the results are printed out showing the best model for maximum accuracy, and their best parameters. The results have been compared with our implementation and is published in the report.
