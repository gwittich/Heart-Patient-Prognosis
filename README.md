# Heart-Patient-Prognosis
Heart Patient Prognosis - Classification
# Heart Patient Prognosis

<p align="center">
  <img src="images/HeartPatient.jpg">
</p>

## Read Me

Data Scientist:   __Gail Wittich__<br>
Email:      gwittich@optusnet.com.au <br>
Website:    www.linkedin.com/in/gail-wittich <br>
Copyright:  Copyright 2020, Gail Wittich <br>

# Contents
*  Problem Statement
  *  Data Description

* Approach
  *  0.   Obtain Data
  *  1.   Pre-process the data
  *  2.   Data Analysis and Visualizations (EDA)
  *  3.   Prepare train and test datasets
  *  4.   Select models
  *  5.   Train models
  *  6.   Evaluate models
  *  7.   Optimize best model
  *  8.   Prediction on New Test data
  *  9.   Conclusion


# Problem Statement

Management of our client's hospital has been trying to improve heart patient outcomes by looking at historic data on patient survival. They could not identify the factors leading to positive patient outcomes.

Develop a model that will predict the patients' prognosis, in terms of whether they will survive for at least 1 year after initial treatment.

### Data Description
The dataset provided by the client contains patient records collected by the  hospital. 
The following is based on information provided by the client, in regard to the data:

#### The Target Variable
* **Survived_1_year**: contains binary values representing whether the patient survived for at least 1 year after treatment or not.

    *  **0** - indicates that the patient **did not** survive for 1 year after treatment
    *  **1** - indicates that the patient **survived** for at least 1 year after treatment

#### Other Features:

*  **ID_Patient_Care_Situation**: 
*  **Diagnosed_Condition**: The condition the patient was diagnosed as suffering from
*  **ID_Patient**: Patient identifier number
*  **Treated_with_drugs**: Class of drugs used during treatment
*  **Survived_1_year**: If the patient survived for at least 1 year after treatment or not. (0 indicates they did not, 1 indicates they did)
*  **Patient_Age**: Age of the patient at time of treatment
*  **Patient_Body_Mass_Index**: A calculated value based on the patient’s weight, height, etc.
*  **Patient_Smoker**: If the patient was a smoker or not
*  **Patient_Rural_Urban**: If the patient lives in Rural or Urban area
*  **Patient_mental_condition**: If the patient is mentally stable or unstable
*  **A, B, C, D, E, F, Z** : 7 individual features each representing one predefined condition.
    *  **1** - indicates indicates that the patient **did** have the predefined condition prior to presenting for treatment (for a suspected heart condition)
    *  **0** - indicates indicates that the pateint **did not** previously have the predefined condition prior to presenting for treatment (for a suspected heart condition)
* **Number_of_prev_cond**: A count of the number of predefined pre-existing as per the preceeding 7 columns.

A, B, C, D, E, F and Z are the previous conditions of the patient. 1 indicates that the patient had that condition previously (prior to treatment for heart condition) while 0 indicates that the pateint didn't previously have that condition.

EXAMPLE: If the entry in column 'A' is '1', the patient had that condition prior to presenting for treatment. 
If a patient previously had conditions 'A' and 'C' only, columns 'A' and 'C' will each contain '1', while features 'B', 'D', 'E', 'F', 'Z' will each contain '0'. 
The column 'Number_of_prev_cond' will have entry as '2' (i.e. 1 + 0 + 1 + 0 + 0 + 0 + 0 + 0 = 2)

# Approach
The task were undertaken in several notebooks as below:
> **01_Heart_Patient__DataPrep.ipynb**
> * Pre-process the data

> **02_Heart_Patient__DataExploration.ipynb**
 * Data Analysis and Visualizations (EDA)

> **03_Heart_Patient__Preprocessing.ipynb**
> * Prepare Train and test datasets
>> * Feature Generation (Feature Engineering)
>> * Encoding

> **04_Heart_Patient__ModelTuning.ipynb**
> * Select several models
> * Train models
> * Evaluate models
> * Optimize best model
>> * Feature Selection
>>> * Correlation
>>> * Boruta
>> * Paramater Tuning

> **05_Heart_Patient__PredictUnseenData.ipynb**
> * Prediction on New Test data

> * Conclusion (see end of this document)

## 01_DataPrep.ipynb

### 1.   Load Data
The data was loaded into 2 dataframes:
1.  **training_df**
2.  **new_test_df**


### 2.   Pre-process the data
The Training data is made up of **25079 rows x 18 columns**.

The feature 'Treated_with_drugs' had **dirty data** (multiple values representing the same drug) i.e. DX6 & Dx6 and dx6. All values in the feature 'Treated_with_drugs' were converted to upper case and trailing spaces were removed. This reduced the no. of unique values from 182 to 32.

'Patient_Smoker' alos had dirty data. All values in the feature 'Patient_Smoker' were converted to upper case and spelling errors corrected. This reduced the no of unique values from 10 to 3.

There were 1982 **duplicate rows** ie had the same 'Patient ID' and the same ID_Patient_Care_Situation. The duplicate rows were dropped from the dataframe.

**Unnesesary features**
As there is only one value in the 'Patient_mental_condition' feature, i.e. all patients are stable, this feature was removed from the dataframe because it does not provide any insight and wont affect the predictions. 
**NOTE**: This limits the value of the model for future data that may have patients that are not 'Stable'. The data remains in the dataset for future reference.

The 'new_test_df was inspected and found to be in the same format as the training_df had originally except for the Target feature, 'Survived_1_year', as expected. 

After a more detailed inspection of the data in new_test_df, the same operations were performed on the new_test_df as were perfomend on the training_df.
The new_test_df did not have any missing values.

The new_test_df is now the same format as the training_df a after Data Preparation, except for the Target feature, 'Survived_1_year', which is only in the training_df, as required.

## 02_DataExploration.ipynb

### 3.   Data Analysis and Visualizations (EDA)
The data was inspected and the following was observered:
 - There are 23097 records

 - The data set has 16 features plus the Target.

 - There are some missing values in the folowing features (non-null < 23097):
  * Treated_with_drugs____________23089 
  * A______________________________21830
  * B______________________________21830
  * C______________________________21830
  * D______________________________21830
  * E______________________________21830
  * F______________________________21830
  * Z______________________________21830
  * Number_of_prev_cond_________21830

 - This suggests that:
*   1259 patients (23089 - 21830 = 1259) did not have any of the predefined pre-existing conditions.
*   8 patients (23097 - 23089 = 8) were not treated with any drugs.

 - There is a mix of feature data types: float64(9), int64(5), object(3)

**Balanced Data?**\
Visualise the distribution of the target variable, 'Survived_1_year', to determine if the dataset is balanced.

<p align="center">
  <img src="images/Heart Patient Survival distribution.jpg">
</p>

There are more than 8000 patients who did not survive for 1 year after treatment and more than 14000 patients who survived for at least 1 year after treatment. The ratio is less than 1:2. This imbalance (skew) is aceptable and does not require addressing.

**Numeric (Continuous) variables**\
**Not all Numeric features are actually statistically significant Continuous Variables.** ie 
* ID_Patient_Care_Situation' is record identification data, they will be ignored for data analysis. 
* 'Patient_ID' is record identification data, they will be ignored for data analysis.
* 'Diagnosed_Condition' uses a number to represent conditions, making it a Categorical feature.
* A, B, C, D, E, F, Z are Categorical features. (addressed later)  A count of their values results in the feature 'Number_of_prev_cond'.

Leaving the following Numeric Variables with their corresponding count of unique values:
- Patient_Age                   74
- Patient_Body_Mass_Index    10599
- Number_of_prev_cond            5
- Survived_1_year                2
The number of unique values for each of the numeric features appears reasonable.  Each is with the range of possblities for that feature.
Note: 'Patient_Body_Mass_Index' is a value with 6 decimal places.

**Missing Values**\
'Number_of_prev_cond' has 1356 (5.87% of records) missing values.

Expected value of 'Number_of_prev_cond' is a discrete value integer ranging from 0 to 7.

'Number_of_prev_cond' is dependent on the values in the seven features - 'A', 'B', 'C', 'D', 'E', 'F', 'Z' (i.e. what specifically the previous conditions were).
One option is to fill the missing values with, for example, the Mode of the feature. Patients that DID NOT have any previous conditions would not have null values in the seven features - 'A', 'B', 'C', 'D', 'E', 'F', 'Z', meaning that the value of 'Number_of_prev_cond' would also not be Null, it would be '0'. Another option is to fill it with '0', to represent Unknown (prefered option in this situation).

**Basic Statistical Description of the data**\
This shows that the minimum and maximum values for numerical features are not all within expected ranges.

<p align="center">
  <img src="images/DataStatDesc.jpg">
</p>

**Box Plot of Numerical Features**

A Box Plot shows the data by feature, displaying:
- The five-number summary of a set of the data, ie:
  *  minimum
  *  first quartile (Q1)
  *  median
  *  third quartile (Q3)
  *  maximum
- Value of first quartile
- Value of third quartile
- Median
- outliers
- outlier values
- symmetry of data
- how tightly data is grouped (distributed)
- if and how the data is skewed.

**Box Plots:**\
<p align="center">
  <img src="images/outliers.jpg">
</p>
There are some outliers in the features:
 - 'Patient_Age'
 - 'Patient_Body_Mass_Index'
 - 'Number_of_prev_cond' (This is not of any concern - it is valuable data.)

**'Patient_Age' Outliers**\
The 8 Outliers for 'Patient_Age' are in excess of 110. This is unlikely since the oldest person ever to have lived in the country died before turning 103yo. The outliers were replaced with the mean.

**'Patient_Body_Mass_Index' Outliers**\
The outliers for 'Patient_Body_Mass_Index' are closer to zero than the bulk of the data. BMI = (Weight in kilograms) divided by (Height in metres squared)

A normal BMI is one that falls between 18.5 and 24.9. This indicates that a person is within the normal weight range for his or her height.

  Body Mass Index  (BMI)  | Weight Status
  -------------  | -------------
  Below 18.5 | Underweight
  18.5 - 24.9 |	Normal
  25.0 - 29.9	| Overweight
  30.0 plus	| Obese

Given the data above, the outliers for 'Patient_Body_Mass_Index' are acceptable data.


**Corelation**\
Heat map of the correlation analysis between the continuous varibles.

<p align="center">
  <img src="images/heart_correlation_heatmap.jpg">
</p>

There is very little correlation between variables.

**Categorical (Discrete) Variables**

 Variable | Unique Values
  -------------  | -------------
Diagnosed_Condition  |  53
Treated_with_drugs   |  33
Patient_Smoker       |   3
Patient_Rural_Urban  |   2
A                    |   2
B                    |   2
C                    |   2
D                    |   2
E                    |   2
F                    |   2
Z                    |   2
Survived_1_year      |   2

'Diagnosed_Condition' feature has 53 unique values - this is as expected.\
'Treated_with_drugs' feature has 32 unique values - this is as expected.\
'Patient_Smoker' has only 3 unique values - this is also expected (Yes, No, Unknown)

**Visualization of categorical Variables**

<p align="center">
  <img src="images/categoricalFeatures.jpg">
</p>

## 03_Heart_Patient__Preprocessing.ipynb
### 4.   Data Preprocessing
#### Feature Generation - Training Data
- 'Treated_with_drugs' column is a categorical column. In addition to single drug values, it has values representing combinations of drugs. It would be of value to know the impact of each drug alone. 
Split combined drug values into individual drugs and create dummies variables.

- 'Patient_Smoker' is also a categorical column. To create dummies for it 'Unknown' needs to be addressed. Fill with Mode ('NO')
There are several ways to deal with the category 'Unknown'. In this situation the safest thing is to consider it as missing data and replace those values '0' rather than the mode value of the column.

#### Data Encoding - Training Data
Convert the remaining categorical column to numerical using get_dummies() function of pandas (i.e. one hot encoding).

Change Datatype of the following columns to 'uint8' from 'float64' or 'int64' accordingly:\
'Diagnosed_Condition' \
'Patient_Age' \
'A'\
'B'\
'C'\
'D'\
'E'\
'F'\
'Z'\
'Number_of_prev_cond'\
'DX1'\
'DX2'\
'DX3'\
'DX4'\
'DX5'\
'DX6'

There are now no missing values and all the data is of numerical type

#### Data Identification
There are two ID columns - 'ID_Patient_Care_Situation' and 'Patient_ID'. We can Review with a view to removing these columns if these are do not provide any benefit. and there is not any id repeated Check these two ID columns.
There are 23097 unique 'ID_Patient_Care_Situation', the same no of total records in the Training data.

There are only 10570 unique values in the feature 'Patient_ID'. This means there were some patients who presented two or more times to the hospital for treatment (which is likely). And the same patient will have different caring condition for different presentations (visites to the hospital).

The combination of 'ID_Patient_Care_Situation' and 'Patient_ID' represent who and how many repeat patients there were. Therefore:

There is useful information in the feature 'ID_Patient_Care_Situation' This feature will be kept. (ie identified those who presented more than once)
Dropping 'Patient_ID' feature means losing information relating to a repeat patient. This feature will be kept.


#### Reorder features so Target is last feature in dataframe

All Data Preprocessing steps were repeated for the Test data as necessaary. 

#### Check Format of Training and Test Data consistency.


Now, the only difference between the format of the data in 'PP_train_df' and 'PP_test_df' is that 'PP_test_df' has no Survived_1_year', as this is the target feature.  



## 04_ModelTuning.ipynb
### 5.   Prepare train and test datasets
1) Separate the input and output variables:
- Features (Input variables) are represented by 'X' (capital)
- Target (Output variables) are represented by 'y' (lowercase)
2. Train/test split
- Split the training data into train and test sets.

### 6. Build Models
1) Select models
The target, 'Survived_1_year' has two possible values:
- 0 - meaning the patient DID NOT survive for one year after treatment began,
- 1 - meaning the patient DID survive for one year or longer after treatment began.

This is therfore a Classification Problem in Supervised Machine Learning.

Classification models used:
- Support Vector Classifier,
- Logistic Regression, 
- Decision Tree Classifier,
- Random Forest Classifier.

### 7. Train Models

### 8. Evaluate Models
The results of each model are as follows:

 MODEL NAME 		|	MODEL ACCURACY
 -------------  | -------------
 LinearSVC 	|	 0.7887555482492191
 LogisticRegression |		 0.7991705216142925
 DecisionTreeClassifier |		 0.8183659689399054
 RandomForestClassifier |		 0.8600602611315701

Random Forest Classifier with all the features gives F1 score of 86%.


### 9. Optimize Best Model
1) Feature Selection\
Random Forest Classifier is the best performig model. There are a lot of feaures to train the model on. Reducing the number of Features may improve the accuracy of the Random Forest model.

Both 'Feature Correlation' and 'Boruta Feature Selector' will be tried.

> A) Results of Feature Correlation (Importance):

FEATURE | IMPORTANCE
 -------------  | -------------
Diagnosed_Condition         |  -0.723040
Patient_ID                 |   -0.603341
Patient_Age                |   -7.735313
Patient_Body_Mass_Index    |  -11.161206
A                           |  -8.136319
B                          |   -5.692359
C                         |    -4.035315
D                         |   -11.275322
E                          |    0.375306
F                          |   -0.557779
Z                           |   1.419638
Number_of_prev_cond         | -10.995566
No_Drugs                    |   1.419638
DX1                         |  12.843942
DX2                         |  10.286165
DX3                        |  12.388574
DX4                        |    9.359594
DX5                       |    18.662407
DX6                        |  -32.952306
Patient_Smoker_NO          |   24.598817
Patient_Smoker_UNKNOWN      |   1.419638
Patient_Smoker_YES         |  -24.654729
Patient_Rural_Urban_URBAN   | -11.999322
Survived_1_year             |100.000000

Features with a Feature importance of (correlation) of greater than +0.7 or less than -0.7 will be used.\
Survived_1_year is excluded as it is the Target variable.\
**No. of correlated features : 20 from the 24 provided.**\
New datasets of just the most important features, ie those with a rank of 16 or better, were created (X_important_train and X_important_test) and the model rerun.
This resulted in a slight reduction of prediction performance when compared to the full featured Random Forest Model.
> **RandomForestClassifier With Important Features: 0.8590514362057449**

> B) Boruta ranks the importance of each feature in respect of the target feature as follows:\
<p align="center">
  <img src="images/BorutaFeatureRank2.jpg">
</p>

New datasets of just the most significant features, ie those with a rank of 16 or better, were created (X_ranked_train and X_ranked_test) and the model rerun.\
**No. of significant features (ie. Ranking < 17): 19 from the 24 provided.**
\
This resulted in a small improvement of prediction performance when compared to the full featured Random Forest Model.
> **RandomForestClassifier With Ranked Features: 0.8604923798358735**

After selecting the more relavent features, using Boruta, the Random Forest Classifier has performed slightly better based on the F1 score. There is also a reduction in the complexity of the model which is also very desirable.

2) Hyper Parameter Tunning\
Hyper parameter tunnning helps choose a set of optimal parameters for a model. \
Values for some parameters were chosen randomly ie. for max_depth, n_estimators. Tuning these and other parameters related to Random Forest model should improve the results of the model.

**Grid Search CV** was selected as the method to tune hyperparameters as it is exhaustive.
The best values for the parameters, from the range provided, was found to be:
> 'bootstrap': [False]\
> 'max_depth': [11]\
> 'max_features': [7]\
> 'min_samples_leaf': [3]\
> 'min_samples_split': [14]\
> 'n_estimators': [100]\
> 'random_state': [1]

> **RandomForestClassifier With Ranked Features and Tunned Parameters: 0.8702163061564059**

Using the dataset of of the more relavent features, using Boruta, and tuning the parameters for the Random Forest Classifier the model has performed better based on the F1 score. 

### 10. Conclusion
Random Forrest Classifier was the best performing Model from the four tested.
Boruta Feature Selection and Grid Search Paramater Tuning has been beneficial in improving the performance of the Random Forest Classifier for this data.

 MODEL NAME 		|	MODEL ACCURACY
 -------------  | -------------
 LinearSVC 	|	 0.7887555482492191
 LogisticRegression |		 0.7991705216142925
 DecisionTreeClassifier |		 0.8183659689399054
 RandomForestClassifier |		 0.8600602611315701
 RandomForestClassifier Important Features (Correlation)|		 0.8590514362057449
 RandomForestClassifier Ranked Features (Boruta)|		 0.8604923798358735
 RandomForestClassifier Ranked Features and Tuned Params |		 0.8702163061564059



## 05_Heart_Patient__PredictUnseenData.ipynb
### 11. Prediction on New Test data
Make Predictions on unseen Test Data for client assessment.

# **Client Assessment Results:**
The above predictions on the unseen test dataset (New_X) were submitted to the client for evaluation which resulted in a F1 Score of ...

# *Future improvements*

It would be interesting to see the change in results if the least significant 'previous conditions' were removed from the data.
I wonder what conditions those letters represent?
