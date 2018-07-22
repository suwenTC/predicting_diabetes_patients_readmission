# Background
In this project, I compared different machine learning classifiers and used the best one to predict if a diabetes patient is subjected to hospital readmission. Hospital readmission rates for certain conditions are now considered an indicator of hospital quality, and also affect the cost of care adversely.

Note that getting a high predictive accuracy is not the ultimate goal in this situation. High sensitivity is more desirable because it allows a hospital to correctly identify if a patient is more likely being readmitted in the future. Once a patient is identified with a high risk of being readmitted, a hospital can take action immediately. And as a result, the readmission rate and the cost of care can be decreased.

# Data Description
The dataset represents 10 years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks. It includes over 50 features representing patient and hospital outcomes. Information was extracted from the database for encounters that satisfied the following criteria.

It is an inpatient encounter (a hospital admission).
It is a diabetic encounter, that is, one during which any kind of diabetes was entered to the system as a diagnosis.
The length of stay was at least 1 day and at most 14 days.
Laboratory tests were performed during the encounter.
Medications were administered during the encounter.
The data contains such attributes as patient number, race, gender, age, admission type, time in hospital, medical specialty of admitting physician, number of lab test performed, HbA1c test result, diagnosis, number of medication, diabetic medications, number of outpatient, inpatient, and emergency visits in the year before the hospitalization, etc.

Data Source: https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008