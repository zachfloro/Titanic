import pandas as pd

df = pd.read_csv('Data/train.csv')

def feat_eng(df):
    '''
    Parameters: 
        df - a dataframe with features for the titanic dataset from Kaggle expected columns are ('Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked')
    Returns:
        df_eng - the transformed dataframe ready for machine learning
    '''
