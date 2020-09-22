import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

def feat_eng(df):
    '''
    Parameters: 
        df - a dataframe with features for the titanic dataset from Kaggle expected columns are ('Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked')
    Returns:
        df - the transformed dataframe ready for machine learning
    '''
    
    # Drop Name, Ticket & Cabin columns
    df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)
    
    # Create boolean columns for sex and Embarked
    dummies = pd.get_dummies(df[['Sex', 'Embarked']])
    df.drop(columns=['Sex', 'Embarked'],inplace=True)
    df = df.merge(dummies, how='inner', left_index=True, right_index=True)
    
    # Fill missing values in Age using Median
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    df = pd.DataFrame(imp.fit_transform(df), columns=df.columns)
    
    return df

def scale_feats(df):
    '''
    Parameters:
        df - a dataframe that has already been engineered by feat_eng
    Returns:
        df_scaled - A dataframe that is normalized
    '''
    scaler = MinMaxScaler()
    scaler.fit(df)
    df_scaled = pd.DataFrame(scaler.transform(df),columns=df.columns)
    
    return df_scaled
        