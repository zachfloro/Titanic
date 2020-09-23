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
    
    # Create boolean columns for Embarked
    dummies = pd.get_dummies(df[['Embarked']])
    df.drop(columns=['Embarked'],inplace=True)
    df = df.merge(dummies, how='inner', left_index=True, right_index=True)
    # convert sex to 1 = female, 0 = male
    df['Sex'] = df['Sex'].map({'male':0, 'female':1})
    
    # Fill missing values in Age using Median
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    df = pd.DataFrame(imp.fit_transform(df), columns=df.columns)
    
    # Convert Age column into boolean columns
    df['<10'] = df['Age'] < 10
    df['10-20'] = (df['Age'] >= 10) & (df['Age'] < 20)
    df['20-40'] = (df['Age'] >= 20) & (df['Age'] < 40)
    df['40-60'] = (df['Age'] >= 40) & (df['Age'] < 60)
    df['>60'] = df['Age'] >= 60
    
    # Drop Age
    df.drop(columns=['Age'], inplace=True)
    
    # Convert Fare column into boolean columns
    df['Fare_25'] = df['Fare'] <= 25
    df['Fare_50'] = (df['Fare'] > 25) & (df['Fare'] <= 50)
    df['Fare_150'] = (df['Fare'] > 50) & (df['Fare'] <= 150)
    df['Fare_275'] = (df['Fare'] > 150) & (df['Fare'] <= 275)
    df['Fare_other'] = df['Fare'] > 275
    
    # Drop Fare
    df.drop(columns=['Fare'], inplace=True)
    
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
        