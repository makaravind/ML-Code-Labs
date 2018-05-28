import pandas as pd
from sklearn import preprocessing


def convert_to_numeric_int(df, col_name):
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')

    n_null_rows = len(df[col_name][df[col_name].isnull()])
    if n_null_rows > 0:
        # print('dropping null rows... ', col_name, ' ', len(df[col_name][df[col_name].isnull()]))
        df.dropna(inplace=True)
    # print('converting ', col_name, ' to type int')
    df[col_name] = df.YearBuilt.astype(int)


def categorize_to_numeric(df, col_name):
    le = preprocessing.LabelEncoder()
    le.fit(df[col_name])
    df[col_name] = le.transform(df[col_name])
    return df


features = ['YearBuilt', 'LotArea', 'YrSold', 'MiscVal', 'SaleCondition']
target = ['SalePrice']


def getPreprocessedData():
    df = pd.read_csv('F:\ML-Labs\data\housing_train.csv')

    # print(df.head(2))

    # can create one new feature from yearBuild and yrSold (yearBuild - yrSold = age)

    df = df[features + target]
    # print(df.head(2))

    convert_to_numeric_int(df, 'YearBuilt')
    convert_to_numeric_int(df, 'LotArea')
    convert_to_numeric_int(df, 'YrSold')
    convert_to_numeric_int(df, 'MiscVal')
    convert_to_numeric_int(df, 'SalePrice') # target
    df = categorize_to_numeric(df, 'SaleCondition')

    # print('creating age feature from yearBuilt and yrSold')
    df['age'] = df['YrSold'].subtract(df['YearBuilt'])
    return df, features, target


def getTest():
    df = pd.read_csv('F:\ML-Labs\data\housing_test.csv')
    convert_to_numeric_int(df, 'YearBuilt')
    convert_to_numeric_int(df, 'LotArea')
    convert_to_numeric_int(df, 'YrSold')
    convert_to_numeric_int(df, 'MiscVal')
    df = categorize_to_numeric(df, 'SaleCondition')
    df = df[features]
    return df
