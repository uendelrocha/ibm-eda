import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def floatCols(dataframe):
    # Get float columns list
    mask = dataframe.dtypes == float
    return dataframe.columns[mask]

def intCols(dataframe):    
    # Get int columns list
    return dataframe.select_dtypes('int').columns

def objCols(dataframe):    
    # Get int columns list
    return dataframe.select_dtypes('object').columns

'''
Calculates log(1 + x).
  - value >= 0: log(1 + value)
  - value < 0: -log(1 + abs(value))
'''

def roundUp(number):

  if round(number, 0) > number: # round() rounded up
    result = round(number, 0)
  else: # round() rounded down
    result = round(number, 0) + 1

  return result


  (round(cities_succ, 0) + 1) if (cities_succ - round(cities_succ, 0)) > 0 else cities_succ

def myLog1p(value):

    #print(type(value))
    #print(value)

    if type(value) == pd.Series:
      result = value.apply(lambda x : -np.log1p(abs(x)) if x < 0 else np.log1p(x))
    else:
      result = -np.log1p(abs(value)) if value < 0 else np.log1p(value)

    #print(result)

    return result
    '''
    if x < 0:
      return -np.log1p(abs(value))

    return np.log1p(value)
    '''

'''
Customize DataFrame.describe()
'''
def myDescribe(dataframe, cols=[]):
    
    if len(cols) == 0:
        cols = floatCols(dataframe)
    
    # Add range
    dfStat = dataframe[cols].describe()
    dfStat.loc['range'] = dfStat.loc['max'] - dfStat.loc['min']

    # Add iqr
    dfStat.loc['iqr'] = dfStat.loc['75%'] - dfStat.loc['25%']

    # Add lower and upper bounds
    dfStat.loc['lower'] = dfStat.loc['75%'] - 1.5 * dfStat.loc['iqr']
    dfStat.loc['upper'] = dfStat.loc['75%'] + 1.5 * dfStat.loc['iqr']

    # Add Pearson
    dfStat.loc['pearson'] = (3 * (dfStat.loc['mean'] - dfStat.loc['50%'])) / dfStat.loc['std']

    # Add Skew
    dfStat.loc['skew'] = dataframe[cols].skew(axis=0)

    # Add kurtosis
    dfStat.loc['kurtosis'] = dataframe[cols].kurt(axis=0)
    
    for col in cols:
      mode = dataframe[col].mode().to_list()

      ## Add asymmetry
      if (abs(dfStat.loc['pearson', col]) > 0.15) and (abs(dfStat.loc['pearson', col]) < 1):
        dfStat.loc['asymmetry', col] = 'small'
      elif (abs(dfStat.loc['pearson', col]) > 1):
        dfStat.loc['asymmetry', col] = 'great'
      else:
        dfStat.loc['asymmetry', col] = 'none'

      # Add skewed
      if dfStat.loc['skew', col] == 0:
        dfStat.loc['skewed', col] = 'normal (0)'
      elif dfStat.loc['skew', col] < 0:
        dfStat.loc['skewed', col] = 'left (-)'
      elif dfStat.loc['skew', col] > 0:
        dfStat.loc['skewed', col] = 'right (+)'

      # Add outlier
      if dfStat.loc['kurtosis', col] > 3:
        dfStat.loc['outliers', col] = 'high'
      elif dfStat.loc['kurtosis', col] < 3:
        dfStat.loc['outliers', col] = 'low'
      elif dfStat.loc['kurtosis', col] == 3:
        dfStat.loc['outliers', col] = 'flat'



      # Add mode
      if len(mode) == len(dataframe):
        dfStat.loc['mode', col] = 'amodal'
      elif len(mode) <= 5:
        dfStat.loc['mode', col] = str(mode)
      else:
        dfStat.loc['mode', col] = 'multimodal'

    # Add count missing values
    dfStat.loc['NaN'] = dataframe[cols].isna().sum().reset_index()[0].to_list()

    # Add Dtypes
    dfStat.loc['Dtype'] = dataframe[cols].dtypes.reset_index()[0].to_list()

    display(dfStat.T)
    #display(df3[float_cols].describe().T)
    dataframe[cols].boxplot(figsize=(15, 5))
    #dataframe[cols].info()
    
    # df3.boxplot()


# Return a dataframe with distinct rows
def myDistinct(dataframe, col=''):
    result = pd.DataFrame()
    
    if isinstance(col, list):
        # Return a DataFrame
        result = dataframe.loc[:, col].drop_duplicates().sort_values(col)
    else:
        # Return a Series
        result = dataframe.loc[:, col].drop_duplicates().sort_values()
        
    return result

# Count and return NaN columns
def myNaN(dataframe):
  return dataframe[dataframe.columns[(dataframe
               .isna().sum() > 1)
             ]
  ].isna().sum().reset_index().rename(
    columns = {
        'index':'Column', 
        0:'Count NaN'})

# Update one dataframe column
def myUpdateCol (dataframe, col="", old="", new=""): 
    dataframe.loc[dataframe[col]==old, [col]]=new

# Update dataframe columns
def myUpdate (dataframe, cols=[], values=[('old', 'new')]):
    result = pd.DataFrame()
    for col in cols:
        for old, new in values:
            myUpdateCol(dataframe, col, old, new)

        # Fazemos um distinct para cada uma das colunas e ordenamos os dados em ordem alfabética
        #distinct.insert(cols.index(col), col, df2.loc[:, col].drop_duplicates().to_numpy())
        #result[col] = df2.loc[:, col].drop_duplicates().sort_values().to_numpy()
        result[col] = myDistinct(dataframe, col).to_numpy()

    # Retorna o resultado da atualização
    return result

def myGroupBy(dataframe, select_cols=['year', 'gdp', 'state'], 
              grouper_cols=['state', 'year'], 
              agg_cols=['gdp'], agg_type='sum', agg_having='year >= 2015'):

    grouped = dataframe[select_cols].groupby(grouper_cols)
    
    if agg_cols:
        grouped = grouped[agg_cols].agg(agg_type).reset_index()
    else:
        if agg_type == 'count':
            grouped = grouped.count()
        elif agg_type == 'sum':
            grouped = grouped.sum()
        elif agg_type == 'max':
            grouped = grouped.max()
        elif agg_type == 'min':
            grouped = grouped.min()
        elif agg_type == 'median':
            grouped = grouped.median()
        elif agg_type == 'mean':
            grouped = grouped.mean()
        else:
            grouped = grouped.count()
            
    if agg_having:        
        grouped = grouped.query(agg_having)
        
    return grouped


'''
  Remove outliers and show a report before and after changes
  Se a remoção for cumulativa, o cálculo do iqr é sensível a atualização dos dados e ocorre com os limites atualizados após remoção de outliers
  Se a remoção não for cumulativa, o cálculo do iqr não é sensível a atualização dos dados e oorre com os limites fixados antes da remoção dos outliers
'''
def rmOutliers(dataframe, cols=[], cumulative=False, verbose=False):
  result = dataframe.copy()

  if verbose:
    count = result.shape[0]
    total = 0

  for col in cols:
      
      
      if verbose:
        print(col)
        before = result.shape[0]
        print('Before:\t', before)

      if cumulative:
        q1, q3 = ( result[col].quantile(0.25), result[col].quantile(0.75) )
        iqr = q3 - q1
      else:
        q1, q3 = ( dataframe[col].quantile(0.25), dataframe[col].quantile(0.75) )
        iqr = q3 - q1
      
      lower_bound = q1 - 1.5 * iqr
      upper_bound = q3 + 1.5 * iqr
      
      result = result.loc[~((result[col] < lower_bound) | (result[col] > upper_bound))]
                            
      if verbose:
        after = result.shape[0]
        diff = before - after
        total += diff
        print('After:\t', after)
        print('Diff:\t', diff, f'({round(diff / count * 100, 2)}%)')
      
  if verbose:
    print('------\nTOTAL:\t', total, f'({round(total / count * 100, 2)}%)', 'outliers removed\n')
  
  return result

def lsOutliers(dataframe, cols=[], cumulative=False, verbose=False):

  result = dataframe.copy()
  values = 0
  outliers = 0

  for col in cols:

      if verbose:
        print(col)
        
        # Get NaN's total
        count = result[col].count() # result.loc[(result[col].isna())].shape[0]
        print('Count:\t\t', '{:8.0f}'.format(count))

      if cumulative:
          q1, q3 = ( result[col].quantile(0.25), result[col].quantile(0.75) )
          iqr = q3 - q1
      else:
          q1, q3 = ( dataframe[col].quantile(0.25), dataframe[col].quantile(0.75) )
          iqr = q3 - q1

      lower_bound = q1 - 1.5 * iqr
      upper_bound = q3 + 1.5 * iqr

      if cumulative:
          temp = result.copy()
          temp = temp.loc[~((temp[col] < lower_bound) | (temp[col] > upper_bound))]
          temp.loc[:, col] = np.nan
          result.loc[temp.index, col] = temp.loc[:, col]
          # Update inliers values to NaN
          # result[col].loc[~((result[col] < lower_bound) | (result[col] > upper_bound))] = np.nan
      else:
          temp = dataframe.copy()
          temp = temp.loc[~((temp[col] < lower_bound) | (temp[col] > upper_bound))]
          temp.loc[:, col] = np.nan
          result.loc[temp.index, col] = temp.loc[:, col]
          #result[col] = temp[col].copy()

      if verbose:
        # Get outliers
        after = result.loc[~(result[col].isna())].shape[0]
        outliers += after
        values += count
        
        print('Outliers:\t', '{:8.0f}'.format(after), '{:8.2f}%'.format(after / count * 100), '\n')

  if verbose:
    print('TOTAL')
    print('COUNT:\t\t', '{:8.0f}'.format(values), '{:8.2f}%'.format(100))
    print('OUTLIERS:\t', '{:8.0f}'.format(outliers), '{:8.2f}%'.format(outliers / values * 100), '\n')

  return result