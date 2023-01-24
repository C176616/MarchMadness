import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pkg_resources

from binaryTree import Node
from PIL import Image, ImageDraw

from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV



cwd = os.getcwd()

# Load Data
tourney_cresults = pd.read_csv(cwd + '/data_stage2/MNCAATourneyCompactResults.csv')
seeds = pd.read_csv(cwd + '/data_stage2/MNCAATourneySeeds.csv')
seeds['Seed'] =  pd.to_numeric(seeds['Seed'].str[1:3], downcast='integer',errors='coerce')
season_dresults = pd.read_csv(cwd +'/data_stage2/MRegularSeasonDetailedResults.csv')

targetYear = 2003
tourney_cresults = tourney_cresults.loc[tourney_cresults['Season'] >= targetYear]

training_set = pd.DataFrame()

training_set['Result'] = np.random.randint(0,2,len(tourney_cresults.index))
training_set['Season'] = tourney_cresults['Season'].values
training_set['Team1'] = training_set['Result'].values * tourney_cresults['WTeamID'].values + (1-training_set['Result'].values) * tourney_cresults['LTeamID'].values 
training_set['Team2'] = (1-training_set['Result'].values) * tourney_cresults['WTeamID'].values + training_set['Result'].values * tourney_cresults['LTeamID'].values
training_set.head()


def delta_seed(row):
    cond = (seeds['Season'] == row['Season'])
    return seeds[cond & (seeds['TeamID'] == row['Team1'])]['Seed'].iloc[0] - seeds[cond & (seeds['TeamID'] == row['Team2'])]['Seed'].iloc[0]

def delta_winPct(row):
    cond1 = (record['Season'] == row['Season']) & (record['WTeamID'] == row['Team1'])
    cond2 = (record['Season'] == row['Season']) & (record['WTeamID'] == row['Team2'])
    return (record[cond1]['wins']/record[cond1]['games']).mean() - (record[cond2]['wins']/record[cond2]['games']).mean()

def get_points_against(row):
    wcond = (dfW['Season'] == row['Season']) & (dfW['WTeamID'] == row['WTeamID']) 
    fld1 = 'LScore'
    lcond = (dfL['Season'] == row['Season']) & (dfL['LTeamID'] == row['WTeamID']) 
    fld2 = 'WScore'
    retVal = dfW[wcond][fld1].sum()
    if len(dfL[lcond][fld2]) > 0:
        retVal = retVal + dfL[lcond][fld2].sum() 
    return retVal

def get_points_for(row):
    wcond = (dfW['Season'] == row['Season']) & (dfW['WTeamID'] == row['WTeamID']) 
    fld1 = 'WScore'
    lcond = (dfL['Season'] == row['Season']) & (dfL['LTeamID'] == row['WTeamID']) 
    fld2 = 'LScore'
    retVal = dfW[wcond][fld1].sum()
    if len(dfL[lcond][fld2]) > 0:
        retVal = retVal + dfL[lcond][fld2].sum() 
    return retVal

def get_remaining_stats(row, field):
    wcond = (dfW['Season'] == row['Season']) & (dfW['WTeamID'] == row['WTeamID']) 
    fld1 = 'W' + field
    lcond = (dfL['Season'] == row['Season']) & (dfL['LTeamID'] == row['WTeamID']) 
    fld2 = 'L'+ field
    retVal = dfW[wcond][fld1].sum()
    if len(dfL[lcond][fld2]) > 0:
        retVal = retVal + dfL[lcond][fld2].sum()
    return retVal

def delta_stat(row, field):
    cond1 = (record['Season'] == row['Season']) & (record['WTeamID'] == row['Team1'])
    cond2 = (record['Season'] == row['Season']) & (record['WTeamID'] == row['Team2'])
    return (record[cond1][field]/record[cond1]['games']).mean() - (record[cond2][field]/record[cond2]['games']).mean()

training_set['deltaSeed'] = training_set.apply(delta_seed,axis=1)
training_set.head()

record = pd.DataFrame({'wins': season_dresults.groupby(['Season','WTeamID']).size()}).reset_index();
losses = pd.DataFrame({'losses': season_dresults.groupby(['Season','LTeamID']).size()}).reset_index();

record = record.merge(losses, how='outer', left_on=['Season','WTeamID'], right_on=['Season','LTeamID'])
record = record.fillna(0)
record['games'] = record['wins'] + record['losses']

# create dataframes of both winners and losers
dfW = season_dresults.groupby(['Season','WTeamID']).sum().reset_index()
dfL = season_dresults.groupby(['Season','LTeamID']).sum().reset_index()

# add points for and points against data
record['PointsFor'] = record.apply(get_points_for, axis=1)
record['PointsAgainst'] = record.apply(get_points_against, axis=1)

# This cell takes ~3 min. To slides
cols = ['FGM','FGA','FGM3','FGA3','FTM','FTA','OR','DR','Ast','TO','Stl','Blk','PF']

for col in cols:
    print("Processing",col)
    record[col] = record.apply(get_remaining_stats, args=(col,), axis=1)

record['FGprct'] = record['FGM'] / record['FGA']  
record.tail()

# This will take ~ 3 min. To slides
cols = ['PointsFor','PointsAgainst','FGM','FGA','FGM3','FGA3','FTM','FTA','OR','DR','Ast','TO','Stl','Blk','PF']

for col in cols:
    print("Processing",col)
    training_set['delta' + col] = training_set.apply(delta_stat,args=(col,),axis=1)

training_set['deltaWinPct'] = training_set.apply(delta_winPct,axis=1)
training_set.head()

training_set.to_csv("training_set.csv", index=False)
record.to_csv("record.csv", index=False)







Part 2 - model exploration

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pkg_resources

from binaryTree import Node
from PIL import Image, ImageDraw

from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection

import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

cwd = os.getcwd()

training_set = pd.read_csv("training_set.csv")
record = pd.read_csv('record.csv')

correlation = training_set.corr()

plt.rcParams['figure.figsize'] = (20.0, 10.0)
sns.heatmap(correlation, vmax=.8, square=True)

sns.heatmap(correlation[["Result"]], vmax=.8, square=True, center=0, annot=True)

# cols = ['deltaSeed', 'deltaWinPct','deltaPointsFor','deltaPointsAgainst','deltaFGM','deltaFGA','deltaFGM3','deltaFGA3','deltaFTM',
#         'deltaFTA','deltaOR','deltaDR','deltaAst','deltaTO','deltaStl','deltaBlk','deltaPF']
cols = ['deltaSeed', 'deltaWinPct', 'deltaPointsFor', 'deltaFGM', 'deltaAst', 'deltaBlk']
# define features as X
X = training_set[cols]
# define target as the Result
y = training_set['Result']   


from sklearn import linear_model
linearModel = linear_model.LinearRegression()
linearModel.fit(X,y)
linearModel.score(X,y)
linearModel.predict(X)

for year in range(2003,2019):
    X_test = training_set[training_set['Season'] == year][cols]
    y_test = training_set[training_set['Season'] == year]['Result']

    df_results = X_test
    df_results['Prediction'] = linearModel.predict(X_test)
    df_results['Result'] = y_test
    df_results
    
    correct = df_results.loc[(df_results['Result']==0) & (df_results['Prediction']<0.5)].shape[0]
    correct = correct + df_results.loc[(df_results['Result']==1) & (df_results['Prediction']>0.5)].shape[0]

    total = df_results.shape[0]

    accuracy = correct/total

    error = -np.log(1-df_results.loc[df_results['Result'] == 0]['Prediction']).mean()
    print("Year:", year, "Error:", error, "Accuracy:", accuracy)


from sklearn.linear_model import LogisticRegression
logisticModel = linear_model.LogisticRegression(solver='lbfgs')
logisticModel.fit(X,y)
logisticModel.score(X,y)
for year in range(2003,2019):
    X_test = training_set[training_set['Season'] == year][cols]
    y_test = training_set[training_set['Season'] == year]['Result']

    df_results = X_test
    df_results['Prediction'] = logisticModel.predict_proba(X_test)[:,1]
    df_results['Result'] = y_test
    df_results

    correct = df_results.loc[(df_results['Result']==0) & (df_results['Prediction']<0.5)].shape[0]
    correct = correct + df_results.loc[(df_results['Result']==1) & (df_results['Prediction']>0.5)].shape[0]

    total = df_results.shape[0]

    accuracy = correct/total

    df_results.loc[df_results['Prediction'] > 0.9, 'Prediction']=0.99
    df_results.loc[df_results['Prediction'] < 0.1, 'Prediction']=0.01

    error = -np.log(1-df_results.loc[df_results['Result'] == 0]['Prediction']).mean()
    print("Year:", year, ","," Error:", error, "Accuracy:", accuracy)

from sklearn.ensemble import RandomForestClassifier
RFClassifier = RandomForestClassifier(n_estimators = 1)
RFClassifier.fit(X, y)
RFClassifier.score(X, y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1)
RFClassifier.fit(X_train,y_train)

RFClassifier.score(X_train, y_train)

RFClassifier.score(X_test, y_test)

for year in range(2003,2019):
    X_test = training_set[training_set['Season'] == year][cols]
    y_test = training_set[training_set['Season'] == year]['Result']

    #not sure if this should be :,1 or :,0
    df_results = X_test
    df_results['Prediction'] = RFClassifier.predict_proba(X_test)[:,1]
    df_results['Result'] = y_test
    df_results
    
    df_results.loc[df_results['Prediction'] > 0.9, 'Prediction']=0.99
    df_results.loc[df_results['Prediction'] < 0.1, 'Prediction']=0.01
    
    correct = df_results.loc[(df_results['Result']==0) & (df_results['Prediction']<0.5)].shape[0]
    correct = correct + df_results.loc[(df_results['Result']==1) & (df_results['Prediction']>0.5)].shape[0]

    total = df_results.shape[0]

    accuracy = correct/total

    error = -np.log(1-df_results.loc[df_results['Result'] == 0]['Prediction']).mean()
    print("Year:", year, ","," Error:", error, "Accuracy:", accuracy)