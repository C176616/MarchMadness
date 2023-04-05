# MarchMadness
 A dash app for march madness machine learning

Objectives:
save model
save tournament
move building structure out of app
implement the rest of the models
    remember to scale neural net
implement train/test split
allow download
implement binary tree iterator

test:

python -m pytest

deploy
rsconnect deploy dash ./ --name lillyRSConnect --python .venv/Scripts/python.exe 



testing 
coverage run -m pytest

publishing to posit-connect
rsconnect deploy dash . --python .venv/Scripts/python.exe --server https://rstudio-connect.am.lilly.com/



pathlib


Setting up a new season:
Move the following files into folder "data_stage2"
MNCAATourneyCompactResults.csv'
MNCAATourneySeeds.csv
MTeams
MNCAATourneySlots
MRegularSeasonDetailedResults
MNCAATourneySeeds
MNCAATourneyCompactResults

Open Part 1- Stage 1 data manipulation to create training_set.csv


run generate tournament
you will need to adjust the pre round 1 slot names




need to put in the following files


seeds['Seed'] =  pd.to_numeric(seeds['Seed'].str[1:3], downcast='integer',errors='coerce')
season_dresults = pd.read_csv(cwd +'/data_stage2/MRegularSeasonDetailedResults.csv')


need to run 
create the file "training_set.csv"
Result	Season	Team1	Team2	deltaSeed	deltaPointsFor	deltaPointsAgainst	deltaFGM	deltaFGA	deltaFGM3	deltaFGA3	deltaFTM	deltaFTA	deltaOR	deltaDR	deltaAst	deltaTO	deltaStl	deltaBlk	deltaPF	deltaWinPct
0	2003	1411	1421	0	1.593103448	-7.614942529	0.354022989	-1.526436782	-0.549425287	0.5	1.434482759	7.135632184	0.890804598	1.627586207	1.165517241	-0.973563218	-0.635632184	-0.766666667	-0.803448276	0.151724138

training_set_stage2?


add these files:
/data_stage2/MSampleSubmissionStage2.csv"

MNCAATourneySeeds
Season	Seed	TeamID
1985	W01	1207

MNCAATourneySlots
Season	Slot	StrongSeed	WeakSeed
1985	R1W1	W01	W16

MSampleSubmissionStage2
ID	Pred
2022_1103_1104	0.5

MTeams
TeamID	TeamName	FirstD1Season	LastD1Season
1101	Abilene Chr	2014	2022

create a df_training_set
df_training_set_stage 2



run app3