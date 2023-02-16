# MarchMadness
 A dash app for march madness machine learning

todo:
figure out how to plot a correlation matrix in plotly express

to publish:


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

pip install --proxy http://C176616:K@C122022abc@proxy.gtm.lilly.com:9000 *whateverpackage*

testing 
coverage run -m pytest

next phase:
DONE - have them put their name on their bracket then submit
DONE - create a version where you can just select for regular tournaments
DONE - create a grading app that you can select the master bracket then grade the rest based on that bracket
neural network


Todo:
include upset
add in pre-round games
borders
update textfont

DONE - fix the heatmap going back to negatives
DONE - remove final slot numbers
DO NOT DO - select tournament year to pull in different seeds

DONE - time outs is turnovers
rounding issues in accuracy
DONE - change 0.67 to a percent
consolidate some code for the models
DONE - error with callback if you move range sliders first
where is season 2022?
dynamically update options for number of layers vs number of trees