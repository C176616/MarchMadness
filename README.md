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
have them put their name on their bracket then submit
create a version where you can just select for regular tournaments
create a grading app that you can select the master bracket then grade the rest based on that bracket
