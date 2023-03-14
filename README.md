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