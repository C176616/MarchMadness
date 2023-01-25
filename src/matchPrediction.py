class MatchPrediction:
    def __init__(self,ID,pred):
        self.ID = ID
        IDArray = ID.split("_")
        self.year = IDArray[0]
        self.team1ID=IDArray[1]
        self.team2ID=IDArray[2]
        self.pred=pred
    
    def getID(self):
        print("2022"+"_"+str(self.team1ID)+"_"+str(self.team2ID))
        
    def __str__(self):
        return self.ID