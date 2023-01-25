class Team():
    def __init__(self, seed, ID, name):        
        self.teamSeed = seed
        self.teamID = ID
        self.teamName = name
    
    def __str__(self):
        return str(self.teamName)

    def getID(self):
        return self.teamName

    def getString(self):
        return(str(self.teamName))