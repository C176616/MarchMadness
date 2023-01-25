class Team():
    def __init__(self):
        self.TeamName = ""
        self.TeamSeed = ""
        self.TeamID = 0
    
    def __str__(self):
        return self.TeamName

    # def populateTeamInfo(self):
    #     slot = df_slots[slot['Slot']==node.value]





    
# class Team():
#     def __init__(self):
#         self.TeamName = ""
#         self.TeamSeed = ""
#         self.TeamID = 0
    
#     def __str__(self):
#         return self.TeamName

#     def populateTeamInfo(self):
#         slot = df_slots[slot['Slot']==node.value]





#         #populate the slots
#         for node in self.nodeList:
#             a = tourneySlots[tourneySlots['Slot']==node.value]
#             node.team1Seed = a['StrongSeed'].values[0]
#             node.team2Seed = a['WeakSeed'].values[0]
        
#         #populate the IDs
#         tourneySeeds = pd.read_csv(csv)
#         tourneySeeds = tourneySeeds[tourneySeeds['Season']==season]
#         tourneySeeds.head(30)
        
# #         for i in range(0,31):
#         for node in self.nodeList:
# #             print(node.team1Seed)
# #             if [*tourn.nodeList[5].value][1] != '2':
# #             print(node.value)
#             if [*node.value][1] != '2' and [*node.value][1] != '3' and [*node.value][1] != '4' and [*node.value][1] != '5' and [*node.value][1] != '6':
# #                 print(node.value)
# #                 print(node.team1Seed)
# #                 print(node.team2Seed)
#                 a = tourneySeeds[tourneySeeds['Seed']==node.team1Seed]
#                 b = tourneySeeds[tourneySeeds['Seed']==node.team2Seed]
# #                 print(b)
#                 if(a.empty):
#                     print('a empty')
#                 if(b.empty):
#                     print('b empty')
# #                 print(a['TeamID'].values[0])
# #                 print(b['TeamID'].values[0])
#                 if a.empty != True:
#                     node.team1ID = a['TeamID'].values[0]
#                 if b.empty != True:
#                     node.team2ID = b['TeamID'].values[0]

#         self.teamName = teamNames[teamNames['TeamID']==self.teamID]
#         for node in self.nodeList:
#             if [*node.value][1] != '2' and [*node.value][1] != '3' and [*node.value][1] != '4' and [*node.value][1] != '5' and [*node.value][1] != '6':
#                 a = teamNames[teamNames['TeamID']==node.team1ID]
#                 b = teamNames[teamNames['TeamID']==node.team2ID]
#                 if a.empty != True:
#                     node.team1Name = a['TeamName'].values[0]
#                 if b.empty != True:
#                     node.team2Name = b['TeamName'].values[0]


