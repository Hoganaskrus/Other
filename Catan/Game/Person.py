
class Person:
    def __init__(self, name, id, color):
        self.name = name
        self.id = id
        self.color = color
        self.score = 0
        self.hand = {"Clay" :0,"Wood" :0,"Wheat" :0, "Ore" :0,"Sheep" :0}
        self.DC = {"Knight" :0, "VP" :0, "Monopoly":0,"Road Building":0,"Year of Plenty":0}
        self.knights = 0
        self.longestRoad = 0
        self.numberofHouses = 5
        self.numberofTown = 4
        self.numberofRoads = 15
        self.ports = [0]*6