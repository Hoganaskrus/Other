from Game import Board
from Game import Person
from graphics import *
import math
import random

def main():
    endScore = 10
    bestScore = 0
    numberOfPlayer = 4
    IsVisible = True

    board = Board.Board(False)
    DCdeck = []
    friends = []
    Houses = {}

    if IsVisible:
        win = GraphWin("Catan", 500, 500)
        win.setBackground(color_rgb(255, 255, 255))
        i = 1
        for j in range(1,4):
            index = 16*i+1+2*j
            name, point = GetPictureNIdex(board.TileCoord[index][0],win.getHeight(),win.getWidth(),index)
            img = Image(point, name)
            img.draw(win)
            txt = Text(point,str(board.TileCoord[index][1]))
            cir = Circle(point,12)
            cir.setFill(color_rgb(255, 255, 255))
            cir.draw(win)
            if board.TileCoord[index][1] == 6 or board.TileCoord[index][1] == 8:
                txt.setTextColor(color_rgb(255, 0, 0))
            txt.draw(win)
        i += 1
        for j in range(1,5):
            index = 16*i+2*j
            name, point = GetPictureNIdex(board.TileCoord[index][0],win.getHeight(),win.getWidth(),index)
            img = Image(point, name)
            img.draw(win)
            txt = Text(point,str(board.TileCoord[index][1]))
            cir = Circle(point,12)
            cir.setFill(color_rgb(255, 255, 255))
            cir.draw(win)
            if board.TileCoord[index][1] == 6 or board.TileCoord[index][1] == 8:
                txt.setTextColor(color_rgb(255, 0, 0))
            txt.draw(win)
        i += 1
        for j in range(0,5):
            index = 16*i+1+2*j
            name, point = GetPictureNIdex(board.TileCoord[index][0],win.getHeight(),win.getWidth(),index)
            img = Image(point, name)
            img.draw(win)
            txt = Text(point,str(board.TileCoord[index][1]))
            cir = Circle(point,12)
            cir.setFill(color_rgb(255, 255, 255))
            cir.draw(win)
            if board.TileCoord[index][1] == 6 or board.TileCoord[index][1] == 8:
                txt.setTextColor(color_rgb(255, 0, 0))
            txt.draw(win)

        i += 1
        for j in range(1,5):
            index = 16*i+2*j
            name, point = GetPictureNIdex(board.TileCoord[index][0],win.getHeight(),win.getWidth(),index)
            img = Image(point, name)
            img.draw(win)
            txt = Text(point,str(board.TileCoord[index][1]))
            cir = Circle(point,12)
            cir.setFill(color_rgb(255, 255, 255))
            cir.draw(win)
            if board.TileCoord[index][1] == 6 or board.TileCoord[index][1] == 8:
                txt.setTextColor(color_rgb(255, 0, 0))
            txt.draw(win)
        i += 1
        for j in range(1,4):
            index = 16*i+1+2*j
            name, point = GetPictureNIdex(board.TileCoord[index][0],win.getHeight(),win.getWidth(),index)
            img = Image(point, name)
            img.draw(win)
            txt = Text(point,str(board.TileCoord[index][1]))
            cir = Circle(point,12)
            cir.setFill(color_rgb(255, 255, 255))
            cir.draw(win)
            if board.TileCoord[index][1] == 6 or board.TileCoord[index][1] == 8:
                txt.setTextColor(color_rgb(255, 0, 0))
            txt.draw(win)

    Jonathan0 = Person.Person("Jonathan0", 1, color_rgb(0,191,255))
    Jonathan1 = Person.Person("Jonathan1", 2, color_rgb(0, 191, 255))
    Jonathan2 = Person.Person("Jonathan2", 3, color_rgb(0, 191, 255))
    Jonathan3 = Person.Person("Jonathan3", 4, color_rgb(0, 191, 255))
    friends.append(Jonathan0)
    friends.append(Jonathan1)
    friends.append(Jonathan2)
    friends.append(Jonathan3)
    Jonathan0.hand["Clay"] = 100
    Jonathan0.hand["Wood"] = 100
    Jonathan0.hand["Wheat"] = 100
    Jonathan0.hand["Ore"] = 100
    Jonathan0.hand["Sheep"] = 100
    Jonathan0.numberofHouses = 100
    shuffleDC(DCdeck)
    print(board.TileCoord)


    #Debug
    for i in range(1):
        test = win.getMouse()
        print(test)

    for b in board.EdgeCoord:
        board.EdgeCoord[b] = [1]

    for b in board.NodeCoord:
        #st = ValidMoveNAct(board, DCdeck, Jonathan0, 0, b, 0, 0)

        DrawHouse(b,Jonathan0,Houses,win,win.getHeight(),win.getWidth())

    while bestScore != endScore:
        pass




def BotHandler(State):
    pass

#[Place House, Place Road, Town, Buy DC, [TradeOffer TradeWant, amount], DCact]
def ValidMove(board,person,action,index,trade,DCact):
    if action == 0:
        if index in board.NodeCoord:
            if board.NodeCoord[index][0] == 0:
                if person.hand["Clay"] > 0 and person.hand["Wood"] > 0 and person.hand["Wheat"] > 0 and person.hand["Sheep"] > 0:
                    if person.numberofHouses > 0:
                        a,b,c = NodetoEdge(index)
                        if all(NodeNeighbours(board,index)):
                            if person.id in [board.EdgeCoord[a][0], board.EdgeCoord[b][0], board.EdgeCoord[c][0]]:
                                return True
        return False
    if action == 1:
        if index in board.EdgeCoord:
            if board.EdgeCoord[index][0] == 0:
                if person.hand["Clay"] > 0 and person.hand["Wood"] > 0:
                    if person.numberofRoads > 0:
                        if any(RoadNeighbours(board, index, person.id)):
                            return True
        return False
    if action == 2:
        if index in board.NodeCoord:
            if board.NodeCoord[index][0] == person.id:
                if person.hand["Wheat"] > 1 and person.hand["Ore"] > 2:
                    if person.numberofTown > 0:
                        if board.NodeCoord[index][2] == 0:
                            return True
        return False
    if action == 3:
        if person.hand["Sheep"] > 0 and person.hand["Wheat"] > 0 and person.hand["Ore"] > 0:
            return True
        return False
    if action == 4:
        if trade[0] not in person.hand or trade[1] not in person.hand:
            return
        if person.hand[trade[0]] >= trade[2]:
            if trade[2] == 4:
                return True
            if trade[2] == 3:
                if 6 in person.ports:
                    return True
            if trade[2] == 2:
                if NametoPort(trade[0]) in person.ports:
                    return True
        return False
    if action == 5:
        if DCact in person.DC:
            if person.DC[DCact] > 0:
                return True

def ValidMoveNAct(board,DCdeck,person,action,index,trade,DCact):
    if action == 0:
        if index in board.NodeCoord:
            if board.NodeCoord[index][0] == 0:
                if person.hand["Clay"] > 0 and person.hand["Wood"] > 0 and person.hand["Wheat"] > 0 and person.hand["Sheep"] > 0:
                    if person.numberofHouses > 0:
                        if all(NodeNeighbours(board,index)):
                            if person.id in NodetoEdge(board,index):
                                a,b,c = NodetoTile(index)
                                if a in board.TileCoord:
                                    board.TileCoord[a][3][person.id] += 1
                                if b in board.TileCoord:
                                    board.TileCoord[b][3][person.id] += 1
                                if c in board.TileCoord:
                                    board.TileCoord[c][3][person.id] += 1
                                board.NodeCoord[index][0] += person.id
                                person.hand["Clay"] -= 1
                                person.hand["Wood"] -= 1
                                person.hand["Sheep"] -= 1
                                person.hand["Wheat"] -= 1
                                person.numberofHouses -= 1
                                if board.NodeCoord[index][1] != 0:
                                    person.ports[board.NodeCoord[index][1]-1] = 1
                                return "nice"
        return
    if action == 1:
        if index in board.EdgeCoord:
            if board.EdgeCoord[index][0] == 0:
                if person.hand["Clay"] > 0 and person.hand["Wood"] > 0:
                    if person.numberofRoads > 0:
                        if any(RoadNeighbours(board,index,person.id)):
                            board.EdgeCoord[index][0] += person.id
                            person.hand["Clay"] -= 1
                            person.hand["Wood"] -= 1
                            person.numberofRoads -= 1
                            return "nice"
        return
    if action == 2:
        if index in board.NodeCoord:
            if board.NodeCoord[index][0] == person.id:
                if person.hand["Wheat"] > 1 and person.hand["Ore"] > 2:
                    if person.numberofTown > 0:
                        if board.NodeCoord[index][2] == 0:
                            a, b, c = NodetoTile(index)
                            if a in board.TileCoord:
                                board.TileCoord[a][3][person.id] += 1
                            if b in board.TileCoord:
                                board.TileCoord[b][3][person.id] += 1
                            if c in board.TileCoord:
                                board.TileCoord[c][3][person.id] += 1
                            person.hand["Ore"] -= 3
                            person.hand["Wheat"] -= 2
                            person.numberofHouses += 1
                            person.numberofTown -=1
                            return "nice"
        return
    if action == 3:
        if person.hand["Sheep"] > 0 and person.hand["Wheat"] > 0 and person.hand["Ore"] > 0:
            person.DC[DCdeck.pop()] += 1
            return "nice"
        return
    if action == 4:
        if trade[0] not in person.hand or trade[1] not in person.hand:
            return
        if person.hand[trade[0]] >= trade[2]:
            if trade[2] == 4:
                person.hand[trade[0]] -= 4
                person.hand[trade[1]] += 1
                return
            if trade[2] == 3:
                if 6 in person.ports:
                    person.hand[trade[0]] -= 3
                    person.hand[trade[1]] += 1
                    return
            if trade[2] == 2:
                if NametoPort(trade[0]) in person.ports:
                    person.hand[trade[0]] -= 2
                    person.hand[trade[1]] += 1
                    return
        return
    if action == 5:
        if DCact in person.DC:
            if person.DC[DCact] > 0:
                if DCact == "Monopoly":
                    return
                if DCact == "Road Building":
                    return
                if DCact == "Year of Plenty":
                    return
                if DCact == "Knight":
                    return

def GetPictureNIdex(value,windowH,windowW,index):
    switcher = {
        0 : "desertHex.gif",
        1 : "clayHex.gif",
        2 : "oreHex.gif",
        3 : "sheepHex.gif",
        4 : "wheatHex.gif",
        5 : "woodHex.gif",
    }
    gameboardH = math.ceil(windowH * 0.75)
    gameboardW = math.ceil(windowW * 0.75)
    boardindexH = math.ceil(gameboardH * (1/7))
    boardindexW = math.ceil(gameboardW * (1/7))
    i = index // 16
    j = index % 16
    return switcher.get(value, "nothing"), Point(boardindexW + j*27, boardindexH +23.5 +(i-1)*47)

#Dosent work at rightside
def GetNode(point,windowH,windowW):
    gameboardH = math.ceil(windowH * 0.75)
    gameboardW = math.ceil(windowW * 0.75)
    boardindexH = math.ceil(gameboardH * (1/7))-9
    boardindexW = math.ceil(gameboardW * (1/7))-8.4
    x = point.x - boardindexW
    x = math.ceil(x//8 / 3)
    y = point.y - boardindexH
    y = math.ceil(y//8 / 3)+1
    if x < 1:
        x = 1
    if x > 11:
        x = 11
    if y < 1:
        y = 1
    if y > 12:
        y = 12
    return x,y

def shuffleDC(DCdeck):
    for i in range(14):
        DCdeck.append("Knight")
    for i in range(5):
        DCdeck.append("VP")
    DCdeck.append("Monopoly")
    DCdeck.append("Monopoly")
    DCdeck.append("Road Building")
    DCdeck.append("Road Building")
    DCdeck.append("Year of Plenty")
    DCdeck.append("Year of Plenty")
    random.shuffle(DCdeck)

#Stupid since we should save name at port instead
def NametoPort(name):
    switcher = {
        "Clay" : 1,
        "Ore" : 2,
        "Sheep" : 3,
        "Wheat" : 4,
        "Wood" : 5,
        "31" : 6
    }
    return switcher.get(name, "nothing")

def ResourcetoName(number):
    switcher = {
        1 : "Clay",
        2 : "Ore" ,
        3 : "Sheep",
        4 : "Wheat",
        5 : "Wood",
    }
    return switcher.get(number, "nothing")

def NodetoEdge(board,index):
    Nodey = index // 16
    Nodex = index % 16
    neigbour = []
    if Nodey % 2 == 0:
        if Nodex * 2 > 15:
            if (Nodex-1)*2 > 15:
                if (Nodey-1)*256+(Nodex-1)*32 in board.EdgeCoord:
                    neigbour.append(board.EdgeCoord[(Nodey-1)*256+(Nodex-1)*32][0])
                if (Nodey-1)*256+Nodex*32 in board.EdgeCoord:
                    neigbour.append(board.EdgeCoord[(Nodey-1)*256+Nodex*32][0])
                if Nodey*16+Nodex+1 in board.EdgeCoord:
                    neigbour.append(board.EdgeCoord[Nodey*16+Nodex+1][0])
                return neigbour
            else:
                if (Nodey-1)*16+(Nodex-1)*2 in board.EdgeCoord:
                    neigbour.append(board.EdgeCoord[(Nodey-1)*16+(Nodex-1)*2][0])
                if (Nodey-1)*256+Nodex*32 in board.EdgeCoord:
                    neigbour.append(board.EdgeCoord[(Nodey-1)*256+Nodex*32][0])
                if Nodey*16+Nodex+1 in board.EdgeCoord:
                    neigbour.append(board.EdgeCoord[Nodey*16+Nodex+1][0])
                return neigbour
        else:
            if (Nodey - 1) * 16 + (Nodex - 1) * 2 in board.EdgeCoord:
                neigbour.append(board.EdgeCoord[(Nodey - 1) * 16 + (Nodex - 1) * 2][0])
            if (Nodey - 1) * 16 + Nodex * 2 in board.EdgeCoord:
                neigbour.append(board.EdgeCoord[(Nodey - 1) * 16 + Nodex * 2][0])
            if Nodey * 16 + Nodex + 1 in board.EdgeCoord:
                neigbour.append(board.EdgeCoord[Nodey * 16 + Nodex + 1][0])
            return neigbour
    else:
        if Nodex * 2 > 15:
            if (Nodex-1)*2 > 15:
                if Nodey*256+(Nodex-1)*32 in board.EdgeCoord:
                    neigbour.append(board.EdgeCoord[Nodey*256+(Nodex-1)*32][0])
                if Nodey*256+Nodex*32 in board.EdgeCoord:
                    neigbour.append(board.EdgeCoord[Nodey*256+Nodex*32][0])
                if (Nodey-1)*16+Nodex+2 in board.EdgeCoord:
                    neigbour.append(board.EdgeCoord[(Nodey-1)*16+Nodex+2][0])
                return neigbour
            else:
                if Nodey*16+(Nodex-1)*2 in board.EdgeCoord:
                    neigbour.append(board.EdgeCoord[Nodey*16+(Nodex-1)*2][0])
                if Nodey*256+Nodex*32 in board.EdgeCoord:
                    neigbour.append(board.EdgeCoord[Nodey*256+Nodex*32][0])
                if (Nodey-1)*16+Nodex+2 in board.EdgeCoord:
                    neigbour.append(board.EdgeCoord[(Nodey-1)*16+Nodex+2][0])
                return neigbour
        else:
            if Nodey * 16 + (Nodex - 1) * 2 in board.EdgeCoord:
                neigbour.append(board.EdgeCoord[Nodey * 16 + (Nodex - 1) * 2][0])
            if Nodey*16+Nodex*2 in board.EdgeCoord:
                neigbour.append(board.EdgeCoord[Nodey*16+Nodex*2][0])
            if (Nodey - 1) * 16 + Nodex + 2 in board.EdgeCoord:
                neigbour.append(board.EdgeCoord[(Nodey - 1) * 16 + Nodex + 2][0])
            return neigbour

def NodetoTile(index):
    Nodey = index // 16
    Nodex = index % 16
    Tilecentery = math.floor(Nodey / 2)
    if Tilecentery % 2 == 0:
        if Tilecentery == 0:
            return 0,0,Nodey*16+Nodex-1
        else:
            return (Tilecentery-1)*16 +Nodex - 1, (Tilecentery)*16 +Nodex - 2,(Tilecentery) * 16 + Nodex
    else:
        return (Tilecentery)*16 +Nodex-2, (Tilecentery)*16 +Nodex, (Tilecentery+1) * 16 + Nodex -1

def NodeNeighbours(board,index):
    neigbour = []
    if (index // 16) % 2 == 0:
        if (index + 16) in board.NodeCoord:
            neigbour.append(board.NodeCoord[index+16][0] == 0)
        if (index-16-1) in board.NodeCoord:
            neigbour.append(board.NodeCoord[index-16-1][0] == 0)
        if (index-16+1) in board.NodeCoord:
            neigbour.append(board.NodeCoord[index - 16+1][0] == 0)
        return neigbour
    else:
        if (index - 16) in board.NodeCoord:
            neigbour.append(board.NodeCoord[index - 16][0] == 0)
        if (index + 16 + 1) in board.NodeCoord:
            neigbour.append(board.NodeCoord[index + 16 + 1][0] == 0)
        if (index + 16 - 1) in board.NodeCoord:
            neigbour.append(board.NodeCoord[index + 16 - 1][0] == 0)
        return neigbour

def RoadNeighbours(board,index,id):
    neigbour = []
    if (index // 16) % 2 == 0:
        if (index - 16+1) in board.EdgeCoord:
            neigbour.append(board.EdgeCoord[index - 16+1][0] == id)
        if (index - 16 + 3) in board.EdgeCoord:
            neigbour.append(board.EdgeCoord[index - 16 + 3][0] == id)
        if (index + 16 + 1) in board.EdgeCoord:
            neigbour.append(board.EdgeCoord[index + 16 + 1][0] == id)
        if (index + 16+3) in board.EdgeCoord:
            neigbour.append(board.EdgeCoord[index + 16+3][0] == id)
        if (index -2) in board.NodeCoord:
            neigbour.append(board.NodeCoord[index -2][0] == id)
        if (index + 16-2) in board.EdgeCoord:
            neigbour.append(board.NodeCoord[index + 16-2][0] == id)
        return neigbour
    else:
        if (index - 16-1) in board.EdgeCoord:
            neigbour.append(board.EdgeCoord[index - 16-1][0] == id)
        if (index + 16 - 1) in board.EdgeCoord:
            neigbour.append(board.EdgeCoord[index + 16 - 1][0] == id)
        if (index - 2) in board.EdgeCoord:
            neigbour.append(board.EdgeCoord[index - 2][0] == id)
        if (index +2) in board.EdgeCoord:
            neigbour.append(board.EdgeCoord[index +2][0] == id)
        if (index -2) in board.NodeCoord:
            neigbour.append(board.NodeCoord[index -2][0] == id)
        if (index + 16-3) in board.EdgeCoord:
            neigbour.append(board.NodeCoord[index + 16-3][0] == id)
        return neigbour

def RollDiceNGiveOut(board,friends):
    roll = random.randint(1, 6) + random.randint(1,6)
    if roll == 7:
        return True
    for tiles in board.TileCoord:
        if board.TileCoord[tiles][1] == roll:
            for i in range(len(friends)):
                friends[i].hand[ResourcetoName(board.TileCoord[tiles][0])] += board.TileCoord[tiles][3][i]
    return False

def DrawHouse(index,person,Houses,window,windowH,windowW):
    x = index % 16
    y = index // 16
    gameboardH = math.ceil(windowH * 0.75)
    gameboardW = math.ceil(windowW * 0.75)
    boardindexH = math.ceil(gameboardH * (1/7))-6
    boardindexW = math.ceil(gameboardW * (1/7))-28
    if y % 2 :
        rect = Rectangle(Point(boardindexW + x * 27 - 4, boardindexH  + (y - 1) * 23- 8),
                         Point(boardindexW + x * 27 + 4, boardindexH  + (y - 1) * 23 + 8))
    else:
        rect = Rectangle(Point(boardindexW + x * 27 - 4, boardindexH  + (y - 2) * 23 - 8+14),
                         Point(boardindexW + x * 27 + 4, boardindexH  + (y - 2) * 23 + 8+14))
    Houses[index] = rect
    Houses[index].setFill(person.color)
    Houses[index].draw(window)

if __name__ == "__main__":
    main()