import random

def Resource(name):
    if name == "Desert":
        return 0
    elif name == "Clay":
        return 1
    elif name == "Ore":
        return 2
    elif name == "Sheep":
        return 3
    elif name == "Wheat":
        return 4
    elif name == "Wood":
        return 5
    else:
        print("Unknown name, return null")
        return None

def Port(name):
    if name == "None":
        return 0
    elif name == "Clay":
        return 1
    elif name == "Ore":
        return 2
    elif name == "Sheep":
        return 3
    elif name == "Wheat":
        return 4
    elif name == "Wood":
        return 5
    elif name == "31":
        return 6
    else:
        print("Unknown name, return null")
        return None

class Board:
    def __init__(self,Israndom):
        #Coord, [Resource, Number, Blocked,[Persons]]
        TileCoord = {}

        #Coord, [Person, Port,Town]
        NodeCoord = {}

        #Coord, [Person]
        EdgeCoord = {}

        #Harcood Standard
        TileCoord[0x13] = [5, 11, 0, [0,0,0,0]]
        TileCoord[0x15] = [3, 12, 0, [0,0,0,0]]
        TileCoord[0x17] = [4, 9, 0, [0,0,0,0]]

        TileCoord[0x22] = [1, 4, 0, [0,0,0,0]]
        TileCoord[0x24] = [2, 6, 0, [0,0,0,0]]
        TileCoord[0x26] = [1, 5, 0, [0,0,0,0]]
        TileCoord[0x28] = [3, 10, 0, [0,0,0,0]]

        TileCoord[0x31] = [0, 0, 1, [0,0,0,0]]
        TileCoord[0x33] = [5, 3, 0, [0,0,0,0]]
        TileCoord[0x35] = [4, 11, 0, [0,0,0,0]]
        TileCoord[0x37] = [5, 4, 0, [0,0,0,0]]
        TileCoord[0x39] = [4, 8, 0, [0,0,0,0]]

        TileCoord[0x42] = [1, 8, 0, [0,0,0,0]]
        TileCoord[0x44] = [3, 10, 0, [0,0,0,0]]
        TileCoord[0x46] = [3, 9, 0, [0,0,0,0]]
        TileCoord[0x48] = [2, 3, 0, [0,0,0,0]]

        TileCoord[0x53] = [2, 5, 0, [0,0,0,0]]
        TileCoord[0x55] = [4, 2, 0, [0,0,0,0]]
        TileCoord[0x57] = [5, 6, 0, [0,0,0,0]]

        #NodeCoord
        NodeCoord[0x14] = [0, 6, 0]
        NodeCoord[0x16] = [0, 3, 0]
        NodeCoord[0x18] = [0, 0, 0]

        NodeCoord[0x23] = [0, 6, 0]
        NodeCoord[0x25] = [0, 0, 0]
        NodeCoord[0x27] = [0, 3, 0]
        NodeCoord[0x29] = [0, 0, 0]

        NodeCoord[0x33] = [0, 0, 0]
        NodeCoord[0x35] = [0, 0, 0]
        NodeCoord[0x37] = [0, 0, 0]
        NodeCoord[0x39] = [0, 6, 0]

        NodeCoord[0x42] = [0, 2, 0]
        NodeCoord[0x44] = [0, 0, 0]
        NodeCoord[0x46] = [0, 0, 0]
        NodeCoord[0x48] = [0, 0, 0]
        NodeCoord[0x4a] = [0, 2, 0]

        NodeCoord[0x52] = [0, 2, 0]
        NodeCoord[0x54] = [0, 0, 0]
        NodeCoord[0x56] = [0, 0, 0]
        NodeCoord[0x58] = [0, 0, 0]
        NodeCoord[0x5a] = [0, 0, 0]

        NodeCoord[0x61] = [0, 0, 0]
        NodeCoord[0x63] = [0, 0, 0]
        NodeCoord[0x65] = [0, 0, 0]
        NodeCoord[0x67] = [0, 0, 0]
        NodeCoord[0x69] = [0, 0, 0]
        NodeCoord[0x6b] = [0, 6, 0]

        NodeCoord[0x71] = [0, 0, 0]
        NodeCoord[0x73] = [0, 0, 0]
        NodeCoord[0x75] = [0, 0, 0]
        NodeCoord[0x77] = [0, 0, 0]
        NodeCoord[0x79] = [0, 0, 0]
        NodeCoord[0x7b] = [0, 6, 0]

        NodeCoord[0x82] = [0, 4, 0]
        NodeCoord[0x84] = [0, 0, 0]
        NodeCoord[0x86] = [0, 0, 0]
        NodeCoord[0x88] = [0, 0, 0]
        NodeCoord[0x8a] = [0, 0, 0]

        NodeCoord[0x92] = [0, 4, 0]
        NodeCoord[0x94] = [0, 0, 0]
        NodeCoord[0x96] = [0, 0, 0]
        NodeCoord[0x98] = [0, 0, 0]
        NodeCoord[0x9a] = [0, 1, 0]

        NodeCoord[0xa3] = [0, 0, 0]
        NodeCoord[0xa5] = [0, 0, 0]
        NodeCoord[0xa7] = [0, 0, 0]
        NodeCoord[0xa9] = [0, 1, 0]

        NodeCoord[0xb3] = [0, 6, 0]
        NodeCoord[0xb5] = [0, 0, 0]
        NodeCoord[0xb7] = [0, 5, 0]
        NodeCoord[0xb9] = [0, 0, 0]

        NodeCoord[0xc4] = [0, 6, 0]
        NodeCoord[0xc6] = [0, 3, 0]
        NodeCoord[0xc8] = [0, 0, 0]

        #EdgeCoord Lite dumt när fler än e (borde fixa så man lägger till)
        EdgeCoord[0x16] = [0]
        EdgeCoord[0x18] = [0]
        EdgeCoord[0x1a] = [0]
        EdgeCoord[0x1c] = [0]
        EdgeCoord[0x1e] = [0]
        EdgeCoord[0x111] = [0]

        EdgeCoord[0x25] = [0]
        EdgeCoord[0x27] = [0]
        EdgeCoord[0x29] = [0]
        EdgeCoord[0x2b] = [0]

        EdgeCoord[0x34] = [0]
        EdgeCoord[0x36] = [0]
        EdgeCoord[0x38] = [0]
        EdgeCoord[0x3a] = [0]
        EdgeCoord[0x3c] = [0]
        EdgeCoord[0x3e] = [0]
        EdgeCoord[0x311] = [0]
        EdgeCoord[0x313] = [0]

        EdgeCoord[0x43] = [0]
        EdgeCoord[0x45] = [0]
        EdgeCoord[0x47] = [0]
        EdgeCoord[0x49] = [0]
        EdgeCoord[0x4b] = [0]

        EdgeCoord[0x52] = [0]
        EdgeCoord[0x54] = [0]
        EdgeCoord[0x56] = [0]
        EdgeCoord[0x58] = [0]
        EdgeCoord[0x5a] = [0]
        EdgeCoord[0x5c] = [0]
        EdgeCoord[0x5e] = [0]
        EdgeCoord[0x511] = [0]
        EdgeCoord[0x513] = [0]
        EdgeCoord[0x515] = [0]

        EdgeCoord[0x61] = [0]
        EdgeCoord[0x63] = [0]
        EdgeCoord[0x65] = [0]
        EdgeCoord[0x67] = [0]
        EdgeCoord[0x69] = [0]
        EdgeCoord[0x6b] = [0]

        EdgeCoord[0x72] = [0]
        EdgeCoord[0x74] = [0]
        EdgeCoord[0x76] = [0]
        EdgeCoord[0x78] = [0]
        EdgeCoord[0x7a] = [0]
        EdgeCoord[0x7c] = [0]
        EdgeCoord[0x7e] = [0]
        EdgeCoord[0x711] = [0]
        EdgeCoord[0x713] = [0]
        EdgeCoord[0x715] = [0]

        EdgeCoord[0x83] = [0]
        EdgeCoord[0x85] = [0]
        EdgeCoord[0x87] = [0]
        EdgeCoord[0x89] = [0]
        EdgeCoord[0x8b] = [0]

        EdgeCoord[0x94] = [0]
        EdgeCoord[0x96] = [0]
        EdgeCoord[0x98] = [0]
        EdgeCoord[0x9a] = [0]
        EdgeCoord[0x9c] = [0]
        EdgeCoord[0x9e] = [0]
        EdgeCoord[0x911] = [0]
        EdgeCoord[0x913] = [0]

        EdgeCoord[0xa5] = [0]
        EdgeCoord[0xa7] = [0]
        EdgeCoord[0xa9] = [0]
        EdgeCoord[0xab] = [0]

        EdgeCoord[0xb6] = [0]
        EdgeCoord[0xb8] = [0]
        EdgeCoord[0xba] = [0]
        EdgeCoord[0xbc] = [0]
        EdgeCoord[0xbe] = [0]
        EdgeCoord[0xb11] = [0]

        self.TileCoord = TileCoord
        self.NodeCoord = NodeCoord
        self.EdgeCoord = EdgeCoord

