from Map import Map
from Bot import Bot
from Game import Game
import importlib
import traceback
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd




LATENCY = 0
VISUALS = True
CLS = False
botName = 'Bot464808'
module = importlib.import_module(botName)
cls = getattr(module, botName)

#List of possible values for the weight of the heuristic of the adjacent tiles
# 0.2, 0.4, 0.6, 0.8, 1ß
hyperparameters = {
	"DISTANCE_PENALTY_MULTIPLIER": 11,
}
maps = [
	["maps/map_10_30_3_3_25_2_25_3_.csv", 30, 3, 3],
	["maps/map_10_30_40_1_40_2_0_0_.csv", 30, 40, 1],
	["maps/map_9_30_10_3_1_5_3_15_.csv", 30, 10, 3],
	["maps/map_7_30_30_2_0_0_0_0_.csv", 30, 30, 2],
	['graded_maps/map_7_50_50_1_0_0_0_0_.csv', 50, 50, 1],
	['graded_maps/map_7_25_1_4_0_0_0_0_.csv', 25, 1, 4],
]

map = maps[4]
MAX_STEPS = map[1] * map[1] * 2
settings = {
	'mapName' : map[0],
	'nrCols' : map[1],
	'nrRows' : map[1],
	'nrStains' : map[2],
	'nrPillars' : 0,
	'nrWalls' : 0,
	'sizeStains' : map[3],
	'sizePillars' : 0,
	'sizeWalls' : 0,
	'checkpoint' : [1,1],
}
bot = cls(settings)
myMap = Map(settings)
if not getattr(bot, 'name', False):
	bot.setName(botName)

bot.setDistancePenaltyMultiplier(hyperparameters['DISTANCE_PENALTY_MULTIPLIER'])
					
game = Game(bot, myMap, MAX_STEPS, LATENCY, VISUALS, CLS)
				
try:
	game.play()

	
except Exception as e:
	print(e)
	traceback.print_exc()



