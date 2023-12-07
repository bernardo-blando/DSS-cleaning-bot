from Map import Map
from Bot import Bot
from Game import Game
import importlib
import traceback
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os



LATENCY = 0.1
VISUALS = True
CLS = False
botName = 'Bot464808'
module = importlib.import_module(botName)
cls = getattr(module, botName)




#List of possible values for the weight of the heuristic of the adjacent tiles
# 0.2, 0.4, 0.6, 0.8, 1
hyperparameters = [
	["DISTANCE_PENALTY_MULTIPLIER", [11]],
]
dir_name = "graded_maps"
#LIST of maps based on all the files in the maps folder, followed by the width, number of stains and size of the stains
maps = []
for file in os.listdir(dir_name):
	if file.endswith(".csv"):
		map = []
		map.append(os.path.join(dir_name, file))
		map.append(int(file.split("_")[2]))
		map.append(int(file.split("_")[3]))
		map.append(int(file.split("_")[4]))
		map.append(int(file.split("_")[1]))
		maps.append(map)
print(maps)

#maps= [['graded_maps/map_7_50_50_1_0_0_0_0_.csv', 50, 50, 1, 7]]

scores = []
#A Loop that iterates through all possible combinations of hyperparameters
for m in range(len(maps)):
	for i in range(len(hyperparameters[0][1])):
		settings = {
			'mapName' : maps[m][0],
			'nrCols' : maps[m][1],
			'nrRows' : maps[m][1],
			'nrStains' : maps[m][2],
			'nrPillars' : 0,
			'nrWalls' : 0,
			'sizeStains' : maps[m][3],
			'sizePillars' : 0,
			'sizeWalls' : 0,
			'checkpoint' : [1,1],
		}
		
		MAX_STEPS = settings["nrCols"] * settings["nrCols"] * 2
		bot = cls(settings)
		myMap = Map(settings)
		if not getattr(bot, 'name', False):
			bot.setName(botName)

		bot.setDistancePenaltyMultiplier(hyperparameters[0][1][i])
		
		game = Game(bot, myMap, MAX_STEPS, LATENCY, VISUALS, CLS)
	
		try:
			game.play()
			score = game.energy
			
		except Exception as e:
			print(e)
			traceback.print_exc()
			score = 0
		scores.append({
						"DISTANCE_PENALTY_MULTIPLIER": hyperparameters[0][1][i],
						"MAP": maps[m][0],
						"MAP_LEVEL": maps[m][4],
						"SCORE": score/MAX_STEPS,
						"NR_PI": settings["nrPillars"],
						"NR_WA": settings["nrWalls"],
						"NR_ST": settings["nrStains"],
						"SZ_PI": settings["sizePillars"],
						"SZ_WA": settings["sizeWalls"],
						"SZ_ST": settings["sizeStains"],
						"MAP_SIZE": settings["nrCols"]
						})

scores = pd.DataFrame(scores)
print(scores.describe())

#Write df into csv file
scores.to_csv("scores.csv", index=False)


#plot to compare the distance multiplier
fig = sns.barplot(x="DISTANCE_PENALTY_MULTIPLIER", y="SCORE", hue="MAP", data=scores)
fig.figure.savefig('scores_distance_penalty_multiplier.png')
plt.show()

print(scores.describe())