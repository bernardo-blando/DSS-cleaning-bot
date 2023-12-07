from Bot import *
import numpy as np
from scipy.ndimage import convolve

class Bot464808(Bot):

	def __init__(self, settings):
		super().__init__(settings)
		self.ADJACENT_CELLS_RELATIVE_COORDINATES = np.array([(-1, 0), (0, -1), (0, 1), (1, 0)])
		self.ADJACENT_CELLS_RELATIVE_COORDINATES_WITH_DIAGONALS = np.array([(-1, -1), (-1, 0), (-1, 1) ,(0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)])
		self.setName('Curiosity')
		self.info = settings
		self.setDistancePenaltyMultiplier(11)
		self.setStainBt1()
		self.known_map = self.createEmptyMap(self.info['nrCols'], self.info['nrRows'], 'memory')
		self.current_cell = (1, 1)
		self.update_heuristics = False
		self.must_see_cells = self.createEmptyMap(self.info['nrCols'], self.info['nrRows'], 'mustSee')
		self.updateHeuristicsMap()
		self.active_path = [tuple(), []] 
#1st Level Functions
	def nextMove(self, current_cell, currentEnergy, vision, remainingStainCells):
		vision[1][1] = "."
		self.current_cell = tuple(current_cell)
		self.processVision(vision)
		direction = self.getDirection()
		self.last_move = direction
		return direction
#2nd Level Functions
	def processVision(self, vision):
			new_stains=[]
			new_obstacles=[]
			#Update the bot's known map with new discovered cells
			visionWithCoordinates = self.appendVisionAbsoluteCoordinates(vision)
			if self.known_map[1,1] != '?':
				if self.last_move == UP :
					visionWithCoordinates = visionWithCoordinates[[0,1,2,4]]
				elif self.last_move == LEFT :
					visionWithCoordinates = visionWithCoordinates[[0,3,4,6]]
				elif self.last_move == DOWN :
					visionWithCoordinates = visionWithCoordinates[[4,6,7,8]]
				elif self.last_move == RIGHT :
					visionWithCoordinates = visionWithCoordinates[[2,4,5,8]]

			for x, y, val in visionWithCoordinates:
				#Check if this is a known stain to know whether or not to make predictions about it
				if self.known_map[x.astype(int)][y.astype(int)] == '?':
					if val == '@':
						new_stains.append((x.astype(int), y.astype(int)))
						self.update_heuristics = True
					elif val == 'x':
						new_obstacles.append((x.astype(int), y.astype(int)))
						self.update_heuristics = True

				self.updateCell(self.known_map, x.astype(int), y.astype(int), val)

				self.updateCell(self.must_see_cells, x.astype(int), y.astype(int), 0)

			#If there are new stains then deduct the position of the whole stain.
			# if self.is_stain_size_bt_1 and len(new_stains) != 0:
			# 	self.deductStainPosition(new_stains)		
			
			#Deduct wal and pilar positions from the known map					
	def getDirection(self):
		brc = self.brc
		#if the bot is at the target cell or the heuristics map needs to be updated
		if brc == self.current_cell or self.update_heuristics:
			oldbrc = brc
			self.active_path = [tuple(), []]   # clear the active path
			self.updateHeuristicsMap()
			self.update_heuristics = False
			if self.brc == oldbrc:
				self.setDistancePenaltyMultiplier(self.DISTANCE_PENALTY_MULTIPLIER - 0.1)

			return self.getDirection()
		
		# Check if the path to the target cell is already cached
		if brc == self.active_path[0]:
			path = self.active_path[1]
		else:
			path = self.aStarSearch(tuple(self.current_cell), brc)
			self.active_path = [brc, path]  # cache the path for future use

		if  self.known_map[brc[0]][brc[1]]=='x' or self.known_map[path[1][0]][path[1][1]] == 'x':
				self.active_path = [tuple(), []]   # clear the active path
	
				return self.getDirection()
		
		# Get the next cell from the path
		if len(path) > 1:
			next_cell = path[1]
		elif len(path) == 1:
			next_cell = path[0]
		else:
			oldbrc=brc
			for cell in self.ADJACENT_CELLS_RELATIVE_COORDINATES:
				self.updateCell(self.known_map, cell[0] + brc[0], cell[1] +brc[1], 'x')
				self.updateCell(self.must_see_cells, cell[0] + brc[0], cell[1] +brc[1], 0)

			self.updateHeuristicsMap()
			newbrc=self.brc
			if oldbrc == newbrc:
				except_msg = "The old BRC is the same as the new BRC - The program is looping"
				raise Exception(except_msg)
			return self.getDirection()

		# Get the direction to the next cell
		direction = None
		dx = next_cell[0] - self.current_cell[0]
		dy = next_cell[1] - self.current_cell[1]
		if dx != 0 or dy != 0:
			if dx > 0:
				direction = DOWN
			elif dx < 0:
				direction = UP
			elif dy > 0:
				direction = RIGHT
			elif dy < 0:
				direction = LEFT
			else:
				except_msg = "Unexpected direction: dx = {}, dy = {}".format(dx, dy)
				raise Exception(except_msg)
			
		self.active_path[1] = path[1:]  # remove the first cell from the path
		return direction
	
#3rd Level Functions
	def updateHeuristicsMap(self):
		self.heuristics_map = []

		#Calculate the heuristic value of each cell by summing the number of unseen must see cells surounding it
		adjacent_cell_visibility_score_map = self.getCellVisibilityScoreMap(3)
		wide1_cell_visibility_score_map = self.getCellVisibilityScoreMap(5)

		distance_penalty_map = self.getDistancePenaltyMap()
		#Subtract the distance penalty from the heuristic value of each cell
		self.heuristics_map = np.round(adjacent_cell_visibility_score_map * 100, 2)
		
		self.heuristics_map[self.known_map == 'x'] = -100
		self.heuristics_map[self.known_map == '@'] = 100

		self.heuristics_with_distance_penalty = np.round(self.heuristics_map - (distance_penalty_map*self.DISTANCE_PENALTY_MULTIPLIER), 1)

		self.update_heuristics = False
		self.brc = self.getBestRankingCell()
	def deductStainPosition(self, new_stains):
		for stain in new_stains:
			up, left, right, down = self.getAbsoluteCoordinatesOfAdjacentCells(False, stain)

			if self.known_map[up[0]][up[1]] in ['.', 'x', 'Æ']: #Is the square on top of my new stain free or unknown?
				stain_rows = np.array(range(up[0]+1, up[0]+self.info['sizeStains']+1))
			elif self.known_map[down[0]][down[1]] in ['.', 'x', 'Æ']:  #Is the square under of my new stain free or unknown?
				stain_rows = np.array(range(down[0]-self.info['sizeStains'], down[0]))
			else: #This is not a corner stain
				continue
			if self.known_map[left[0]][left[1]] in ['.', 'x', 'Æ']: #Is the square on the left of my new stain free or unknown?
				stain_cols = np.array(range(left[1]+1, left[1]+self.info['sizeStains']+1))
			elif self.known_map[right[0]][right[1]] in ['.', 'x', 'Æ']: #Is the square on the right of my new stain free or unknown?
				stain_cols = np.array(range(right[1]-self.info['sizeStains'], right[1]))
			else: #This is not a corner stain
				continue

			stain_coordinates = [(x, y) for x in stain_rows for y in stain_cols]
			for x, y in stain_coordinates:
				self.updateCell(self.known_map, x, y, '@')
	def getScoreForTarget(self, target='@', marking_value=1, base_value=0):
		
		# Create a result array with default values
		result = np.full(self.known_map.shape, base_value, dtype=int)
		
		# Mark the locations of the target_value with the marking_value
		result[self.known_map == target] = marking_value
		
		return result
	def getCellVisibilityScoreMap(self, size):

		kernel = np.ones((size, size))
		map = convolve(self.must_see_cells, kernel, mode='constant', cval=0)
		response = map/size**2
		if size == 3:
			response[1:2 ,:] = map[1:2 ,:]/6
			response[-2:-1 ,:] = map[-2:-1 ,:]/6
			response[:, 1:2] = map[:, 1:2]/6
			response[:, -2:-1] = map[:, -2:-1]/6

		response = np.round(response, 2)
		return response
	def getBestRankingCell(self):
		brc = tuple(np.unravel_index(np.argmax(self.heuristics_with_distance_penalty, axis=None), self.heuristics_with_distance_penalty.shape))
		return brc		   
	def aStarSearch(self, start, goal):
		open_set = [start]
		came_from = {}
		g_score = {start: 0}
		f_score = {start: self.heuristics_map[start]}

		while open_set:

			# get cell in open set with lowest f_score
			current = min(open_set, key=lambda x: f_score.get(x, float('inf')))

			# If this cell is the goal, reconstruct the path
			if current == goal:
				return self.reconstructPath(came_from, current)

			open_set.remove(current)
			neighbours = self.getAbsoluteCoordinatesOfAdjacentCells(False, current)
			neighbours = [(x, y) for x, y in neighbours if self.known_map[x][y] != 'x']
			for neighbour in neighbours:
				dx = neighbour[0] - current[0]
				dy = neighbour[1] - current[1]
				# validate neighbour bounds and ensure it's not a wall
				if (0 <= neighbour[0] < self.info['nrRows'] and 
					0 <= neighbour[1] < self.info['nrCols'] and 
					(dx != 0 or dy != 0)):  # exclude the center cell

					tentative_g_score = g_score[current] + 1  # assuming every move has a cost of 1
					if tentative_g_score < g_score.get(neighbour, float('inf')):
							# This path to neighbour is better than any previous one. Record it!
							came_from[neighbour] = current
							g_score[neighbour] = tentative_g_score
							f_score[neighbour] = (tentative_g_score + self.heuristics_map[neighbour[0]][neighbour[1]])

							if neighbour not in open_set:
								open_set.append(neighbour)
								
		# There is no possible path to the goal
		self.updateCell(self.known_map, goal[0], goal[1], 'x') #mark the goal as a wall (it could also be the inside of a pillar or a unaccessible part of the map)
		self.updateCell(self.must_see_cells, goal[0], goal[1], 0) #mark the goal as a wall (it could also be the inside of a pillar or a unaccessible part of the map)
		self.updateHeuristicsMap()
		return self.aStarSearch(start, self.brc) #reiterate the search with a new goal
	def updateCell(self, map, x, y, val):
		map[x][y] = val	

#4th Level Functions
	def appendVisionAbsoluteCoordinates(self, vision):
		"""Append absolute coordinates to a vision list.

		Parameters:
		current_cell (tuple): Current cell coordinates (row, column).
		vision (list): Vision data.

		Returns:
		list: Vision data with absolute coordinates.
		"""
		# Calculate absolute coordinates for adjacent cells
		absolute_coordinates = self.getAbsoluteCoordinatesOfAdjacentCells(True, self.current_cell)
    
		# Flatten the vision data
		flatten_vision = np.array(vision).flatten()
    
		# Combine absolute coordinates with vision data
		vision_with_coordinates = np.column_stack((absolute_coordinates, flatten_vision))
		
		return vision_with_coordinates
	def reconstructPath(self, came_from, current):
		path = [current]
		while current in came_from:
			current = came_from[current]
			path.append(current)
		path.reverse()  # reverse to get path from start to end
		return path
	def getAbsoluteCoordinatesOfAdjacentCells(self, diagonals, cell):
		row, col = cell
		if diagonals:
			absolute_coordinates = np.add(self.ADJACENT_CELLS_RELATIVE_COORDINATES_WITH_DIAGONALS, (row, col))
		else:
			absolute_coordinates = np.add(self.ADJACENT_CELLS_RELATIVE_COORDINATES, (row, col))
			
		return absolute_coordinates
	
	#HELPER FUNCTIONS
	def print_2d_array(self, arr):
		# Print column numbers
		print("   ", end="")
		for col in range(len(arr[0])):
			print("{:4}".format(col), end="")
		print()

		# Print rows with row numbers
		for row_num, row in enumerate(arr):
			print("{:3d}".format(row_num), end="")
			for val in row:
				print("{:4}".format(val), end='')
			print()  # To move to the next line after each row

	#SETUP FUNCTIONS
	def createEmptyMap(self, cols, rows, type):
		empty_map = []
		
		def populateMap(borders, base):
			nonlocal empty_map
			empty_map = np.full((rows, cols), base)
			empty_map[:,: ] = base
			empty_map[0, :] = borders
			empty_map[-1, :] = borders
			empty_map[:, 0] = borders
			empty_map[:, -1] = borders	

		def extraMustSeeSetup():
			not_needed = 0
			stain_size = int(self.info['sizeStains'])
			empty_map[1:stain_size, :] = not_needed  # Top Rows
			empty_map[-stain_size:, :] = not_needed  # Bottom Rows
			empty_map[:, 1:stain_size] = not_needed  # Left Cols
			empty_map[:, -stain_size:] = not_needed  # Right Cols

			vision_size = 3
			outside_walls_size = 1
			base_unit = (stain_size-1)+vision_size
			def setupNotNeededCells(offset):
				base_start = outside_walls_size + (stain_size-1) + vision_size
				base_end = outside_walls_size + 2*(stain_size-1) + vision_size

				# Define a function to compute the slice for different directions
				def getSlices(direction):
					if direction == 'top':
						return (base_start + offset, base_end + offset, base_start + offset, -base_start - offset)
					elif direction == 'left':
						return (base_start + offset, -base_start - offset, base_start + offset, base_end + offset)
					elif direction == 'right':
						return (base_start + offset, -base_start - offset, -base_end - offset, -base_start - offset)
					elif direction == 'bottom':
						return (-base_end - offset, -base_start - offset, base_start + offset, -base_start - offset)

				# Use the function to get slices for different directions and apply to the empty_map
				for direction in ['top', 'left', 'right', 'bottom']:
					row_start, row_end, col_start, col_end = getSlices(direction)
					empty_map[row_start:row_end, col_start:col_end] = not_needed

			i=2
			o=0
			while self.info['nrRows'] > i * base_unit + 2*(stain_size-1) + 2*outside_walls_size:
				setupNotNeededCells(o*base_unit)
				o += 1
				i += 1

		if type == 'memory':
			populateMap('x', '?')
		elif type == 'mustSee':
			populateMap(0, 1)
			if self.is_stain_size_bt_1: extraMustSeeSetup() 
		else:
			raise ValueError("Unsupported 'type'. Use 'memory' or 'mustSee'.")

		return empty_map
	def getDistancePenaltyMap(self):
		distance_penalty_map = np.full((self.info['nrRows'], self.info['nrCols']), 0)
		max_distance = self.info['nrRows'] + self.info['nrCols'] - 2  # maximum possible distance
		for r in range(self.info['nrRows']):
			for c in range(self.info['nrCols']):
				distance_penalty_map[r][c] = abs(r - self.current_cell[0]) + abs(c - self.current_cell[1])
		response = np.round((distance_penalty_map / max_distance) * 100, 2)  # scale to 0-100 range
		return response

	#SETTER FUNCTIONS
	def setStainBt1(self):
		#Activates prediction of stain positions if stains are bigger than 1 cell
		self.is_stain_size_bt_1 = False
		if self.info['sizeStains'] > 1:
			self.is_stain_size_bt_1 = True
	def setDistancePenaltyMultiplier(self, multiplier):
		self.DISTANCE_PENALTY_MULTIPLIER = multiplier
	def getDistancePenaltyMultiplier(self):
		return self.DISTANCE_PENALTY_MULTIPLIER
	
	