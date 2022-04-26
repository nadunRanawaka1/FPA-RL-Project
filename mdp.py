#move down, move right, move up, move left
actions = [(1,0), (0,1), (-1,0), (0,-1)]



# given an action and current state, this will return the reward for the action,
# the new map and the next state
def transition(map, curr, goal, action): 
	move = actions[action]
	new_pos = (curr[0] + move[0], curr[1] + move[1])

	if (new_pos == goal): #we are at goal
		map[goal,0:3] = 0.0
		return 10, map, new_pos

	elif (new_pos[0] < 0 or new_pos[1] < 0 or new_pos[0] > 599 or new_pos[1] > 599): #we went off the map
		return -2, map, curr

	pixel = map[new_pos[0], new_pos[1], :]

	mhat_dist_old = abs(goal[0] - curr[0]) + abs(goal[1] - curr[1]) #calculating the manhattan distance from the old position to goal
	mhat_dist_new = abs(goal[0] - new_pos[0]) + abs(goal[1] - new_pos[1]) #same for new position

	euc_dist_old = ((goal[0] - curr[0])**2 + (goal[1] - curr[1])**2)**0.5 #calculating euclidean distance between goal and old pos
	euc_dist_new = ((goal[0] - new_pos[0])**2 + (goal[1] - new_pos[1])**2)**0.5

	if (pixel[0] == 0.0): #we hit the path, TODO: fix this with something better
		return -2, map, new_pos

	elif (pixel[0] != 1.): #we hit something else, like an obstacle
		return -5, map, curr

	elif (mhat_dist_new < mhat_dist_old): #the new position is closer to the goal
		map[new_pos[0],new_pos[1], 0:3] = 0.0
		return -0.05, map, new_pos

	else: #mark path, small negative reward for moving
		map[new_pos[0],new_pos[1], 0:3] = 0.0
		return -0.1, map, new_pos

