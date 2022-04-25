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
	pixel = map[new_pos[0], new_pos[1], :]
	if (pixel[0] == 0.0): #we hit the path, TODO: fix this with something better
		return -2, map, new_pos
	elif (pixel[0] != 1.): #we hit something else, like an obstacle
		return -5, map, curr
	else: #mark path, small negative reward for moving
		map[new_pos[0],new_pos[1], 0:3] = 0.0
		return -0.1, map, new_pos

