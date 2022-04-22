# start = (60, 170)
# goal =  (490, 440)

# A state is represented by a pixel in an image
# A state will have the following features:
# 1) x-position of pixel
# 2) y-position of pixel
# 3) color of pixel encoded as grayscale intensity
# 4) euclidean distance from goal
# 5) euclidean distance from start
# TODO : extract more features from pixel
# Idea: take area around pixel, convolve and pass into NN
# - pass entire image into CNN, mark pixel of interest with different color, path should also be marked
# - pass entire image into CNN and also pass (x,y) of pixel as parameter
# - maybe try to search around the pixel for an obstacle, if the obstacle is within some distance,
# pass that in as an input

def extract_features(img_arr, pixel, start, goal):
	features = []
	features.append(pixel[0]) #append x
	features.append(pixel[1]) #append y

	pixel_vals = img_arr[pixel[0], pixel[1], :] #get values (RGB) of pixel
	gray_val = 0.299*pixel_vals[0] + 0.587*pixel_vals[1] + 0.114*pixel_vals[2] #convert RGB to grayscale 

	features.append(gray_val)

	d_goal = ((pixel[0] - goal[0])**2 + (pixel[1] - goal[1])**2)**0.5 #calculate distance to goal
	d_start = ((pixel[0] - start[0])**2 + (pixel[1] - start[1])**2)**0.5

	features.append(d_goal)
	features.append(d_start)

	return features