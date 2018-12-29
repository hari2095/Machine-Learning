#!/usr/bin/env python
import sys
import numpy as np
import math
import time

####################
#DEBUG ZONE
####################
down = u'\u25bc'
up = u'\u25b2'
left = u'\u25c0'
right = u'\u25b6'

directions = {}
directions["0"] = left
directions["1"] = up
directions["2"] = right
directions["3"] = down
####################
#DEBUG ZONE
####################

def is_invalid_cell(cell):
    if (math.isnan(cell)):
        return True
    return False

def get_new_state(state,action):
    row,column = state
    #go left
    if ("0" == action):
        column -= 1
    #go up
    if ("1" == action):
        row    -= 1
    #go right
    if ("2" == action):
        column += 1
    #go down
    if ("3" == action):
        row    += 1
    new_state = (row,column)
    return new_state

def is_valid_move(state,action,values):
    row,column = get_new_state(state,action)
    #check bounds first
    if (column >= 0 and column < breadth and row < length and row >= 0):
        #within bounds, now check for obstacle
        if is_invalid_cell(values[row][column]):
            return False
        
        #within bounds and obstacle-less cell
        return True

    #Not within bounds
    return False

start_time = time.time()

maze_input_file_name = sys.argv[1]
value_file_name = sys.argv[2]
q_value_file_name = sys.argv[3]
policy_file_name = sys.argv[4]
num_epoch = int(sys.argv[5])
discount_factor = float(sys.argv[6])

maze_input_file = open(maze_input_file_name,"r")
value_file      = open(value_file_name,"w+")
q_value_file    = open(q_value_file_name,"w+")
policy_file     = open(policy_file_name,"w+")

maze = maze_input_file.read().splitlines()
length  = len(maze)
breadth = len(maze[0])

actions = ["0","1","2","3"]

#The table for values
values  = np.zeros((length,breadth),dtype=float)

goal_state = (-float('inf'),-float('inf'))

for i in range(length):
    for j in range(breadth):
        if "*" == maze[i][j]:
            values[i][j] = float('NaN')
        elif "G" == maze[i][j]:
            goal_state = (i,j)
#print (values)

converged = False
epoch = 0
reward = -1.0
Q = {}

#Last minute hack to get things in the right format for autograder
for i in range(length):
    for j in range(breadth):
        state = (i,j)
        Q[state] = {}
        for action in actions:
            if "*" == maze[i][j]:
                Q[state][action] = float('NaN')
            if "G" == maze[i][j]:
                Q[state][action] = 0.0

while not converged:
    #val_diff = False
    for i in range(length):
        for j in range(breadth):
            if (is_invalid_cell(values[i][j]) or (i,j) == goal_state):
                continue
            state = (i,j)
            for action in actions:
                #initial reward, irrespective of action taken
                Q[state][action] = reward
                #deferred reward depending on the case
                if (is_valid_move(state,action,values)):
                    row,column = get_new_state(state,action)
                    Q[state][action] += discount_factor*values[row][column]
                else:
                    Q[state][action] += discount_factor*values[i][j]
            old_val = values[i][j]
            values[i][j] = max(Q[state].values())
            #print old_val,values[i][j]
            #if ((abs(values[i][j]) - abs(old_val)) > 0.001):
                #val_diff = True
    #if not val_diff:
        #converged = True
    print values
    print "\n"
    epoch += 1
    if epoch == num_epoch + 1:
        converged = True

end_time = time.time()
total_time = end_time - start_time
#print "Time taken to solve the maze: ",total_time
#print "Converged in ",epoch," steps"

#print "MAZE:"
#for line in maze:
#    print line

#print "\nSOLUTION:"
for i in range(length):
    for j in range(breadth):
        state = (i,j)
        if ("*" == maze[i][j]):
            #print "*",
            continue
        max_val = -float('inf')
        policy_action = "-1"
        for action in actions:
            q_value_file.write(str(state[0]) + " " + str(state[1]) + " " + str(action) + " " + str(Q[state][action]) + "\n")
            if Q[state][action] > max_val:
                max_val = Q[state][action]
                policy_action = action
        policy_file.write(str(state[0]) + " " + str(state[1]) + " " + str(float(policy_action)) + "\n")
        value_file.write(str(state[0]) + " " + str(state[1]) + " " + str(max_val) + "\n")
        #if ("G" == maze[i][j]):
            #print "G",
        #else:
            #print directions[policy_action],
    #print ""

