#!/usr/bin/env python
import sys
import numpy as np
import math
from environment import Environment
import random
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

def select_action(current_state):
    rand_num = random.random()
    #print ("rand_num: ",rand_num,"epsilon: ",epsilon)
    action = "" 
    if rand_num < epsilon:
        action = actions[np.random.randint(len(actions))]
        #print ("random_act_chosen: ",action)
    else:
        values = Q[current_state]
        #print ("max_action_candidates: ",current_state," ",values)
        max_val = -float('inf')
        for act in actions:
            if Q[current_state][act] > max_val:
                max_val = Q[current_state][act]
                action = act
        #print ("max_act_chosen: ",action)
    #print repr(action)
    return action 
        
start_time = time.time()        

maze_input_file_name = sys.argv[1]
value_file_name = sys.argv[2]
q_value_file_name = sys.argv[3]
policy_file_name = sys.argv[4]
num_episodes = int(sys.argv[5])
max_episode_length = int(sys.argv[6])
learning_rate = float(sys.argv[7])
discount_factor = float(sys.argv[8])
epsilon = float(sys.argv[9])

actions = ["0","1","2","3"]

#Q table, initially, no states are known
Q = {}
e = Environment(maze_input_file_name)
start_state = e.reset()
maze_input_file = open(maze_input_file_name,"r")
value_file      = open(value_file_name,"w+")
q_value_file    = open(q_value_file_name,"w+")
policy_file     = open(policy_file_name,"w+")
maze = maze_input_file.read().splitlines()
maze_input_file.close()
length  = len(maze)
breadth = len(maze[0])

#It's a small world, initialize everything
for i in range(length):
    for j in range(breadth):
        state = (i,j)
        Q[state] = {}
        for action in actions:
            if "*" == maze[i][j]:
                Q[state][action] = float('NaN')
            else:
                Q[state][action] = 0.0
current_state = start_state        

converged = False
epoch = 0
episode_length = 0
ep_lengths = []
while not converged:
    action  = select_action(current_state)
    #if action != "0":
        #print action
    next_state,reward,is_terminal = e.step(action)
    #print (current_state,action,next_state," ",reward," ",is_terminal)
    temp = ((1-learning_rate)*Q[current_state][action])
    max_val = -float('inf')
    for action_prime in actions:
        if Q[next_state][action_prime] > max_val:
            max_val = Q[next_state][action_prime]
    if 1 == is_terminal:
        max_val = 0
    temp += (learning_rate*(reward + discount_factor*(max_val)))
    Q[current_state][action] = temp
    current_state = next_state
    episode_length += 1
    if is_terminal or episode_length == max_episode_length:
        #print (is_terminal,episode_length,"resetting.....\n")
        current_state = e.reset()
        ep_lengths.append(episode_length)
        episode_length = 0 
        epoch += 1
    #print "\n"
    if epoch == num_episodes:
        converged = True

avg_length = np.average(np.array(ep_lengths))
end_time = time.time()
total_time = end_time - start_time
#print "Time taken to solve the maze: ",total_time
#print "Average steps per episode: ", avg_length

print "MAZE:"
for line in maze:
    print line

print "\nSOLUTION:"

for i in range(length):
    for j in range(breadth):
        state = (i,j)
        if ("*" == maze[i][j]):
            print "*",
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
        if ("G" == maze[i][j]):
            print "G",
        else:
            print directions[policy_action],
    print ""            
