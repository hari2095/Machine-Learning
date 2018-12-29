#!/usr/bin/env python
import sys
import numpy as np
import math

class Environment:
    reward = -1
    def is_invalid_cell(self,cell):
        if ("*" == cell):
            return True
        return False

    def get_new_state(self,state,action):
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

    def is_valid_move(self,state,action,maze):
        row,column = self.get_new_state(state,action)
        #print ("old_state: ",state,"action: ",action,"new_state: ",(row,column))
        #check bounds first
        if (column >= 0 and column < self.breadth and row < self.length and row >= 0):
            #within bounds, now check for obstacle
            if self.is_invalid_cell(self.maze[row][column]):
                return False
            #within bounds and obstacle-less cell
            return True

        #Not within bounds
        return False

    def __init__(self,maze_input_file_name):
        maze_input_file = open(maze_input_file_name,"r")
        maze = maze_input_file.read().splitlines()
        maze_input_file.close()
        #Instantiate a copy of the maze
        self.maze = maze
        #Initialize current state of agent
        self.length  = len(self.maze)
        self.breadth = len(self.maze[0])
        
        self.actions = ["0","1","2","3"]

        self.goal_state = (-float('inf'),-float('inf'))
        self.start_state = (-float('inf'),-float('inf'))

        for i in range(self.length):
            for j in range(self.breadth):
                if "G" == maze[i][j]:
                    self.goal_state = (i,j)
                elif "S" == maze[i][j]:
                    self.start_state = (i,j)
                    self.current_state = self.start_state
    
    def step(self,action):
        new_state = self.current_state
        is_terminal = 0
        if self.is_valid_move(self.current_state,action,self.maze):
            new_state = self.get_new_state(self.current_state,action)
        self.current_state = new_state
        if self.current_state == self.goal_state:
            is_terminal = 1
        return new_state,self.reward,is_terminal

    def reset(self):
        self.current_state = self.start_state
        return self.current_state
                 
if __name__ == "__main__":
    maze_input_file_name = sys.argv[1]
    output_seq_file_name = sys.argv[2]
    action_seq_file_name = sys.argv[3]

    e = Environment(maze_input_file_name)
    action_seq_file = open(action_seq_file_name,"r")
    output_seq_file = open(output_seq_file_name,"w+")
    action_seq_line = action_seq_file.read().rstrip()
    action_seq = action_seq_line.split(" ")
    for action in action_seq:
        output = e.step(action)
        new_state,reward,is_terminal = output
        output_seq_file.write(str(new_state[0]) + " " + str(new_state[1]) + " " + str(reward) + " " + str(is_terminal) + "\n")
