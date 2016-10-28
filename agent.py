import numpy as np 
import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)# sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'black'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        # TODO: Initialize any additional variables here
        self.state = None
        self.action = None
        self.alpha = 0.50
        self.gamma = 0.05
        self.epsilon = .01
        self.q_table = {}
        self.actions = [None, 'forward', 'left', 'right']
        self.trips = 0 
        softmax_probabilities = {} 
        self.previous_state = None
        self.last_action = None 
        self.last_reward = None
        
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        # 
        self.trips += 1
        if self.trips >= 10:
            self.epsilon == .10
            if self.trips >= 20: 
                self.epsilon == 0.00


    def get_action(self, state,softmax_probabilities):
        #limited randomness on action choice per decreasing self.epsilon--avoid local min/max
        if random.random()< self.epsilon:
            action = np.random.choice(self.actions, p = softmax_probabilities) 
        else: 
            action = self.actions[np.argmax(softmax_probabilities)]
        return action 

    def update(self, t):
        self.next_waypoint = self.planner.next_waypoint() #from route planner,also displayed, simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
    
        #TODO: Update state
        state = (inputs['light'], inputs['oncoming'], inputs['right'],inputs['left'], self.next_waypoint)
        self.state = state
        
        # Initialization of q_table values to .97
        for action in self.actions:
            if(state,action) not in self.q_table:
                self.q_table[(state,action)] = .75
            
        #I used softmax function of q in q_table--so that 0-1 values for q-score  
        softmax_probabilities =[self.q_table[(state,None)],self.q_table[(state,'forward')], self.q_table[(state,'left')], self.q_table[(state, 'right')]] 
        # softmax = (e ^ x) / sum(e ^ x) 
        softmax_probabilities = np.exp(softmax_probabilities)/ np.sum(np.exp(softmax_probabilities), axis = 0)
        #Execute the action get reward
        action = self.get_action(self.state, softmax_probabilities)
        # Get a reward for the action
        reward = self.env.act(self, action)
        

# TODO: Learn policy based on state, action, reward  
        if self.previous_state == None: 
            self.q_table[(state, action)] = ((1-self.alpha) * reward) + (self.alpha * self.q_table[(state, action)])
        else: 
            current_q = self.q_table[(self.previous_state, self.last_action)]
            future_q = self.q_table[(state,action)]
            current_q = (1 - self.alpha) * current_q + self.alpha * (self.last_reward + self.gamma * future_q)
            self.q_table[(self.previous_state, self.last_action)] = current_q
        
        # use previous_state to store the last state so that I can lag between current state and next 
        self.previous_state = self.state
        self.last_action = action 
        self.last_reward = reward
        
def run():
    """Run the agent for a finite number of trials."""
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.00001, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()