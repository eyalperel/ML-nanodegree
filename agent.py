import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.actionOptions = (None, 'forward', 'left', 'right')
        self.Qtable = {} # the dictionary that will hold the Q values
        self.gamma = 0.25
        self.alpha = 0.5 # learning rate
        self.prevStateActionPair = () # for storing last state and action
        self.currentEpsilon = 1 # for implementing greedy-epsilon
        self.epsCutoff = 0.25 # we will do purely random exploration for the first epsCutoff trials and then decay epsilon rapidly. This is a free parameter
        self.currentTrial = 0
        # for testing and analysis purposes
        self.totalTrials = 100
        self.totalGoodTrials = 0
        self.nonRandomFoodTrials = 0
        self.successRatio = 0.0
        self.numOfPenalties = 0
            
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.prevStateActionPair = ()
        self.currentTrial = self.currentTrial + 1
        if self.currentTrial > int(self.epsCutoff * self.totalTrials):
            self.currentEpsilon = (1.0 / float(self.currentTrial)) * self.currentEpsilon
        self.numOfPenalties = 0
        
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint) #seems reasnible 
        
        # TODO: Select action according to your policy
        decision = random.random()
        if (decision <= self.currentEpsilon):
            action = random.choice(self.actionOptions)
        else:
            action = self.selectingActionBasedOnQvalue()
        
        self.prevStateActionPair = (self.state, action)
        # Execute action and get reward
        reward = self.env.act(self, action)
        if reward < 0 and self.currentTrial > int(self.epsCutoff * self.totalTrials): #a penalty during the nonrandom period
            self.numOfPenalties += 1
        
        if reward >= 9 and deadline > 0:
            self.totalGoodTrials += 1
            if self.currentTrial > int(self.epsCutoff * self.totalTrials):
                self.nonRandomFoodTrials += 1
                self.successRatio = float(self.nonRandomFoodTrials) / (self.totalTrials - int(self.epsCutoff * self.totalTrials))
                if self.currentTrial > 79:
                    print "num of penalties is: {}".format(self.numOfPenalties)
                    print "current success rate is: {}%".format(self.successRatio * 100)
                    print "current deadline when meeting the destination is: {}".format(deadline)
            
        
        # TODO: Learn policy based on state, action, reward
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        #print "Current parameter configuration is: epsilonCutoff in trials = {}, alpha = {}, gamma = {}".format(int(self.epsCutoff * self.totalTrials), self.alpha, self.gamma)
        #print "Current trial is: {}".format(self.currentTrial)
        #print "Current number of penalties: {}".format(self.numOfPenalties)
        #print "current number of total good trials is: {}".format(self.totalGoodTrials)
        #print "current number of non-random good trials is: {}".format(self.nonRandomFoodTrials)
        
        #now we need to find the action that gives the maximum Q-value for the new state we found ourselves in.
        bestQvalue = self.maxQvalueForActionsOverNextState() # this will store the best action given the new state
        self.Qtable[self.prevStateActionPair] = (1 - self.alpha) * self.Qtable.get(self.prevStateActionPair, 0) + self.alpha * (reward + self.gamma * bestQvalue)
     
     # this is a helper function for looping over all possible actions (there are four) for the current state (which is what we ended up in after taking an action)
     # We examine the Q value for each action and return the maximum Q value associated with some action of the possible four. This will be used for updating the Q table
    def maxQvalueForActionsOverNextState(self):
        # we need to construct the new state we are in (this is s')
        currentEnv = self.env.sense(self)
        currentNextWaypoint = self.planner.next_waypoint()
        newState = (currentEnv['light'], currentEnv['oncoming'], currentNextWaypoint)
        
        # we search for the max Q value by going through all possilbe actions from s'
        return self.findingMaxActionQValuePair(newState, 'Qvalue')
    
    def selectingActionBasedOnQvalue(self):
        return self.findingMaxActionQValuePair(self.state, 'Action')
        
    def findingMaxActionQValuePair(self, state, returnMode):
        maxAction = self.actionOptions[0]
        maxQvalue = self.Qtable.get((state,self.actionOptions[0]), 0)
        for action in self.actionOptions:
            currQvalue = self.Qtable.get((state, action) ,0)
            if currQvalue >= maxQvalue:
                maxQvalue = currQvalue
                maxAction = action
        if returnMode == 'Qvalue':
            return maxQvalue
        else:
            return maxAction
            
            
        
    
        
    
                
            
            
        
        
            

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline = True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay = 0.5, display = False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials = 100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
