#!/usr/bin/env python

import pylab
import numpy, random
import animate

gamma = 0.9

'''
Binary representation of each state. 
Not used at the moment
'''
states = (
	(0,0,0,0),
	(1,0,0,0),
	(0,1,0,0),
	(1,1,0,0),

	(0,0,1,0),
	(1,0,1,0),
	(0,1,1,0),
	(1,1,1,0),

	(0,0,0,1),
	(1,0,0,1),
	(0,1,0,1),
	(1,1,0,1),

	(0,0,1,1),
	(1,0,1,1),
	(0,1,1,1),
	(1,1,1,1)
)

'''
This is the transition matrix delta. At state s and action a,
trans(s,a) = s' = next state
'''
trans = (
	(1 , 3 , 4 , 12) ,
	(0 , 2 , 5 , 13) ,
	(3 , 1 , 6 , 14) ,
	(2 , 0 , 7 , 15) ,
	(5 , 7 , 0 , 8) ,
	(4 , 6 , 1 , 9) ,
	(7 , 5 , 2 , 10) ,
	(6 , 4 , 3 , 11) ,
	(9 , 11 , 12 , 4) ,
	(8 , 10 , 13 , 5) ,
	(11 , 9 , 14 , 6) ,
	(10 , 8 , 15 , 7) ,
	(13 , 15 , 8 , 0) ,
	(12 , 14 , 9 , 1) ,
	(15 , 13 , 10 , 2) ,
	(14 , 12 , 11 , 3)
)

'''
Reward matrix that makes it walk according to our opinion and view of walking
This works better for policy iteration. Is to specific in its reward
'''
rew_walks = (
	(1, 0, 0, 0) , #0
	(0, 1, 0, 0) , #1
	(1, 0, 0, 0) ,#2
	(0, 0, 1, 0) , #3
	(0, 0, 0, 0) , #4
	(0, 0, 0, 0) ,#5
	(0, 0, 0 , 0), #6
	(0, 0, 0 , 0), #7
	(0, 0, 1, 0), #8
	(0, 0, 0, 0), #9
	(0, 0 ,0 , 0), #10
	(0, 1, 0, 0), #11
	(1, 0, 0, 0), #12
	(0, 1, 0 , 0), #13
	(0, 0, 0 , 1), #14
	(0 , 0 , 0, 0), #15
)

'''
This is the reward matrix 16x4. For each state s and action a,
rew(s,a) = r = reward value for doing a in s
'''
rew = (
	(1, 0, 0, 0) , #0
	(0, 1, 0, 0) , #1
	(1, 0, 0, 0) ,#2
	(0, 0, 1, 0) , #3
	(0, 0, 0, 0) , #4
	(0, 0, 0, 0) ,#5
	(0, 0, 0 , 0), #6
	(0, 0, 0 , 0), #7
	(0, 0, 1, 0), #8
	(0, 0, 0, 0), #9
	(0, 0 ,0 , 0), #10
	(0, 1, 0, 0), #11
	(1, 0, 0, 0), #12
	(0, 1, 0 , 0), #13
	(0, 0, 0 , 1), #14
	(0 , 0 , 0, 0), #15
)

policy = [ None for s in trans ]
value = [ 0 for s in trans ]

def argmax(f, args):
	mi = None
	m = -1*pow(10,-10)
	for i in args:
		v = f(i)
		if v > m:
			m = v
			mi = i
	return mi

def policyIteration():
	global gamma, policy, value, trans, rew
	
	for s in range(len(policy)):
		policy[s] = argmax( lambda a: rew[s][a]+gamma*value[trans[s][a]], range(4))
	for s in range(len(value)):
		a = policy[s]
		value[s] = rew[s][a]+gamma*value[trans[s][a]]

for p in range(100):
	policyIteration()
	#print policy, 'total value:', sum(value)


print 'Optimal policy: ', policy
optimal_state_map = [ trans[a][policy[a]] for a in range(len(policy)) ]    
print optimal_state_map

current_state = int(random.random()*len(policy))
path = [current_state]
for i in range(20):
	current_state = optimal_state_map[current_state]
	path += [current_state]

print path
#animate.draw(path)

## Q-LEARNING ##

# Change the reward matrix to include less cred than for temporal learning
class Environment():
	def __init__(self, trans, rew, state = 0):
		self.state = state
		self.trans =  trans
		self.rew = rew

	def go(self, action):
		r = self.rew[self.state][action]
		self.state = self.trans[self.state][action]
		return self.state, r


Q = [ [random.random()*0.5 for a in range(4)] for s in range(16)]

epsilon = 0.01
magic = 0.5
s = random.choice(range(16))
state_sequence = []

def qLearning():
	global s, gamma, epsilon, magic, state_sequence, trans, rew
	environment = Environment(trans, rew, s)

	for step in range(100000):
		action_list = Q[s]
		if random.random() < epsilon:
			a = random.choice(range(len(action_list)))
		else:
			a = action_list.index(max(action_list))
		
		s_next, r = environment.go(a)

		a_next_list = Q[s_next]
		a_next = a_next_list.index(max(a_next_list))

		Q[s][a] = Q[s][a] + magic*(r + gamma*Q[s_next][a_next] - Q[s][a])
		state_sequence += [s]
		s = s_next

qLearning()
l = len(state_sequence)
print state_sequence[l-20:]
animate.draw(state_sequence[l-20:])