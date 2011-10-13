#!/usr/bin/env python

import pylab
import numpy

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
This is the reward matrix 16x4. For each state s and action a,
rew(s,a) = r = reward value for doing a in s
'''
rew = (
	(0 , 1 , 0 , 0) ,
	(0 , 0, 0 , 1) ,
	(0 , 0 , 1 , 0) ,
	(0 , 0 , 1 , 0) ,
	(0 , 0 , 1 , 0) ,
	(0 , 0 , 1 , 0) ,
	(0 , 1 , 0 , 0),
	(0 , 1 , 0 , 0),
	(0 , 1 , 0 , 0),
	(1 , 0 , 0 , 0),
	(0 , 0 , -1 , 0),
	(0 , 0 , -1 , 0),
	(0 , 1 , 0 , 0),
	(1, 0 , 0 , 0),
	(0 , 1, 0 , 0),
	(0 , 0 , 1, 0),
)

policy = [ None for s in trans ]
value = [ 0 for s in trans ]

def argmax(f, args):
	mi = None
	m = -1*pow(10,-10)
	for i in args:
		v = f(i)
		if v > m:
			#print '\tf(',i,') = ', v
			m = v
			mi = i
	return mi

def policyIteration():
	#global gamma, policy, value, trans, rew

	for p in range(100):
		for s in range(len(policy)):
			policy[s] = argmax( lambda a: rew[s][a]+gamma*value[trans[s][a]], range(4))
			#print 'policy[s]', s, policy[s]
		for s in range(len(value)):
			a = policy[s]
			#print 'policy[',s,'] = ', a
			value[s] = rew[s][a]+gamma*value[trans[s][a]]


policyIteration()

animate.draw(policy)
	