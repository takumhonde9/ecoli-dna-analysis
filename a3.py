#!/usr/bin/python3

import sys
import random
import math
import numpy as np

#####################################################
#####################################################
# Please enter the number of hours you spent on this
# assignment here
num_hours_i_spent_on_this_assignment = 10
#####################################################
#####################################################

#####################################################
#####################################################
# <No comment.>
#####################################################
#####################################################



# Outputs a random integer, according to a multinomial
# distribution specified by probs.
def rand_multinomial(probs):
    # Make sure probs sum to 1
    assert(abs(sum(probs) - 1.0) < 1e-5)
    rand = random.random()
    for index, prob in enumerate(probs):
        if rand < prob:
            return index
        else:
            rand -= prob
    return 0

# Outputs a random key, according to a (key,prob)
# iterator. For a probability dictionary
# d = {"A": 0.9, "C": 0.1}
# call using rand_multinomial_iter(d.items())
def rand_multinomial_iter(iterator):
    rand = random.random()
    for key, prob in iterator:
        if rand < prob:
            return key
        else:
            rand -= prob
    return 0

class HMM():

    def __init__(self):
        self.num_states = 2
        self.prior = [0.5, 0.5]
        self.transition = [[0.999, 0.001], [0.01, 0.99]]
        self.emission = [{"A": 0.291, "T": 0.291, "C": 0.209, "G": 0.209},
                         {"A": 0.169, "T": 0.169, "C": 0.331, "G": 0.331}]


    # Generates a sequence of states and characters from
    # the HMM model.
    # - length: Length of output sequence
    def sample(self, length):
        sequence = []
        states = []
        rand = random.random()
        cur_state = rand_multinomial(self.prior)
        for i in range(length):
            states.append(cur_state)
            char = rand_multinomial_iter(self.emission[cur_state].items())
            sequence.append(char)
            cur_state = rand_multinomial(self.transition[cur_state])
        return sequence, states

    # Generates a emission sequence given a sequence of states
    def generate_sequence(self, states):
        sequence = []
        for state in states:
            char = rand_multinomial_iter(self.emission[state].items())
            sequence.append(char)
        return sequence

    # Computes the (natural) log probability of sequence given a sequence of states.
    def logprob(self, sequence, states):
        ###########################################
        # Start your code
        T=len(sequence) # number of time steps
        prob=math.log(self.prior[states[0]]) # initial prob
        for t in range (0,T):
            if t>0:
                prob+=math.log(self.transition[states[t]][states[t-1]]) # adding log(P(t|t-1))
            prob+=math.log(self.emission[states[t]][sequence[t]])# adding log(P(evidence[t]|state[t]))
        return prob 
        # End your code
        ##########################################
        
    # - sequence: String with characters [A,C,T,G]
    # - m: matrix of probabilities e.g. [[], [], [],...]
    # - prev: matrix of probabilities e.g. [[], [], [],...] 
    # return: a list of state indices, e.g. [0,0,0,1,1,0,0,...]
    def path(self, m, prev):
        T=len(m)
        sv=[] # state variables
        _,state=max((m[T-1][y], y) for y in range(0,self.num_states)) # find state at end
        sv.append(int(prev[T-1][state]))
        for t in reversed(range(1,T)):
            state=int(prev[t][state]) # update state
            sv.append(int(prev[t][state]))
        sv.reverse() # reverse to get 0:T, instead of T:0
        return sv

    # initializes the fist row of the probability matrix by multiplying the emission prob with the prior prob
    # - m: matrix of probabilities e.g. [[], [], [],...] 
    # returns the matrix m with initialized first row
    def initialize(self, m):
    	D=self.num_states
    	for d in range(0,D):
    		m[0][d]=math.log(self.prior[d])+math.log(self.emission[d][sequence[0]])
    	
    # Outputs the most likely sequence of states given an emission sequence
    # - sequence: String with characters [A,C,T,G]
    # return: list of state indices, e.g. [0,0,0,1,1,0,0,...]
    def viterbi(self, sequence):
        ###########################################
        # Start your code
        T=len(sequence) # number of time steps
        D=self.num_states # dimension of HMM
        m=np.full((T, D), -np.inf) # probability matrix 
        self.initialize(m) # initialize first row of probablity matrix
        prev=np.zeros((T,D)) # matrix to find most likely state 
        for t in range (1,T):
            e_t=sequence[t]
            for i in range (0,D):
                for j in range (0,D):
                    prob=m[t-1][j]+math.log(self.transition[j][i])+math.log(self.emission[i][e_t])
                    if prob>m[t][i]:
                        m[t][i] = prob
                        prev[t][i]=j

        return self.path(m, prev)
        # End your code
        ###########################################

def read_sequence(filename):
    with open(filename, "r") as f:
        return f.read().strip()

def write_sequence(filename, sequence):
    with open(filename, "w") as f:
        f.write("".join(sequence))

def write_output(filename, logprob, states):
    with open(filename, "w") as f:
        f.write(str(logprob))
        
        f.write("\n")
        for state in range(2):
            f.write(str(states.count(state)))
            f.write("\n")
        f.write("".join(map(str, states)))
        
        f.write("\n")

hmm = HMM()
file = sys.argv[1]
sequence = read_sequence(file)
viterbi = hmm.viterbi(sequence)
logprob = hmm.logprob(sequence, viterbi)
name = "my_"+file[:-4]+'_output.txt'
write_output(name, logprob, viterbi)
