import torch

'''
Just a wrapper of LSTM states: a list of (h, c) tuples
'''
class LSTMState(object):
    
    def __init__(self, states):
        self.states = states
    
    def clone(self):
        new_states = [(s[0].clone(), s[1].clone()) for s in self.states]
        return LSTMState(new_states)

    def dim(self):
        return self.states[0][0].dim()
    
    def unsqueeze(self, dim):
        new_states = [(s[0].unsqueeze(dim), s[1].unsqueeze(dim)) for s in self.states]
        return LSTMState(new_states)
        
    def view(self, *sizes):
        new_states = [(s[0].view(*sizes), s[1].view(*sizes)) for s in self.states]
        return LSTMState(new_states)
    
    def expand(self, *sizes):
        new_states = [(s[0].expand(*sizes), s[1].expand(*sizes)) for s in self.states]
        return LSTMState(new_states)
        
    def contiguous(self):
        states = list()
        for s in self.states:
            h = s[0].contiguous()
            c = s[1].contiguous()
            states.append((h, c))
        return LSTMState(states)
        
    def __len__(self):
        return len(self.states)
        
    def detach_(self):
        for s in self.states:
            s[0].detach_()
            s[1].detach_()

    def size(self):
        return self.states[0][0].size()
        
    def cat(self, other, dim):
        states = list()
        for s1, s2 in zip(self.states, other.states):
            h = torch.cat([s1[0], s2[0]], dim)
            c = torch.cat([s1[1], s2[1]], dim)
            states.append((h, c))
        return LSTMState(states)
        
    def __mul__(self, other):
        states = list()
        for s in self.states:
            h = s[0] * other
            c = s[1] * other
            states.append((h, c))
        return LSTMState(states)
        
    def __add__(self, other):
        states = list()
        for s1, s2 in zip(self.states, other.states):
            h = s1[0] + s2[0]
            c = s1[1] + s2[1]
            states.append((h, c))
        return LSTMState(states)
    
    def get_output(self):
        return self.states[-1][0]
    
    def get(self, ind):
        return self.states[ind]
    
    def __getitem__(self, key):
        states = list()
        for s in self.states:
            h = s[0][key]
            c = s[1][key]
            states.append((h, c))
        return LSTMState(states)
    
    def __setitem__(self, key, item):
        for s1, s2 in zip(self.states, item.states):
            s1[0][key] = s2[0]
            s1[1][key] = s2[1]
          
