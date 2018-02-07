class Table(object):
    
    def __init__(self):
        self.entry_dict = dict()
        self.visible_entry_list = list() # whether to print it
        self.metadata = dict() 
    
    def __repr__(self):
        hline = '#' * 30 + '\n'
        out = ''
        if self.metadata:
            for name in sorted(self.metadata):
                out += name.upper() + ': ' + str(self.metadata[name]) + '\n'
            out += '-' * 30 + '\n'
        for entry_name in sorted(self.visible_entry_list):
            out += hline + entry_name.upper() + ':\n' + str(self.entry_dict[entry_name]) + '\n'
        return out
    
    def add_entry(self, name, value, visible=True):
        assert name not in self.entry_dict
        self.entry_dict[name] = value
        if visible:
            self.visible_entry_list.append(name)
    
    def add_metadata(self, name, value):
        self.metadata[name] = value
    
    def __getitem__(self, name):
        if name in self.entry_dict:
            return self.entry_dict[name]
        else:
            return self.metadata[name]
    
    def __setitem__(self, key, value):
        assert key in self.entry_dict
        self.entry_dict[key] = value
        
    def __getattr__(self, name):
        return self.__getitem__(name)
    
    def __len__(self):
        return len(self.entry_dict[self.entry_dict.keys()[0]])