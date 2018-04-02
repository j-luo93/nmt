'''
A wrapper of a list of variables (can include some non-variables). Used for forward call.
It inherits list class, so should be okay for forward function. But to make it easier, it 
also has dictionary-like behavior.
'''

# TODO need another name
# NOTE don't override __getitem__. Has to be used by forward call.
class VariableDict(list):
    
    # takes in a list of duples (name, attr), which will be recorded for __getattr__. 
    # When name is absent, attr is not recorded but still passed to list construction.
    def __init__(self, list_of_duples): 
        assert isinstance(list_of_duples, list)
        lst = list()
        self.records = dict()
        for duple in list_of_duples:
            if isinstance(duple, tuple):
                assert len(duple) == 2
                name, attr = duple
                lst.append(attr)
                assert name not in self.records, 'name duplicate %s' %name
                self.records[name] = attr
            else:
                lst.append(duple)
        super(VariableDict, self).__init__(lst)
        
    def __getattr__(self, name):
        if name in self.records:
            return self.records[name]
        else:
            return None
    
    def append(self, item):
        if isinstance(item, tuple):
            name, attr = item
            assert name not in self.records
            self.records[name] = attr
        super(VariableDict, self).append(item)


