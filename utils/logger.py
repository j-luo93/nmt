import sys

'''
Class that outputs to terminal as well as a log file.
'''
class Logger(object):

    def __init__(self, log_file):
        self.terminal = sys.stderr
        self.log = open(log_file, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
