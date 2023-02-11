class Configuration:
    def __init__(self, args):
        self.max_length_of_topic = 200
        self.context_length = 5
        self.sequence_length = 1
        for arg in vars(args):
            self.__dict__[arg] = getattr(args, arg)
