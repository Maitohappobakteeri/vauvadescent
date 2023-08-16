class Configuration:
    def __init__(self, args, device="cuda"):
        self.max_length_of_topic = 1000
        self.context_length = 128
        self.short_context_length = 32
        self.sequence_length = 32
        self.device = device
        for arg in vars(args):
            self.__dict__[arg] = getattr(args, arg)
