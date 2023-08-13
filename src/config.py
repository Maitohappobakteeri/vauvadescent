class Configuration:
    def __init__(self, args, device="cuda"):
        self.max_length_of_topic = 500
        self.context_length = 128
        self.short_context_length = 32
        self.sequence_length = 4
        self.device = device
        for arg in vars(args):
            self.__dict__[arg] = getattr(args, arg)
