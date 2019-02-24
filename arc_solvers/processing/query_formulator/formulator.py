class Formulator:

    def formulate(self, sample, **kwargs):
        return sample

    @staticmethod
    def does_arg_exists(arg, **kwargs):
        if arg not in kwargs:
            raise ValueError(f"{arg} not in {kwargs.keys()}")
