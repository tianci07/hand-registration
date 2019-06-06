class GeneticOperator:

    def __init__(self, aProbability):
        self.__name__ = "Unspecified genetic operator";
        self.probability = aProbability;
        self.use_count = 0;

    def getName(self):
        return self.__name__;

    def getProbability(self):
        return self.probability;

    def setProbability(self, aProbability):
        self.probability = aProbability;

    def apply(self, anEA):
        raise NotImplementedError("Subclasses should implement this!")

    def __str__(self):
        return "name:\t\"" + self.__name__ + "\"\tprobability:\t" + str(self.probability) + "\"\tuse count:\t" + str(self.use_count);
