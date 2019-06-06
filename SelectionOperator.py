class SelectionOperator:

    def __init__(self, name = "Unspecified selection operator"):
        self.name = name;


    def getName(self):
        return self.name;

    def select(self, anIndividualSet):
        return self.selectGood(anIndividualSet);

    def selectGood(self, anIndividualSet):
        return self.__select__(anIndividualSet, True);

    def selectBad(self, anIndividualSet):
        return self.__select__(anIndividualSet, False);

    def preProcess(self, anIndividualSet):
        raise NotImplementedError("Subclasses should implement this!")

    def __select__(self, anIndividualSet, aFlag): # aFlag == True for selecting good individuals,
                                                  # aFlag == False for selecting bad individuals,
        raise NotImplementedError("Subclasses should implement this!")

    def __str__(self):
        return "name:\t\"" + self.name + "\"";
