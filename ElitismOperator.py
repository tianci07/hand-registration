import GeneticOperator

class ElitismOperator(GeneticOperator.GeneticOperator):

    # Contructor
    def __init__(self, aProbability):
        # Apply the constructor of the abstract class
        super().__init__(aProbability);

        # Set the name of the new operator
        self.__name__ = "Elitism operator";

    def apply(self, anEA):
        raise NotImplementedError("This class does not implement this!")
