import GeneticOperator
import Individual as IND


class NewBloodOperator(GeneticOperator.GeneticOperator):

    # Contructor
    def __init__(self, aProbability):
        # Apply the constructor of the abstract class
        super().__init__(aProbability);

        # Set the name of the new operator
        self.__name__ = "New blood operator";

    def apply(self, anEA):

        self.use_count += 1;

        # Return a new individual whose genes are randomly
        # generated using a uniform distribution
        return (IND.Individual(
            anEA.objective_function.number_of_dimensions,
            anEA.objective_function.boundary_set,
            anEA.objective_function))
