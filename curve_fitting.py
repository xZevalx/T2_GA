from genetic_algorithm import IOrganism, GeneticAlg
import numpy as np
import sys
import ast


class Coefficient(IOrganism):

    def __init__(self, *args, **kwargs):
        """
        Should receive arguments
         - domain_limits, representation or
         - domain_limits, size
        """
        self.lower_limit, self.upper_limit = kwargs.pop('domain_limits')
        super().__init__(*args, **kwargs)

    def spontaneous_generation(self, size):
        d = self.upper_limit - self.lower_limit
        self.representation = np.random.random(size) * d + self.lower_limit

    def mutate(self, *args, **kwargs):
        # Choose number of mutations (always less than half)
        n_muts = int(np.random.random() * self.size / 2)
        # Choose random genes to mutate
        i_genes = np.array(np.random.random(n_muts) * self.size, dtype=np.int)
        for i_gene in i_genes:
            self.representation[i_gene] = np.random.random() * (self.upper_limit - self.lower_limit) + \
                                          self.lower_limit

    def breed(self, organism):
        """ Breeding by averaging coefficients """
        new_repr = (self.representation + organism.representation) / 2
        return Coefficient(domain_limits=(self.upper_limit, self.lower_limit), representation=new_repr)


def coefficient_fitness(coef, data, model):
    """
    Calculates mean squared error.

    :param data: (x,y) tuples to model
    :param coef: Coefficient instance
    :param model: function f to evaluate y -> f(c1, ..., cn | x), so its signature should be model(x, repr)
    :return: float, mse
    """
    mse = 0
    for x, y in data:
        mse += np.power((y - model(x, coef.representation)), 2)
    mse /= len(data)
    return mse


def coefficient_selection(coeffs, fitnesses, return_best=False):
    """ Select the first half coefficients with less MSE.  """
    sort_by_fitness = np.argsort(fitnesses)
    sorted_coeffs = coeffs[sort_by_fitness]
    if return_best:
        return sorted_coeffs[:int(len(coeffs) / 2)], sorted_coeffs[0]
    else:
        return sorted_coeffs[:int(len(coeffs) / 2)]


if __name__ == '__main__':
    """
    Arguments
    
    [1]: Experiment type: synthetic or real [s, r]
    [2]: Polynomial grade:  int.
    [3]: Coefficients domain for GA: tuple [val1, val2] where val1 < val2
    [4]: Population size: int, default 100
    [5]: Generations: int, default 50
    [6]: Mutation rate, float, default 0.1
    """
    import matplotlib.pyplot as plt

    default_conf = ['s', '5', '(-1000, 1000)', '100', '50', '.1']

    print(sys.argv)

    if 1 < len(sys.argv) <= 7:
        args = sys.argv[1:]
        default_conf[:len(args)] = args

    else:
        print('Using default configuration\n')

    args = default_conf
    args[1:] = [ast.literal_eval(elem) for elem in default_conf[1:]]

    print('Experiment type:            {}'.format(args[0]))
    print('Polynomial grade:           {}'.format(args[1]))
    print('Coefficients domain for GA: {}'.format(args[2]))
    print('Population size:            {}'.format(args[3]))
    print('Generations:                {}'.format(args[4]))
    print('Mutation rate:              {}\n'.format(args[5]))

    experiment = args[0]
    poly_grade = args[1]
    coef_domain = args[2]
    pop_size = args[3]
    generations = args[4]
    mut_rate = args[5]

    # Useful rename
    poly_model = np.polynomial.polynomial.polyval

    if experiment == 's':

        # Synthetic data
        synthethic_domain = (-1, 1)
        # (poly_grade + 1) coefficients for a polynomial function
        model_coeffs = [np.random.random() * (synthethic_domain[1] - synthethic_domain[0]) + synthethic_domain[0] for _
                        in range(poly_grade + 1)]
        # Data quantity
        n = 300
        X = np.linspace(start=-50, stop=50, num=n)
        # evaluation + noise
        Y = poly_model(X, model_coeffs)
        noise = np.random.random(X.shape) * (np.max(Y) - np.min(Y)) + np.min(Y)
        Y += noise

        plt.plot(X, Y, label='Original data')
        plt.show()


        def fitness_function(c):
            return coefficient_fitness(c, list(zip(X, Y)), poly_model)


        # GA setup
        ga = GeneticAlg(pop_size=pop_size, organism_type=Coefficient, organism_size=len(model_coeffs),
                        fitness_function=fitness_function, selection_function=coefficient_selection,
                        mutation_rate=mut_rate, organism_kwargs={'domain_limits': coef_domain})
        n_gens = generations
        max_gens, organism_evol, fitness_evol = ga.evolve(generations=n_gens, trace_evolution=True)

        # Plot original data
        plt.plot(X, Y, label='Original data')
        # plt.plot(X, poly_model(X, organism_evol[0].representation), label='Best gen 1')
        plt.plot(X, poly_model(X, organism_evol[int(max_gens / 3)].representation),
                 label='Best gen {}'.format(int(max_gens / 3) + 1))
        plt.plot(X, poly_model(X, organism_evol[2 * int(max_gens / 3)].representation),
                 label='Best gen {}'.format(int(2 * max_gens / 3) + 1))
        plt.plot(X, poly_model(X, organism_evol[-1].representation), label='Best gen {}'.format(max_gens + 1))
        plt.legend()

        plt.show()

    elif experiment == 'r':

        # Real data. Coefficients unknown
        # Proposed model: f(x) = c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4 +
        #                        c5*sin(c6*x + c7)*exp(c8*x) +
        #                        c9*sin(c10*x + c11)*exp(c12*x)

        n_coeffs = poly_grade + 1 + 6


        def polysinexp_model(x, coeffs):
            poly_part = poly_model(x, coeffs[:poly_grade+1])
            sin_exp_1 = coeffs[poly_grade] * np.sin(coeffs[poly_grade+1] * x + coeffs[poly_grade+2])
            sin_exp_2 = coeffs[poly_grade+3] * np.sin(coeffs[poly_grade+4] * x + coeffs[poly_grade+5])
            return poly_part + sin_exp_1 + sin_exp_2



        # Read data

        real_data = np.genfromtxt('datos_reales.txt')

        X, Y = real_data[:, 0], real_data[:, 1]

        plt.plot(X, Y, label='Original data')
        plt.show()


        def fitness_function_real_data(c):
            return coefficient_fitness(c, list(zip(X, Y)), polysinexp_model)


        # GA setup
        ga2 = GeneticAlg(pop_size=pop_size, organism_type=Coefficient, organism_size=n_coeffs,
                         fitness_function=fitness_function_real_data, selection_function=coefficient_selection,
                         mutation_rate=mut_rate, organism_kwargs={'domain_limits': coef_domain})
        n_gens = generations
        max_gens, organism_evol, fitness_evol = ga2.evolve(generations=n_gens, trace_evolution=True)

        # Plot original data
        plt.plot(X, Y, label='Original data')
        # plt.plot(X, polysinexp_model(X, organism_evol[0].representation), label='Best gen 1')
        # plt.plot(X, polysinexp_model(X, organism_evol[int(max_gens / 3)].representation),
        #         label='Best gen {}'.format(int(max_gens / 3) + 1))
        plt.plot(X, polysinexp_model(X, organism_evol[2 * int(max_gens / 3)].representation),
                 label='Best gen {}'.format(int(2 * max_gens / 3) + 1))
        plt.plot(X, polysinexp_model(X, organism_evol[-1].representation), label='Best gen {}'.format(max_gens + 1))
        plt.legend()

        plt.show()
