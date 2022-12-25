import numpy as np
from statsmodels.base.model import GenericLikelihoodModel
from scipy.stats import genpareto


# Data contains 24 experimentally obtained values
data = np.array([3.3768732 , 0.19022354, 2.5862942 , 0.27892331, 2.52901677,
       0.90682787, 0.06842895, 0.90682787, 0.85465385, 0.21899145,
       0.03701204, 0.3934396 , 0.06842895, 0.27892331, 0.03701204,
       0.03701204, 2.25411215, 3.01049545, 2.21428639, 0.6701813 ,
       0.61671203, 0.03701204, 1.66554224, 0.47953739, 0.77665706,
       2.47123239, 0.06842895, 4.62970341, 1.0827188 , 0.7512669 ,
       0.36582134, 2.13282122, 0.33655947, 3.29093622, 1.5082936 ,
       1.66554224, 1.57606579, 0.50645878, 0.0793677 , 1.10646119,
       0.85465385, 0.00534871, 0.47953739, 2.1937636 , 1.48512994,
       0.27892331, 0.82967374, 0.58905024, 0.06842895, 0.61671203,
       0.724393  , 0.33655947, 0.06842895, 0.30709881, 0.58905024,
       0.12900442, 1.81854273, 0.1597266 , 0.61671203, 1.39384127,
       3.27432715, 1.66554224, 0.42232511, 0.6701813 , 0.80323855,
       0.36582134])
params = genpareto.fit(data, floc=0, scale=0)
# HOW TO ESTIMATE/GET ERRORS FOR EACH PARAM?

print(params)
print('\n')


class Genpareto(GenericLikelihoodModel):

    nparams = 2

    def loglike(self, params):
        # params = (shape, loc, scale)
        return genpareto.logpdf(self.endog, params[0], 0, params[2]).sum()