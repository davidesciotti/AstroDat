import numpy as np
import scipy as sc

def test_all_params_in_bounds_LHS(LHS, bound_lower,bound_upper):
	for i in range(len(LHS)):
		assert LHS[i] >= bound_lower
		assert LHS[i] <= bund_upper


