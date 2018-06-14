# PopulationTracking
MATLAB code for implementing the Population Tracking neural data analysis method.

The model is explained in this paper:
http://biorxiv.org/content/early/2016/07/19/064717

No setup required. The following function files are included:

	fitPopTrack : fit the parameters of the Population Tracking model to binary neural population data.
	
	get_jsdiv : estimate the Jensen-Shannon divergence between two fitted distributions given the estimated population tracking model parameters.
	
	get_kldiv : estimate the Kullback-Leibler divergence between two fitted distributions given the estimated population tracking model parameters.

	samplePopTrack: generate samples from the fitted model, given the parameters.

	compute_ak: calculate the set of normalization constants, needed for exact pattern probabilities.

	compute_px: calculate the predicted probability of a specific neural activity pattern, given the model parameters.

See the help functions of each file for more details.
