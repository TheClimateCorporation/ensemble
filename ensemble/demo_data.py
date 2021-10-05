#demo_data.py

"""
Functions for creating synthetic data that demontrates the 
challenges and benefits of probabilistic predictions. 

TODO: Add the x-wing and two normal distributions! These should be
      in the complete forecast code.
"""

import numpy as np 

def create_point_gamma_data(
    n_examples=10000, 
    x_min=0.0, 
    x_max=10., 
    point_mass_value=0,
    point_mass_proportion=0.5
) : 
    """Creates a mixture of data drawn from a point mass 
    and gamma distributions. The idea is for these data to be ideal
    to test developing ML methods to learn and calibrate precipitation
    forecasts from numerical weather predition.

    Parameters
    ----------
    n_examples : int, default=10000
        The number of samples to generate
    x_min, x_max : floats, default=0, 10
        The spans of the data before appending point mass
        at start and end of the data time series. 
    point_mass_value : int, default=0
        The value of the point mass distribution to create.
        Not used if None. 
    point_mass_proportion : float, default=0.5
        Proportion of the examples set to point_mass_value. 


    Return
    ------
    x, y : np.arrays
        The synthetic data to learn
    """

    # Generate the y_data for n_examples x_random data 
    x = np.float32(np.random.uniform(x_min, x_max, n_examples))
    r_data = np.array([np.random.gamma(shape=i, scale=i) for i in x])
    y = np.float32(np.power(x, 3) + r_data*4.0)

    if point_mass_value is not None : 
                
        # Place at random places along x
        n_point_mass = int(point_mass_proportion * n_examples)
        replacement_locations = np.random.randint(0, n_examples, n_point_mass)
        
        y[replacement_locations] = point_mass_value
        

    # Reshape the data for ML 
    x = np.expand_dims(x, 1)
    y = np.expand_dims(y, 1)
    
    return x, y


# TODO: Get the other example data as functions in here. 