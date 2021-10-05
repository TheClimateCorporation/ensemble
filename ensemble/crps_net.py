#!/usr/bin/env python
# coding: utf-8

# Authors: Steven Brey

# The methods in this file can be used to implement CRPS-Net, 
# a Nonparametric Probability Density Function Estimator 

import logging
import numpy as np
from typing import Union

try:
    import tensorflow as tf
    TF_IMPORT_ERROR = None
except ImportError as e:
    TF_IMPORT_ERROR = e
    logging.warning(
        "Tensorflow not available. "
        "Will use numpy backend."
    )

if TF_IMPORT_ERROR is None : 

    def crps_sample_score(
        y_true: Union[np.ndarray, tf.Tensor], 
        y_pred: Union[np.ndarray, tf.Tensor], 
        batch_mean: bool=True
    ) -> tf.Tensor:
        """Calculates the Continuous Ranked Probability Score (CRPS)
        for finite ensemble members. 

        This implementation is based on the identity:
        .. math::
            CRPS(F, x) = E_F|X - x| - 1/2 * E_F|X - X'|

        where X and X' denote independent random variables drawn from the forecast
        distribution F, and E_F denotes the expectation value under F.
        Following the aproach of 
        https://github.com/TheClimateCorporation/properscoring for
        for the actual implementation.

        These code were adapted from a version observed http://www.cs.columbia.edu/~blei/
        on 4/21/2020 

        Reference
        ---------
        Tilmann Gneiting and Adrian E. Raftery. Strictly proper scoring rules,
            prediction, and estimation, 2005. University of Washington Department of
            Statistics Technical Report no. 463R.
            https://www.stat.washington.edu/research/reports/2004/tr463R.pdf
            
        Parameters
        ----------
        y_true: tf.Tensor | np.float32
            Target values
        y_pred: tf.Tensor | np.float32 shape=(len(y_true), n_ensemble_members)
          Same type as y_true. May contain additional columns (members) for
          corresponding y_true values. 
        batch_mean : bool, default=True
            When True the mean CRPS for all examples is returned. 
            When False individual scores are returned. 

        Return
        ------
        tf.Tensor 

        """
        
        # Variable names below reference equation terms in docstring above
        term_one = tf.reduce_mean(tf.abs(tf.subtract(y_pred, y_true)), axis=-1)
        term_two  = tf.reduce_mean( 
            tf.abs(
                tf.subtract(tf.expand_dims(y_pred, -1), tf.expand_dims(y_pred, -2))
            ), 
            axis=(-2, -1)
        ) 
        half = tf.constant(-0.5, dtype=term_two.dtype)
        score = tf.add(term_one, tf.multiply(half, term_two))
        
        if batch_mean :
            score = tf.reduce_mean(score)
         
        return score


else : 

    def crps_sample_score(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        batch_mean: bool=True
    ) -> np.ndarray:
        """Calculates the Continuous Ranked Probability Score (CRPS)
        for finite ensemble members. 

        This implementation is based on the identity:
        .. math::
            CRPS(F, x) = E_F|X - x| - 1/2 * E_F|X - X'|

        where X and X' denote independent random variables drawn from the forecast
        distribution F, and E_F denotes the expectation value under F.
        Following the aproach of 
        https://github.com/TheClimateCorporation/properscoring for
        for the actual implementation.

        These code were adapted from a version observed http://www.cs.columbia.edu/~blei/
        on 4/21/2020 

        Reference
        ---------
        Tilmann Gneiting and Adrian E. Raftery. Strictly proper scoring rules,
            prediction, and estimation, 2005. University of Washington Department of
            Statistics Technical Report no. 463R.
            https://www.stat.washington.edu/research/reports/2004/tr463R.pdf
            
        Parameters
        ----------
        y_true: ndarray
            Target values
        y_pred: ndarray shape=(len(y_true), n_ensemble_members)
          Same type as y_true. May contain additional columns (members) for
          corresponding y_true values. 
        batch_mean : bool, default=True
            When True the mean CRPS for all examples is returned. 
            When False individual scores are returned. 

        Return
        ------
        ndarray

        """
        
        # Variable names below reference equation terms in docstring above
        term_one = np.mean(np.abs(np.subtract(y_pred, y_true)), axis=-1)
        term_two  = np.mean( 
            np.abs(
                np.subtract(np.expand_dims(y_pred, -1), np.expand_dims(y_pred, -2))
            ), 
            axis=(-2, -1)
        ) 
        half = -0.5
        score = term_one - term_two / 2.0
        
        if batch_mean :
            score = np.mean(score)
         
        return score