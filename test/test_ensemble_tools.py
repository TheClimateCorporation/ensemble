"""
Tests for functions in ensemble_tools.py.

Authors: Tony Eckel & Steven Brey

Note that functions that start with an underscore (_) are designed for local 
use only by the primary functions within ensemble_tools.py. Therefore, testing
of those local scripts does not include checking for irrational inputs that
would cause meaningless results and/or an exception since such inputs are 
checked for earlier in the primary functions.
"""

import numpy as np
import pytest
import sys

from ensemble.ensemble_tools import (
    _gumbel_cdf,
    _probability_from_members,
    probability_from_members,
    _prob_from_outside_rank_gumbel,
    _prob_from_outside_rank_exp,
    _deterministic_event_prob,
    probability_from_members,
    prob_between_values,
    ensemble_verification_rank,
    _validate_arg_type
  )


# Define ensemble datasets to test with
MEMBERS_ALPHA = np.array([[0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.5, 0.5, 0.5, 1.0]])
MEMBERS_BRAVO = np.array([[-1.0, -1.0, 0.0, 0.0, 0.1, 0.2, 0.5, 0.5, 1.0, 1.0]])
MEMBERS_CHARLIE = np.array([[7, 7, 7, 7, 7, 7, 7, 7, 7, 7]])

# Set the roundoff decimals for testing precision
ROUNDOFF = 5

# ------------------------------------------------------------------------------
#                                 _gumbel_cdf
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("members, x, expected",
                         [(MEMBERS_ALPHA[0], 1.1, 0.97949),
                          (MEMBERS_ALPHA[0], 10.0, 1.0),
                          (MEMBERS_ALPHA[0], -10.0, 0.0),
                          ])
def test_gumbel_cdf(members, x, expected):
  print("I AM TESTING _gumbel_cdf")
  assert np.round(_gumbel_cdf(members, x), ROUNDOFF) == expected


# ------------------------------------------------------------------------------
#                         _prob_from_outside_rank_gumbel
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("threshold, members, threshold_position, expected",
                         [(1000, MEMBERS_ALPHA, 'above_all_members', 0.0),
                          (-1000, MEMBERS_ALPHA, 'below_all_members', 1),
                          (1.1, MEMBERS_ALPHA, 'mouse_burgers', 1e32),
                          (1.1, MEMBERS_ALPHA, 'above_all_members', 'between_0_1')])
def test_prob_from_outside_rank_gumbel(
  threshold, members, threshold_position, expected
):

  prob = _prob_from_outside_rank_gumbel(threshold, members, threshold_position)

  if isinstance(expected, float) or isinstance(expected, int) : 

    assert(np.round(prob, ROUNDOFF) == expected),\
    (f"prob={prob}, expected={expected}")

  else :

    assert(prob < 1. and prob > 0.),\
    ("Expected prob between zero and one but got {prob}")

# ------------------------------------------------------------------------------
#                         _prob_from_outside_rank_exp
# ------------------------------------------------------------------------------
# this function estimates Prob(V >= thresh | ensemble_members & exp left tail)
# where V is the verifying value, thresh is a user specified threshold for some
# event, and ensemble_members are predictions from e.g. crps-net. 

@pytest.mark.parametrize("threshold, members",
                         [(0.5,  np.array([1,2,3,4])),])
def test_prob_from_outside_rank_exp(threshold, members) : 

  n_bins = len(members) +  1
  n_prob_per_bin = 1 / n_bins

  assert(_prob_from_outside_rank_exp(np.min(members), members) == (1-n_prob_per_bin)),\
  (f"when thresh is tied with lower member, there is a full bin of probability below")

  assert(_prob_from_outside_rank_exp(0, members) == 1)
  (f"Can't be negative, so proba greater or above 0 always 1 regardless of members")

  prob = _prob_from_outside_rank_exp(threshold, members)

  assert(prob < 1 and prob > 0),\
  ("probability of this tail must be between 0 and 1")




# ------------------------------------------------------------------------------
#                            probability_from_members
# ------------------------------------------------------------------------------

# Tests for ValueErrors
@pytest.mark.parametrize("threshold, members, operator_str, presorted, positive_definite",
                         [(-0.15, MEMBERS_ALPHA, 'greater', True, True),
                          (0.1, MEMBERS_BRAVO, 'greater', True, True),
                          (0.1, np.array([[1, 2, 3]]), 'greater', True, True),
                          (0.1, MEMBERS_ALPHA, 'Steve Brey rocks!', True, True),
                          (0.1, np.array([[1, 2, 3, 4, 5, 6, np.NaN, 8, 9, 10]]), 'greater', True, True),
                          ] )
def test1_probability_from_members(threshold, members, operator_str, presorted, positive_definite):
    with pytest.raises(ValueError):
        probability_from_members(threshold, members, operator_str, presorted,
                                 positive_definite)

# Tests for TypeErrors
@pytest.mark.parametrize("threshold, members, operator_str, presorted, positive_definite",
                         [('string', MEMBERS_ALPHA, 'greater', True, True),
                          (1, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'greater', True, True),
                          (1, MEMBERS_ALPHA, True, True, True),
                          (1, MEMBERS_ALPHA, 'greater', 'string', True),
                          (1, MEMBERS_ALPHA, 'greater', True, 999),
                          ] )
def test2_probability_from_members(threshold, members, operator_str, presorted, positive_definite):
    with pytest.raises(TypeError):
        probability_from_members(threshold, members, operator_str, presorted,
                                 positive_definite)


# ------------------------------------------------------------------------------
#                           _probability_from_members
# ------------------------------------------------------------------------------
MEMBER_POSITIVE_DEF = np.array([0, 1, 2, 3, 4, 5])
# Tests for threshold between members
@pytest.mark.parametrize("threshold, members, operator_str, presorted, positive_definite, expected",
                         [(0.15, MEMBERS_ALPHA, 'greater', True, True, 0.5),
                          (0.15, MEMBERS_ALPHA, 'greater', False, False, 0.5),
                          (0.15, MEMBERS_ALPHA, 'greater_equal', True, True, 0.5),
                          (0.15, MEMBERS_ALPHA, 'greater_equal', True, False, 0.5),
                          (0.15, MEMBERS_ALPHA, 'less', True, True, 0.5),
                          (0.15, MEMBERS_ALPHA, 'less', True, False, 0.5),
                          (0.15, MEMBERS_ALPHA, 'less_equal', True, True, 0.5),
                          (0.15, MEMBERS_ALPHA, 'less_equal', False, False, 0.5),
                          (-100, MEMBER_POSITIVE_DEF, 'less', True, True, 0.0),
                          (-100, MEMBER_POSITIVE_DEF, 'less', False, True, 0.0),
                          ] )
def test1__probability_from_members(threshold, members, operator_str, presorted,
                                    positive_definite, expected):
  assert _probability_from_members(threshold, members, operator_str, presorted,
                                   positive_definite)[0][0] == expected

# Tests for threshold outside of members
@pytest.mark.parametrize("threshold, members, operator_str, presorted, positive_definite, expected",
                         [(1.1, MEMBERS_ALPHA, 'greater', True, True, 0.06111),
                          (7, MEMBERS_CHARLIE, 'greater', True, True, 0.0),
                          (8, MEMBERS_CHARLIE, 'greater', True, True, 0.0),
                          (8, MEMBERS_CHARLIE, 'greater_equal', True, True, 0.0),
                          (7, MEMBERS_CHARLIE, 'greater_equal', True, True, 1.0)
                          ] )
def test1__probability_from_members(threshold, members, operator_str, presorted,
                                    positive_definite, expected):
  assert np.round(_probability_from_members(threshold, members, operator_str, presorted,
                                            positive_definite)[0][0], ROUNDOFF) == expected

# Tests for handling zeros
@pytest.mark.parametrize("threshold, members, operator_str, presorted, positive_definite, expected",
                         [(0.0, MEMBERS_ALPHA, 'greater', True, True, np.round(7./11, ROUNDOFF)),
                          (0.0, MEMBERS_ALPHA, 'greater', True, False, np.round(7./11, ROUNDOFF)),
                          (0.0, MEMBERS_ALPHA, 'greater_equal', True, False, np.round(10./11, ROUNDOFF)),
                          (0.0, MEMBERS_ALPHA, 'greater_equal', True, True, 1.0)
                          ] )
def test2__probability_from_members(threshold, members, operator_str, presorted,
                                    positive_definite, expected):
  assert np.round(_probability_from_members(threshold, members, operator_str, presorted,
                                            positive_definite)[0][0], ROUNDOFF) == expected

# Tests for handling tie between threshold and members
@pytest.mark.parametrize("threshold, members, operator_str, presorted, positive_definite, expected",
                         [(0.5, MEMBERS_ALPHA, 'greater', True, True, np.round(2./11, ROUNDOFF)),
                          (0.5, MEMBERS_ALPHA, 'greater', False, False, np.round(2./11, ROUNDOFF)),
                          (0.5, MEMBERS_ALPHA, 'greater_equal', True, True, np.round(4./11, ROUNDOFF)),
                          (0.5, MEMBERS_ALPHA, 'greater_equal', False, False, np.round(4./11, ROUNDOFF)),
                          (0.5, MEMBERS_ALPHA, 'less', True, True, np.round(7./11, ROUNDOFF)),
                          (0.5, MEMBERS_ALPHA, 'less_equal', True, True, np.round(9./11, ROUNDOFF)),
                          (-1.0, MEMBERS_BRAVO, 'less', True, False, np.round(1./11, ROUNDOFF)),
                          (-1.0, MEMBERS_BRAVO, 'less_equal', True, False, np.round(2./11, ROUNDOFF)),
                          ] )
def test3__probability_from_members(threshold, members, operator_str, presorted,
                                    positive_definite, expected):
  assert np.round(_probability_from_members(threshold, members, operator_str, presorted,
                                            positive_definite)[0][0], ROUNDOFF) == expected

# ------------------------------------------------------------------------------
#                           prob_between_values
# ------------------------------------------------------------------------------
MEMBERS_DELTA = np.array([[0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.5, 0.5, 0.5, 1.]])

# TODO: Break this into logical separations 
@pytest.mark.parametrize("members, lower, upper, bracket, positive_definite, expected",
                         [(MEMBERS_DELTA, 0., 0.1, "()", True, np.round(1/11, ROUNDOFF)),
                          (MEMBERS_DELTA, 0., 0.1, "[)", True, np.round(5/11, ROUNDOFF)),
                          (MEMBERS_DELTA, 0., 0.1, "[]", True, np.round(5/11, ROUNDOFF)),
                          (MEMBERS_DELTA, 0., 0.1, "(]", True, np.round(1/11, ROUNDOFF)),
                          (MEMBERS_DELTA, 0., 0.5, "()", True, np.round(3/11, ROUNDOFF)),
                          (MEMBERS_DELTA, 0., 1., "()", True, np.round(6/11, ROUNDOFF)),
                          (MEMBERS_DELTA, 0., 1., "[]", True, np.round(10/11, ROUNDOFF)),
                          (MEMBERS_DELTA, 0., 1., "[]", False, np.round(9/11, ROUNDOFF)),
                          (MEMBERS_DELTA, 10., 11., "[]", False, np.round(0/11, ROUNDOFF)),
                          (MEMBERS_CHARLIE, 0., 10., "[]", False, np.round(11/11, ROUNDOFF)),
                          (MEMBERS_CHARLIE, 6.99, 7.01, "[]", False, np.round(11/11, ROUNDOFF)),
                          (MEMBERS_CHARLIE, 6.99, 7.01, "()", False, np.round(11/11, ROUNDOFF)),
                          (MEMBERS_CHARLIE, 7, 7.0001, "()", False, np.round(0/11, ROUNDOFF)),
                          (MEMBERS_CHARLIE, 7, 7.0001, "[)", False, np.round(11/11, ROUNDOFF)),
                          (MEMBERS_CHARLIE, 7.1, 7.2, "[]", False, np.round(0/11, ROUNDOFF)),
                          (MEMBERS_CHARLIE, 6.9, 6.91, "[]", False, np.round(0/11, ROUNDOFF)),
                         ])

def test1_prob_between_values(members, lower, upper, bracket, 
                              positive_definite, expected) : 

  assert np.round(prob_between_values(members, lower, upper, 
                                      bracket, positive_definite)[0][0], ROUNDOFF) == expected

  # Validate the parameters 


# ------------------------------------------------------------------------------
#                           ensemble_verification_rank
# ------------------------------------------------------------------------------
N = int(1e6)
MEMBERS_ZERO = np.zeros((N, 5))
V_ = np.zeros(shape=(N, 1)) 

@pytest.mark.parametrize("v_, M, expected", 
                         [(V_, MEMBERS_ZERO, np.array([0, 1, 2, 3, 4, 5])),
                         ] )

def test1_ensemble_verification_rank(v_, M, expected) :

  ranks = ensemble_verification_rank(v_, M)
  val, count = np.unique(ranks, return_counts=True)
  
  prob_per_bin = 1 / (M.shape[1] + 1)
  proportion = count / M.shape[0]

  # TODO: assert counts roughly equal to expected value 

  assert (val == expected).any() #and np.testing.assert_almost_equal(proportion[0], prob_per_bin, decimal=7)


# ------------------------------------------------------------------------------
#                           ensemble_verification_rank
# ------------------------------------------------------------------------------
# 1) correct int type 
# 2) give a str when expecting an int, catch TypeError
# 3) give a list when expecting a str, catch TypeError
# 4) give a dict when expecting a dict
@pytest.mark.parametrize("parameter_name, parameter, expected_type, raises", 
                         [("horse_face", 1, int, None),
                          ("horse_face", "moose", int, TypeError),
                          ("wild horses", ["rat"], str, TypeError),
                          ("dog gunna hunt", {"cute":"dog"}, dict, None)])
def test__validate_arg_type(parameter_name, parameter, expected_type, raises) : 
  """Test to make sure the function fails when expected."""

  if raises is not None : 
    # We expect this to raise an error
    with pytest.raises(raises) :
      _validate_arg_type(parameter_name, parameter, expected_type)
  else :
    _validate_arg_type(parameter_name, parameter, expected_type)


# ------------------------------------------------------------------------------
#                           _deterministic_event_prob
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("forecast, thresh, operator_str, expected", 
                         [(1, 1.1, "less", 0),
                          (1, 0.9, "less", 1),
                          (1, 1, "less", 1),
                          (1, 1, "less_equal", 1),
                          (1, 0, "less_equal", 1),
                          (5, 10, "less_equal", 0),
                          (5, 10, "not_a_valid_operator", 1e64),
                         ])
def test_deterministic_event_prob(forecast, thresh, operator_str, expected) : 

  prob = _deterministic_event_prob(forecast, thresh, operator_str)
  assert(prob == expected),\
  (f"prob={prob}, expected={expected}")



# THESE NEED TESTS
# _deterministic_event_prob
# probability_from_members

  
