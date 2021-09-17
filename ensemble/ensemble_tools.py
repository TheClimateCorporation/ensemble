#!/usr/bin/env python
# coding: utf-8

# Authors: Steven Brey & Tony Eckel

# Tools for working with model ensemble members. These 
# tools estimate the probability of exceeding or
# being below some threshold, which enables the user to 
# calculate the probability of some event occuring given
# a set of ensemble members. This enables Brier Scores, etc. 
# to be calculated using the properscoring package. 


# for @numba.jit(nopython=NOPYTHON) decorators 
NOPYTHON = True

import numba 
import numpy as np
import logging

def _validate_arg_type(parameter_name, parameter, expected_type) : 
    """Takes a parameter and checks to see if it matches expected_type. 
    When they do not match a TypeError is raised.
    
    Parameters
    ----------
    parameter_name : str
        The name of the parameter being checked.
    parameter : unknown
        The object being checked for paramter type
    expected_type : type
        The expexted type

    """
    expected = str(expected_type.__name__)
    recieved = str(type(parameter).__name__)
    if not isinstance(parameter, expected_type) :
        raise TypeError("{} must be type '{}'. '{}' detected".format(
            parameter_name, expected, recieved))


@numba.jit(nopython=NOPYTHON)
def _gumbel_cdf(ensemble_members, x) : 
    """calculate the gumbel cumulative density function (CDF) 
    at x given ensemble_members. Wilks (2006) note this distributions  
    handy ability to characterize extreme events (tails of PDF). 
    
    Parameters
    ----------
    ensemble_members : np.array
        ensemble members. 
    x : int | float
        Value where to evaluate the CDF. 
        
    Returns
    -------
    float, gumbel cdf value at x. 
    
    """

    # Euler’s constant, parameter of Gumbel distribution
    gamma = 0.5772156649 
    stdev = np.std(ensemble_members)
        
    # beta_hat and xi_hat are parameters of the Gumbel distribution
    # approximated with this sample ensemble_members
    beta_hat = stdev * np.sqrt(6) / np.pi
    xi_hat = np.mean(ensemble_members) - gamma * beta_hat

    return np.exp(-np.exp( (xi_hat - x) / beta_hat ) )


@numba.jit(nopython=NOPYTHON)
def _prob_from_outside_rank_gumbel(thresh, ensemble_members, thresh_position) :
    """Uses the gumbel distribution’s long right tail to estimate a portion 
    of the probability outside of the ensemble_members, as part of the 
    calculation of probability of exceeding the threshold (thresh).

    When thresh_position='above_all_members', the estimate is a proportional 
    fraction of probability in the right rank according to how far out thresh 
    is within the gumbel’s right tail. When thresh_position='below_all_members', 
    the gumbel right tail is again used by reversing the ensemble_members.
    
    Parameters
    ----------
    thresh : float
    ensemble_members : np.array(n_members,)
    thresh_position : str
        'above_all_members' or 'below all members', the condition
        of thresh relative to the ensemble members. See description above. 
    
    Returns
    -------
    float : Prob(verification >= thresh | ensemble_members & gumbel rhs tail) 
    """

    if thresh_position == 'below_all_members' : 
        #ensemble_members = np.sort(-1. * ensemble_members)
        ensemble_members = -1 * ensemble_members[::-1]
        thresh = -1. * thresh
    
    prob_per_rank = 1. / (len(ensemble_members) + 1.)
    gumbel_cdf_val_at_thresh = _gumbel_cdf(ensemble_members, thresh)
    gumbel_cdf_max_member    = _gumbel_cdf(ensemble_members, np.max(ensemble_members))
    cdf_above_thresh         = (1. - gumbel_cdf_val_at_thresh) 
    cdf_above_max_member     = (1. - gumbel_cdf_max_member)
    
    # probability verification above thresh given members, gumbel 
    # NOTE: assert statements not allowed in jit decorated functions
    # NOTE: so have to throw an error for invalid thresh_position via 
    # NOTE: crazy value
    prob = cdf_above_thresh / cdf_above_max_member * prob_per_rank
    if thresh_position == 'above_all_members' : 
        event_prob = prob 
    elif thresh_position == 'below_all_members' : 
        event_prob  = 1. - prob
    else :
        # something obsurd
        event_prob = 1e32

    return event_prob

    
@numba.jit(nopython=NOPYTHON)
def _prob_from_outside_rank_exp(thresh, ensemble_members, lambda_=3.) :
    """Calculates Prob(V >= thresh | ensemble_members & exp left tail) 
    i.e. what is the probability of verifying value (V) being greater
    or equal to thresh given ensemble_members and assuming
    the verification cannot be less than zero and exponentially decreasing
    towards zero. This function is only valid when thresh is below all
    ensemble members! This function is good for things like precip and wind,
    which cannot be negative. 
    
    Paramters
    ---------
    thresh : float
        Threshold of interest for calculating event probability
    ensemble_members : np.array(n_members, 1)
        Ensemble members in ascending order
    lambda_ : float 
        Parameter of the exponential distribution. 
    
    Returns
    -------
    float : Prob(V >= thresh | ensemble_members & exponential tail)
    """

     # constant in exponential distribution 
    n = len(ensemble_members)
    prob_per_rank = 1. / (n + 1)
    e1_member = np.min(ensemble_members)
    
    # Prob in full and partial ranks above
    prob_full_ranks_above = prob_per_rank * n 
    prob_between_thresh_and_min_member = (1. - (thresh/e1_member )**lambda_) * prob_per_rank

    prob = prob_between_thresh_and_min_member + prob_full_ranks_above
    
    return prob

@numba.jit(nopython=NOPYTHON)
def _deterministic_event_prob(forecast, thresh, operator_str) :
    """A deterministic forecast can only return probabilities
    of 0 and 1 for events, but operator_str must still be handled.
    For each operator_str there are three positions thresh can take
    below, equal, or above forecast. These result in probabilities
    of either 0 or 1. If statements for each conditions written
    separately for clarity. 
    
    Parameters
    ----------
    forecast : float(1,)
        The predicted (forecast) value. 
    thresh : float(1,) : 
        Threshold of interest, combined with operator_str
        to create an event. 
    operator_str :  str
        One of 'greater', 'less', 'less_equal', 'greater_equal'
        to combine with thresh to great an event.

    Returns
    -------
    probability_V_greater_equal_thresh : float(1,)    
        The probability of the event given
        operator_str and forecast*
        *Not intended for use outside of _probability_from_members()

    """

    if thresh > forecast and operator_str == 'greater' :
        probability_V_greater_equal_thresh = 0. 

    elif thresh < forecast and operator_str == 'greater' : 
        probability_V_greater_equal_thresh = 1.

    elif thresh == forecast and operator_str == 'greater' :
         probability_V_greater_equal_thresh = 0. 

    elif thresh > forecast and operator_str == 'less' : 
        probability_V_greater_equal_thresh = 0. 

    elif thresh < forecast and operator_str == 'less' : 
        probability_V_greater_equal_thresh = 1.

    elif thresh == forecast and operator_str == 'less' : 
        probability_V_greater_equal_thresh = 1. 

    elif thresh == forecast and operator_str == 'greater_equal' : 
         probability_V_greater_equal_thresh = 1. 

    elif thresh < forecast and operator_str == 'greater_equal' : 
        probability_V_greater_equal_thresh = 1.

    elif thresh > forecast and operator_str == 'greater_equal' :
        probability_V_greater_equal_thresh = 0.

    elif thresh == forecast and operator_str == 'less_equal' : 
         probability_V_greater_equal_thresh = 1. 

    elif thresh < forecast and operator_str == 'less_equal' : 
        probability_V_greater_equal_thresh = 1.
   
    elif thresh > forecast and operator_str == 'less_equal' : 
        probability_V_greater_equal_thresh = 0.

    else :
        # numba crude 'error handling', should never get here.
        # But if we do, at least the result will be obviously wrong. 
        probability_V_greater_equal_thresh = 1e64

    return probability_V_greater_equal_thresh


@numba.jit(nopython=NOPYTHON)
def _probability_from_members(
    thresh, 
    members, 
    operator_str, 
    presorted,
    positive_definite
) : 
    """After the arguments are validated this function executes the 
    operations described in the probability_from_members() function
    documentation. See that function for a complete docstring. This 
    function is separated in order to narrow the domain of numba.jit
    so that python code (asserting types etc) can be wrapped around
    this faster looping functionality.
    """
        
    # Create an array to store probabilities for each example 
    n_examples = members.shape[0]
    probabilty_of_event = np.full((n_examples, 1), np.nan)
        
    # Loop through each example 
    for i in range(n_examples) : 
    
        # Calculate P(V >= thresh | members)
        M = members[i, :]
        if not presorted : 
            M = np.sort(M)

        # There is always one more rank than ensemble members
        # (see docstring for description of ranks)
        num_members = len(M)
        n_ranks = num_members + 1
    
        # Probability is split evenly between ranks
        prob_per_rank = 1. / n_ranks

        if np.std(M) == 0 :
            # point mass forecast detected, in terms of probability
            # this can be handled like a deterministic forecast
            M_unique =  M[0]
            probability_V_greater_equal_thresh=\
            _deterministic_event_prob(M_unique, thresh, operator_str)

        else : 
            # There are at least two unique values for members M
            # In this section, probabilities are linearly interpolated
            # between members, or distribution tails assumed when 
            # threshold is above all or below all. 

            # Handle full bin probabilities based on the event type
            if operator_str == "greater" : 
                num_full_bins_above = np.sum(M > thresh)
            elif operator_str == "greater_equal" : 
                num_full_bins_above = np.sum(M >= thresh)

            # Handle what happens if there are tied members 
            # and the end goal is getting probability below
            if operator_str == 'less' : 
                num_full_bins_above = np.sum(M >= thresh)
            elif operator_str == 'less_equal' : 
                num_full_bins_above = np.sum(M > thresh)

            if num_full_bins_above == 0 :
                
                # None of the members are above thresh 
                # Handle rhs probability tail with gumbel distribution
                probability_V_greater_equal_thresh = _prob_from_outside_rank_gumbel(thresh, M,
                    thresh_position='above_all_members')     

            elif num_full_bins_above == members.shape[1] :

                # TODO: Figure out why lines 311-318 cannot get hit by unit tests
                # All of the members are above the threshold. Implement left tail. 
                if positive_definite :

                    if np.min(M) == 0. and operator_str == 'greater_equal' :
                        # All values are greater_equal 0. for positive definite
                        probability_V_greater_equal_thresh = 1. 
                    elif np.min(M) == 0. and operator_str == 'less' : 
                        # There are no values less than zero for positve definite.
                        # This value will be taken from 1 later. 
                        probability_V_greater_equal_thresh = 1. 
                    else : 
                        # Approximate probability of V >= thresh given no negatives possible
                        probability_V_greater_equal_thresh = _prob_from_outside_rank_exp(
                            thresh, M)
                else : 
                    # Calculate probability V >= thresh given gumbel rhs for left
                    # side of the distribution these M are drawn from
                    probability_V_greater_equal_thresh = _prob_from_outside_rank_gumbel(
                        thresh, M, thresh_position="below_all_members")

            # TODO: There is another condition!?!?! There are n_bins above 
            # TODO: the thresh for this positive definite
            #elif num_full_bins_above == members.shape[1] + 1
                            
            else :
                # thresh is between two members or equal to one or more members
                
                # Probability from full bins above thresh
                full_bins_above_prob = num_full_bins_above * prob_per_rank
                
                # Calculate partial bin above thresh
                m_above_value = M[-num_full_bins_above]
                m_below_value = M[-(num_full_bins_above + 1)]
                
                # When thresh is equal to a member, there is no partial bin probability. 
                # In addition, when m_above_value and m_below_value are the same, there
                # is no partial bin probability and division by zero must be avoided. 
                if m_above_value == m_below_value : 
                    partial_bin_above_prob = 0. 
                else : 
                    partial_bin_above_prob = np.divide(m_above_value - thresh, 
                        m_above_value - m_below_value) * prob_per_rank
                
                # Total probability of V above thresh is full bins plus partial bins above
                probability_V_greater_equal_thresh = full_bins_above_prob + partial_bin_above_prob
        
        if operator_str == "greater" or operator_str == "greater_equal"  :
            probabilty_of_event[i, 0] = probability_V_greater_equal_thresh
        else :  
            # Flip prob to gives probability below thresh
            probabilty_of_event[i, 0] = 1 - probability_V_greater_equal_thresh

    return probabilty_of_event


def probability_from_members(
    thresh, 
    members, 
    operator_str, 
    presorted=False,
    positive_definite=False
) : 
    """This function calculates probabilities of an event, using the rank method 
    (Hamill & Colucci 1997), for one or more sets of ensemble members, where 
    an event is the occurrence of the random variable within a specified range 
    as defined by a threshold (thresh) and a comparison function (operator_str).
    Ensemble members are assumed to represent a continuum of probability. 
    Gaps  between the ordered members are considered to represent evenly divided 
    quantiles of continuous probability. This is a correct assumption 
    on average but not necessarily for every forecast (example). Note that 
    meaningful probabilities are only possible to estimate if the 
    underlying probability distribution function (PDF) is well sampled, 
    i.e. you have enough members in your ensemble to represent the PDF. 
    Greater than 30 is prefered, less than 10 is not recommended. 
    
    Schematic showing how code estimates probabilities of events from members:    
        
         m0      m..      mn           <- ensemble member values
         |       |        |            <- ensemble members
      1      2       ..       n + 1    <- ranks

    Where "|" are ensemble members with values m0 through mn
    and there are n ensemble members and n+1 ranks.  
    
    Reference
    ---------
    Thomas M. Hamill and Stephen J. Colucci Verification of Eta–RSM
    Short-Range Ensemble Forecasts (1997)
    https://doi.org/10.1175/1520-0493(1997)125<1312:VOERSR>2.0.CO;2

    Parameters
    ----------
    thresh : float
        Value to estimate the probability of exceeding (when above=True) 
        or being below (above=False) this value. 
    members : numpy.array(n_examples, n_members) 
        The predicted ensemble members. n_examples must be at least 1. 
        i.e. use np.expand_dims(members, 0) to turn single forecast to row vector. 
        When n_examples > 1, each forecast must have the same number of members
        (columns). 
    operator_str : str
        - When 'greater', gives answer to "what is the probability of verifying 
          value exceeding thresh"? 
        - When 'greater_equal', gives answer to "what is the probability of 
          verifying value exceeding or being eqaul to thresh"? 
        - When 'less', gives answer to "what is the probability of the verifying 
          value being below thresh"?
        - When 'less_equal', gives answer to "what is the probability of the 
          verifying value being below or equal to thresh"?
    presorted : bool
        Whether or not members are presorted to be increasing to the right.
        When False, they will be sorted. Better speed performance is 
        achieved when data are presorted. 
    positive_definite : bool
        When True members estimate a positive-definite variable e.g. precip
        or wind speed. When this is the case, probabilities of an event less
        than zero are not allowed. When False, A gumbel distribion long tail
        is used to estimate probability of extreme events above or below
        members. 
        
    Returns
    -------
    proba : np.array(n_examples, 1)
        Prob(verification operator_str thresh | members) 
    """

    # Validate parameter types -----
    if isinstance(thresh, int) : 
        thresh = float(thresh) # well all float on
    _validate_arg_type("thresh", thresh, float)
    _validate_arg_type("members", members, np.ndarray)
    _validate_arg_type("operator_str", operator_str, str)
    _validate_arg_type("presorted", presorted, bool)
    _validate_arg_type("positive_definite", positive_definite, bool)

    # Validate parameters -----
    valid_operator_str = np.array(["greater", "greater_equal", "less", "less_equal"])
    if np.sum(operator_str == valid_operator_str) != 1 :
         raise ValueError("operator_str='{}' is invalid. operator_str must be equal to one of {}".format(
            operator_str, [v for v in valid_operator_str]))

    assert(len(members.shape) > 1),\
    ("Members must have row axis even if single forecast used. np.expand_dims(members, 1)")

    if members.shape[1] < 10 : 
        logging.warning("10 or more ensemble members needed in order for meaningful probabilities to be calculated!")

    if members.shape[1] <= 3 : 
        raise ValueError("Paternalism. These code should not be used for 3 members or less.")

    if positive_definite and np.less(members, 0.).any() :
        raise ValueError("No negative 'members' values allowed for positive definite variables!")

    if positive_definite and thresh < 0:
        raise ValueError("Negative thresholds are not allowed for positive definite variables.")

    if positive_definite and thresh == 0. and operator_str == "less":
        logging.warning("Probability of positive definite less than zero is always 0.")

    if positive_definite and thresh == 0. and operator_str == "greater_equal":
        logging.warning("Probability of positive definite greater_equal zero is always 1.")

    if (~np.isfinite(members)).any() : 
        raise ValueError("Non finite values detected in 'members'. All values must be finite.")

    if presorted : 
        logging.warning("If members are not presorted-invalid probabilities may be returned\
            `presorted=False' ensures all examples are sorted for valid results.")

    # Calculate probabilities using jit nopython function, which loops
    # through examples. This private function performs the operations
    # described in the docstring. 
    probabilty_of_event = _probability_from_members(
        thresh=thresh, members=members, operator_str=operator_str, 
        presorted=presorted, positive_definite=positive_definite
    )

    # Validate output -----
    if np.max(np.abs(probabilty_of_event)) > 1e31 : 
        raise ValueError("Make sure `thresh_position` arg in prob_greater_thresh_given_gumbel\
            is set correctly")

    if np.max(np.abs(probabilty_of_event)) > 1e62 : 
        raise ValueError("Make sure 'deterministic' forecast value handled correctly.")

    # Sanity check the estimates to be inside [0-1] range
    assert(np.max(probabilty_of_event) <=1),\
    ("Event probability greather than 1 estimated! Something is wrong.")

    assert(np.min(probabilty_of_event) >=0),\
    ("Event probability less than 0 estimated! Something is wrong.")
    
    return probabilty_of_event


def prob_between_values(
    members, 
    lower, 
    upper, 
    bracket="[]", 
    positive_definite=False
) : 
    """Calculates the probability of a value falling in the range 
    lower to upper with inclusive conditions set by bracket argument. 
    Please see _probability_from_members() for full parameter descriptions. 

    Parameters
    ----------
    members : numpy.array(n_examples, n_members)
        See _probability_from_members() docstring for full details.
    lower : float
        Lower bound
    upper : float 
        Upper bound
    bracket : str
        "[]", "[)", "()", "(]" where () are exclusive and [] inclusive.
        Assuming integers:
        (0, 5) = 1, 2, 3, 4
        (0, 5] = 1, 2, 3, 4, 5
        [0, 5) = 0, 1, 2, 3, 4
        [0, 5] = 0, 1, 2, 3, 4, 5
    positive_definite : bool
        Whether or not members represent a positive definite function
        (cannot take on negative values). See probability_from_members() 
        docstring for full details.

    Returns
    -------
    float : numpy.array(n_examples, 1)
        The probability of the event.
    """
    
    # validate parameter types -----
    if isinstance(lower, int) : 
        lower = float(lower)
    if isinstance(upper, int) : 
        upper = float(upper)

    _validate_arg_type('members', members, np.ndarray)
    _validate_arg_type('lower', lower, float)
    _validate_arg_type('upper', upper, float)
    _validate_arg_type('bracket', bracket, str)
    _validate_arg_type('positive_definite', positive_definite, bool)

    # Check for valid bracket 
    valid_brackets = np.array(["[]", "[)", "()", "(]"])
    if np.sum(bracket == valid_brackets) != 1 :
        raise ValueError("bracket='{}' is invalid. bracket must be equal to one of {}".format(
            bracket, [vb for vb in valid_brackets]))

    assert(upper > lower),\
    ("upper must be greater than lower. Zero width is always zero area silly!")
    
    # Handle upper and lower end inclusive vs. exclusive range
    if bracket[0] == "[":
        lower_operator_str = "greater_equal"
    else : 
        lower_operator_str = "greater"

    if bracket[1] == "]" : 
        upper_operator_str = "greater"
    else :
        upper_operator_str = "greater_equal"

    # Estimate total probability above each limit based on bracket conditions
    prob_ge_low = probability_from_members(
        lower, 
        members, 
        operator_str=lower_operator_str, 
        positive_definite=positive_definite
    )

    prob_ge_high = probability_from_members(
        upper, 
        members, 
        operator_str=upper_operator_str, 
        positive_definite=positive_definite
    )

    probability_between = prob_ge_low - prob_ge_high

    # Sanity check the estimates to be inside [0-1] range
    assert((np.min(probability_between) >= 0) and (np.max(probability_between) <=1)),\
    ("Event probability outside [0-1] range detected!")

    return probability_between


@numba.jit(nopython=NOPYTHON)
def _ensemble_verification_rank(v_, M) : 
    """Managed by ensemble_verification_rank() to limit the domain of
    the numba.jit constraints. Please refer to ensemble_verification_rank() 
    for full docstring and parameter descriptions.
    """
    
    n_examples = len(v_)
    n_members  = M.shape[1]
    # because left rank is 0, right rank is equal
    # to total number of members
    max_rank = n_members 
    
    # Array to store ranks 
    rank = np.full((n_examples, 1), np.nan) 
    
    for i in range(n_examples) : 
        
        # Get ensemple members for this example and sort
        members = np.sort(M[i, :])
           
        # Get verification for this example
        v = v_[i]
        
        # Find the rank ----- 
        members_below_v_count = np.sum(members < v)
        
        # Count the number of perfect matches
        perfect_match_mask = v == members
        perfect_match_count = np.sum(perfect_match_mask)

        if members_below_v_count == n_members :
            # verification is above all members
            # the verification is in the highest rank
            rank[i, 0] = max_rank

        elif members_below_v_count == 0 and perfect_match_count == 0:
            # All members are above v
            # the verification falls into the first rank
            rank[i, 0] = 0

        else : 
            # The verification is between two members or equal 
            # to one or more members. 

            # If there is one or more perfect match, assign the verificaiton
            # to one of those ranks randomly. Over many examples, this is 
            # statistically consistent. 
            if perfect_match_count > 0 : 

                matched_index = np.where(perfect_match_mask)[0]
                min_matched_member_index = matched_index[0]
                max_matched_member_index = matched_index[-1]

                # Place v in rank inbetween, above, or below matching members
                # NOTE: np.random.randint low inclusive, high exclusive
                rank[i, 0] = np.random.randint(low=min_matched_member_index,
                    high=(max_matched_member_index + 2))

            else :
                # No exact matching members and not in outside
                # ranks. The verificaiton is between two members. 
                rank[i, 0] = members_below_v_count

    return rank


def ensemble_verification_rank(v_, M, positive_definite=False) : 
    """Calculates verification ranks for ensemble forecasts. 
    Definition of ranks illustrated below.
    
        0     1     2    3    4         <- Ensemble member label
        |     |     |    |    |         <- Ensemble member value on v scale
      0    1     2     3    4     5     <- Ranks
           v0       v2            v3    <- Verification values 
           
    v ------------------------------> 
      
    In the above schematic, the "|" are ensemble members, the 
    matching labels are the sorted ensemble member labels, and the
    values below and between the "|" are the ranks. v0, v2, and v3
    get ranks 1, 2 or 3, and 5 respectively. When more than one member matches
    a value the rank is randomly assigned to one of the ranks between or to
    the right or left of the matching members. We use the convention that 
    whenever a member value is tied to the verification the ranks on 
    BOTH sides of that member are in play. e.g. below
    
       -1   0   1      <- member values
        |   |   |      <- members
      0   1   2   3    <- Ranks
    
    if v_=0 select rank 1 or 2 at random
    if v_=1 select rank 2 or 3 at random
    ...
    
    Reference
    ---------
    Thomas M. Hamill. Interpretation of Rank Histograms for Verifying 
    Ensemble Forecasts (2001)
    https://doi.org/10.1175/1520-0493(2001)129<0550:IORHFV>2.0.CO;2
    
    Parameters
    ----------
    v_ : np.array(n_examples, 1)
        verification values 
    M : np.array(n_examples, n_members)
        member forecasts
    positive_definite : bool
        Whether or not v_ and M represent positive definite variables
        (things that cannot be negative). When True, the args will be
        validated and negative values will throw errors.
    
    Returns
    -------
    verification ranks np.array(n_examples, 1)
    """

    # Validate passed parameter types -----
    _validate_arg_type('v_', v_, np.ndarray)
    _validate_arg_type('M', M, np.ndarray)
    _validate_arg_type('positive_definite', positive_definite, bool)

    # Validate passed parameters logic -----
    assert(len(M.shape) > 1),\
    ("Members must have row axis even if single forecast used. \
    np.expand_dims(members, 1)")

    assert(len(v_.shape) > 1),\
    ("v_ must have row axis even if single forecast used. np.expand_dims(v_, 1)")

    if positive_definite and (v_ < 0).any() : 
        raise ValueError("There should not be any negative verifications for positive definite.")
    if positive_definite and (M < 0).any() : 
        raise ValueError("There should not be any negative members M for positive definite.")

    ranks = _ensemble_verification_rank(v_, M)

    # Count the unique ranks and warn the user when a rank goes unused 
    n_ranks = M.shape[1] + 1
    rank_values, rank_counts = np.unique(ranks, return_counts=True)
    if len(rank_values) < n_ranks : 
        logging.warning("Not all ranks used! Only ranks used are {}. ".format(
            [str(int(r)) for r in rank_values]))

    return ranks