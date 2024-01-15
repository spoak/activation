# imports
import datetime
import pandas as pd
import numpy as np
import scipy.stats as stats

# static variables 
alpha = 0.05  # significance level
cl = 'high'  # default confidence level (high, medium, or low) -  impacts sample size minimum
base_fp = input('Base filepath: ')  # filepath of folder containing read/write data (ends in /)
input_csv = input('Input CSV name: ')  # read csv name
output_csv = input('Output CSV name: ')  # write csv name
member_cols = [  # set of columns pertaining to member details - not event data
    'designer_id', 
    'created_at', 
    'canceled_at', 
    'canceled_1_hour', 
    'canceled_1_day', 
    'canceled_1_week'
    ]
outcome_col = 'canceled_1_week'  # outcome for univariate tests

def get_sample_size_min(confidence_level, cancel_rate):
    '''
        determine the minimum frequency of an event to consider it in the analysis.
        low confidence is 75%, medium confidence is 85%. we default to high confidence (95%) 
        if no valid level is given
        params:
            confidence_level: string to determine how conservative we should be - 
                              low = least conservative (options (high, low, medium))
            cancel_rate: float of what % of members in this population have canceled
            
        returns:
            sample_size_min: int minimum frequency of an event to include it in analysis
        
    '''
    # sample size minimum calculator:
    if confidence_level == 'low':
        sample_size_min = round((1.15**2 * cancel_rate * (1 - cancel_rate)) / .05**2)
    
    elif confidence_level == 'medium':
        sample_size_min = round((1.44**2 * cancel_rate * (1 - cancel_rate)) / .05**2)
        
    else:
        sample_size_min = round((1.96**2 * cancel_rate * (1 - cancel_rate)) / .05**2)
        
    return sample_size_min

def test_independence(event, outcome, full_event_df):
    '''
        run a chi-squared test of independence for a single event vs. outcome.
        params:
            event - string name for events column
            outcome - string name for outcome column
            event_df - pandas df containing all events and outcomes
        returns:
            contingency table of intervention and outcome
            chi2 value
            p-value
            dof (degrees of freedom)
            result (string description of result [i.e. whether it is significant])
    '''
    # drop any nulls; if null, the member predates the action/event, 
    # and it should not factor into their retention
    event_df = full_event_df[[event, outcome]].dropna()
    event_ct = pd.crosstab(full_event_df[event],
                           full_event_df[outcome],
                           margins=False)
    
    # run chi squared test for independence
    chi2, p_chi, dof, expected = stats.chi2_contingency(event_ct.values, lambda_="log-likelihood")
    oddsr, p_fisher = stats.fisher_exact(event_ct.values)
    

    return event_ct, chi2, p_chi, p_fisher, dof

def run_all_interventions(full_event_df, event_set, outcome):
    '''
        iterate through all events and run a chi-squared test of independence for each one vs. the outcome
        params:
            full_event_df - pandas df containing all events and outcomes
            event - list of str names for all event columns
            outcome - string name for outcome column
        returns:
            result_dict - dictionary containing 
            contingency table of intervention and outcome
            chi2 value
            p-value
            dof (degrees of freedom)
            result (string description of result [i.e. whether it is significant])
    '''
    # run chi2 for each intervention
    result_dict = {}
    for i in event_set:
        event_ct, chi2, p_chi, p_fisher, dof = test_independence(i, 
                                                                 outcome, 
                                                                 full_event_df)
        no_event_population = (event_ct.values[0][0] +
                                event_ct.values[0][1])
        no_event_conversion = (event_ct.values[0][1] /
                               no_event_population)
        
        event_population = (event_ct.values[1][0] +
                             event_ct.values[1][1])
        event_conversion = (event_ct.values[1][1] /
                            event_population)

        result_dict[i] = {'lowest_p_value': min(p_chi, p_fisher),
                          'p_chi_value': p_chi,
                          'p_fisher_value': p_fisher,
                          'contingency_table': event_ct.values,
                          'no_event_population': no_event_population,
                          'event_population': event_population,
                          'no_event_cancel_rate': no_event_conversion,
                          'event_cancel_rate': event_conversion,
                          'effect_size': no_event_conversion - event_conversion,
                          'absolute_effect_size': abs(no_event_conversion - event_conversion),
                          'statistically_significant': 1 if min(p_chi, p_fisher) <= alpha else 0
                         }
        
    return pd.DataFrame(result_dict).T.rename_axis('event').reset_index()

# read in DF, separate member details from event flags
members_and_events = pd.read_csv(base_fp + input_csv)
events = members_and_events.drop(member_cols, axis=1)

# log interim metrics
print('total members: {}'.format(len(members_and_events)))
cancel_rate = sum(members_and_events.canceled_1_week) / len(members_and_events)
print('overall cancel rate: {}'.format(cancel_rate))

# filter out events that occur too infrequently
sample_size_min = get_sample_size_min(cl, cancel_rate)
keep_event_cols = list({k:v for k,v in dict(events.sum()).items() if (v > sample_size_min)}.keys())
keep_cols = member_cols + keep_event_cols
members_and_events = members_and_events[keep_cols]

# run chi2 for each intervention
results = run_all_interventions(members_and_events,
                                keep_event_cols, 
                                outcome_col)

# save results to csv
results.to_csv(base_fp + output_csv, index=False)
