import numpy as np

def Softmax(action_values, tau=1.0):
    preferences = action_values/tau
    max_preference = np.array([max(preferences[i]) for i in range(len(preferences))])
    reshaped_max_preference = max_preference.reshape((-1, 1))
    exp_preferences = np.exp(preferences-reshaped_max_preference)
    sum_of_exp_preferences = np.sum(exp_preferences,1)
    reshaped_sum_of_exp_preferences = sum_of_exp_preferences.reshape((-1, 1))
    action_probs = exp_preferences/reshaped_sum_of_exp_preferences
    action_probs = action_probs.squeeze()
    return action_probs