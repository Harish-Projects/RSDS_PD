import pandas as pd
from scipy.stats import ttest_rel

def select_features(dataframe, num_samples):
    # Split the dataframe into two groups based on the number of samples
    group1 = dataframe[:num_samples]
    group2 = dataframe[num_samples:]

    # Perform t-test for each feature and store the p-values
    p_values = []
    for column in dataframe.columns:
        _, p_value = ttest_rel(group1[column], group2[column])
        p_values.append(p_value)

    # Select features with p-value greater than 0.05
    selected_features = [column for column, p_value in zip(dataframe.columns, p_values) if p_value < 0.05]

    return selected_features