import pandas as pd
import numpy as np
from scipy.stats import pearsonr

class Hypothesis:
    def __init__(self, filename: str) -> None:
        """
        Class constructor.

        Args:
            filename: The name of the file containing the data set

        Returns:
            None
        """
        self.df = pd.read_csv(filename)

    def test_revenue_workers(self) -> list:
        """
        Compares the average revenue of companies with more than 500 workers to those with 500 or fewer workers.

        Returns:
            A Python list of length 3 with the average revenue of the two types of companies
            (> 500 and <= 500 workers) as well as a boolean describing whether the hypothesis is true.
        """
         
        over_500_avg = self.df[self.df['workers'] > 500]['revenue'].mean()
        under_500_avg = self.df[self.df['workers'] <= 500]['revenue'].mean()
        
        result = over_500_avg > under_500_avg
        return [over_500_avg, under_500_avg, result]
          

    def test_ca_ny_revenue(self) -> bool:
        """
        Calculates the average revenue for companies in California and New York.

        Returns:
            A boolean that describes whether there is a difference in average revenue
            between companies in California and New York.
        """
       
        ca = self.df[self.df['state_s'] == 'CA']['revenue'].mean()
        ny = self.df[self.df['state_s'] == 'NY']['revenue'].mean()
        result = ca != ny
        
        return result
        

    def test_over_10_years(self) -> list:
        """
        Calculates the average growth rate for companies that have been on the list for over 10 years
        and those that have been on the list for 10 years or fewer.

        Returns:
            A Python list of length 3 with both averages and whether the hypothesis is supported as a bool.
        """
         
        over_ten = self.df[self.df['yrs_on_list'] > 10]['growth'].mean()
        under_ten = self.df[self.df['yrs_on_list'] >= 10]['growth'].mean()

        result = over_ten < under_ten
        return [over_ten, under_ten, result]
          

    def test_revenue_variance_metro(self) -> list:
        """
        Checks the sample variance of the revenue in the "Los Angeles" metro area.

        Returns:
            A Python list containing the sample variance and a boolean representing the truth of the hypothesis.
        """
         
        la = self.df[self.df['city'] == 'Los Angeles']['revenue'].std()
        if abs(la - 1.5) <= .2:
            result = True
        result = False
        return [la, result]
          

    def test_correlation(self) -> list:
        """
        Checks the Pearson correlation coefficient between growth and revenue.

        Returns:
            A Python list with the coefficient and a boolean for the truth of the hypothesis.
        """
         
        corr = self.df['growth'].corr(self.df['revenue'])
        result = corr > 0.3
        return [corr, result]
          


import os

def evaluate_hypothesis():
    """
    Test your implementation in hypothesis.py.

    Args:
        None

    Returns:
        None
    """

    print('Test that implementation runs without errors and meets basic conditions.\n')

    hypothesis = Hypothesis(os.path.join(os.path.dirname(__file__), "dataset/companies.csv"))

    # Test test_revenue_workers
    results = hypothesis.test_revenue_workers()
    assert isinstance(results, list) and len(results) == 3, "test_revenue_workers should return a list of length 3."
    assert isinstance(results[0], (int, float)), "First element should be a numerical average revenue for companies with >500 workers."
    assert isinstance(results[1], (int, float)), "Second element should be a numerical average revenue for companies with <=500 workers."
    assert isinstance(results[2], bool) or isinstance(results[2], np.bool_), "Third element should be a boolean."
    print('Test test_revenue_workers: passed')

    # Test test_ca_ny_revenue
    result = hypothesis.test_ca_ny_revenue()
    assert isinstance(result, bool) or isinstance(result, np.bool_), "test_ca_ny_revenue should return a boolean."
    print('Test test_ca_ny_revenue: passed')

    # Test test_over_10_years
    results = hypothesis.test_over_10_years()
    assert isinstance(results, list) and len(results) == 3, "test_over_10_years should return a list of length 3."
    assert isinstance(results[0], (int, float)), "First element should be a numerical average growth rate for companies with >10 years."
    assert isinstance(results[1], (int, float)), "Second element should be a numerical average growth rate for companies with <=10 years."
    assert isinstance(results[2], bool) or isinstance(results[2], np.bool_), "Third element should be a boolean."
    print('Test test_over_10_years: passed')

    # Test test_revenue_variance_metro
    results = hypothesis.test_revenue_variance_metro()
    assert isinstance(results, list) and len(results) == 2, "test_revenue_variance_metro should return a list of length 2."
    assert isinstance(results[0], (int, float)), "First element should be the variance (a numerical value)."
    assert isinstance(results[1], bool) or isinstance(results[1], np.bool_), "Second element should be a boolean."
    print('Test test_revenue_variance_metro: passed')

    # Test test_correlation
    results = hypothesis.test_correlation()
    assert isinstance(results, list) and len(results) == 2, "test_correlation should return a list of length 2."
    assert isinstance(results[0], (int, float)), "First element should be the correlation coefficient (a numerical value)."
    assert isinstance(results[1], bool) or isinstance(results[1], np.bool_), "Second element should be a boolean."
    print('Test test_correlation: passed')

if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')
    evaluate_hypothesis()

