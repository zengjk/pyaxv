from util import *


class arXiv_getter(object):
    """docstring for arXiv_getter"""

    def __init__(self):
        self.category = 'quant-ph'

    def get(self, query, category='quant-ph', max_results=1000, verbose=True):
        self.query = query
        self.category = category
        self.max_results = max_results
        df = getting_data(query, category=category, max_results=max_results, verbose=verbose)
        df = add_features(df)
        self.df = df
        return df

    def save(self, file_name='arXiv_df.csv'):
        self.df.to_csv(file_name, index=False)
        print('Saved as ' + file_name)
        
