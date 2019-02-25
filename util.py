import warnings
from urllib3 import HTTPConnectionPool
from collections import Counter
from urllib.request import urlopen
from urllib.parse import urlencode
from bs4 import BeautifulSoup

import pandas as pd
import numpy as np


warnings.filterwarnings('ignore')


def obtaining_raw_data(query, category='quant-ph', max_results=1000, verbose=True):
    """
    This function will scrape from arXiv given a piece of query.

    Parameters
    ----------
    query : str or list
        Can be empty, '', in which case search all articles within the category.
    category : str, optional
        The arXiv category scraping from. Default value is 'quant-ph'
    max_results : int, optional
        Max number of results expected to return. Default value is 1000

    Returns
    -------
    list
        A list of Beautifulsoup object with each entry include information of one arXiv article.
    """
    arxiv_url = 'http://export.arxiv.org/api/query?'

    if isinstance(query, str):
        search_query = query.split()
    else:
        search_query = query
    search_query = ' AND '.join(search_query)

    if search_query == '':
        search_query = 'cat:' + category
    else:
        search_query = 'ti:' + search_query + ' AND cat:' + category

    headers = {'search_query': search_query,
               'sortBy': 'lastUpdatedDate',
               'sortOrder': 'descending',
               'start': 0,
               'max_results': max_results}

    encoded = urlencode(headers)
    total_url = arxiv_url + encoded
    if verbose:
        print('Fetching from ' + total_url)
    html = urlopen(total_url).read()
    html = BeautifulSoup(html)
    entries = html.find_all('entry')

    return entries


def generate_df(entries):
    """
    Convert a list of entries into pandas DataFrame
    
    Parameters
    ----------
    entries : list
        Obtained from 'obtaining_raw_data'
    
    Returns
    -------
    pandas.DataFrame
        
    """
    ids = [entry.id.contents[0][21:] for entry in entries]
    updated_dates = [entry.updated.contents[0] for entry in entries]
    published_dates = [entry.published.contents[0] for entry in entries]
    titles = [entry.title.contents[0] for entry in entries]
    summaries = [entry.summary.contents[0] for entry in entries]
    authors = [[e.contents[1].contents[0]
                for e in entry.find_all('author')] for entry in entries]

    comments = [entry.find('arxiv:comment') for entry in entries]
    comments = [com.contents[0] if com else com for com in comments]

    categories = [[e.attrs['term']
                   for e in entry.find_all('category')] for entry in entries]

    df = pd.DataFrame({'arxiv_id': ids,
                       'updated_date': updated_dates,
                       'published_date': published_dates,
                       'title': titles,
                       'summary': summaries,
                       'authors': authors,
                       'comment': comments,
                       'categories': categories})

    return df


def getting_data(query_list, category='quant-ph', max_results=1000, verbose=True):
    """
    A wrap up of data crawling functions.
    
    Parameters
    ----------
    query_list : list
        A list of queries. When performing searching, they are 'OR' to each other.
    category : str, list, optional
        A list of category, or simply one str. By default 'quant-ph'
    max_results : int, optional
        The maximum number of results returned. By default 1000.
    
    Returns
    -------
    pandas.DataFrame
        a dataframe containing all the data obtained.
    """
    dfs = []
    if isinstance(query_list, list):
        queries = query_list
    else:
        queries = [query_list]

    if isinstance(category, list):
        categories = category
    else:
        categories = [category]

    print('Start fetching...')
    
    for query in queries:
        for cat in categories:
            entries = obtaining_raw_data(
                query, category=cat, max_results=max_results, verbose=verbose)
            new_df = generate_df(entries)
            dfs.append(new_df)

    df = pd.concat(dfs)

    df = df.drop_duplicates('arxiv_id')
    df = df.reset_index()
    df = df.drop(['index'], axis=1)

    print('Totally {} entries'.format(df.shape[0]))

    return df


def getting_page(comment):
    return _helper(comment, 'page')


def getting_figure(comment):
    return _helper(comment, 'figure')


def _helper(comment, search='page'):

    if not isinstance(comment, str):
        return None

    whitelist = set(
        'abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+')

    comment = ''.join(filter(whitelist.__contains__, comment))
    comment = comment.lower()

    if search not in comment:
        return None

    words = comment.split()

    if search+'s' in words:
        ind = words.index(search+'s')
        number = words[ind-1]
    elif search in words:
        ind = words.index(search)
        number = words[ind-1]
    else:
        for word in words:
            if search in word:
                ind = word.find(search)
                number = word[:ind]
                break

    if number.isnumeric():
        return int(number)
    if '+' in number:
        nums = number.split('+')
        s = 0
        for n in nums:
            if n.isnumeric():
                s += int(n)
            elif n == '':
                continue
            else:
                return None
        return s
    else:
        return None


def add_features(df):

    df['pages'] = df['comment'].apply(getting_page)
    df['figures'] = df['comment'].apply(getting_figure)
    df['num_of_authors'] = df['authors'].apply(len)
    df['title_length'] = df['title'].apply(lambda s: len(s.split()))
    df['year_of_publishing'] = df['published_date'].apply(lambda s: int(s[:4]))
    df['month_of_publishing'] = df['published_date'].apply(lambda s: int(s[5:7]))
    df['date_of_publishing'] = df['published_date'].apply(lambda s: int(s[8:10]))

    return df


def find_prime_authors(df, threshold=2, ascending=False):

    name_list = []
    for authors in df.authors.values:
        for author in authors:
            name_list.append(author)

    names = pd.Series(dict(Counter(name_list)))
    names = names[names > threshold]

    return names.sort_values(ascending=ascending)

def name_query(df, query):
    return df[df.authors.apply(lambda authors: query.lower() in ''.join(authors).lower())]

def getting_citation_by_title(title):
    #url = 'https://scholar.google.com/scholar?hl=en&as_sdt=0%2C47&q=1901.11103&btnG='
    query = '+'.join(title.split())
    url = 'https://scholar.google.com/scholar?hl=en&as_sdt=0%2C47&q='+ query + '&btnG='
    pool = HTTPConnectionPool(url)
    http = urllib3.PoolManager()
    r = http.request('GET', url)
    html=BeautifulSoup(r.data)
    
    entries = html.find_all('div',class_="gs_ri")

    for entry in entries:
        scraped_title = entry.h3.getText()
        text = entry.getText()
        if check_title(scraped_title, title):
            if 'Cited' in text:
                text = text.split()
                indx = text.index('Cited')
                return int(text[indx+2])
            else:
                return 0
    return None

def check_title(title_scraped, title_target):
    t1 = title_scraped.lower()
    t2 = title_target.lower()
    whitelist = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    t1 = ''.join(filter(whitelist.__contains__, t1))
    t2 = ''.join(filter(whitelist.__contains__, t2))
    
    l = min(len(t1),len(t2))
    check_length = l//3
    for i in range(len(check_length)):
        if t1[i]!=t2[i]:
            return False
    return True

def getting_citation_by_arxiv_id(arxiv_id):
    
    query = arxiv_id.split('v')[0]
    url = 'https://scholar.google.com/scholar?hl=en&as_sdt=0%2C47&q='+ query + '&btnG='
    pool = HTTPConnectionPool(url)
    http = urllib3.PoolManager()
    r = http.request('GET', url)
    html=BeautifulSoup(r.data)
    if 'Why did this happen' in html.getText():
        raise Exception('Forbidden')
    entries = html.find_all('div',class_="gs_ri")
    for entry in entries:
        scraped_title = entry.h3.getText()
        text = entry.getText()
        if 'Cited' in text:
                text = text.split()
                indx = text.index('Cited')
                return int(text[indx+2])
    return 0
