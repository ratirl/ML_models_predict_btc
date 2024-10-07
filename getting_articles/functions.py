from newspaper import Article
import pandas as pd
import pandas_ta as ta
# https://finvizfinance.readthedocs.io/en/latest/#
from finvizfinance.quote import finvizfinance
from cleantext import clean
import numpy as np
from newspaper import Article
from textblob import Word
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import re
from functions import *
from newspaper import Article
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
from alpha_vantage.timeseries import TimeSeries
pd.set_option('display.max_columns', None)
import joblib

loaded_vectorizer = joblib.load('vectorizer.joblib')
loaded_model = joblib.load('sentiment_model.joblib')


def isForbiddenUrl(url):
    if 'barrons.com' in url or 'www.wsj.com' in url or 'bizjournals.com' in url or 'thestreet.com' in url:
            #print('ForbiddenUrl: Barrons')
        return True
    else:
        return False

def hasExternalLink(url):
    article = Article(url)
    article.download()
    weLookFor = '<a class="link caas-button" href='
    if 'Continue reading' in article.html:
        link = article.html.split(weLookFor)[1].split()[0]
        if (isForbiddenUrl(link)):
            return 'accessError'
        else:
            return True
    else:
        return False
    
def getExternalLink(url):
    article = Article(url)
    article.download()
    weLookFor = '<a class="link caas-button" href='
    if 'Continue reading' in article.html:
        link = article.html.split(weLookFor)[1].split()[0]
        return link
    
def return_2():
     return 2


""" badurl = 'https://finance.yahoo.com/news/alphabet-118-billion-cash-pile-092957812.html'
url = 'https://finance.yahoo.com/m/65b53896-faf4-3a06-9d0d-a63cf3c83192/best-dow-jones-stocks-to-buy.html'
test = 'https://www.barrons.com/amp/articles/anti-tesla-etf-to-close-losses-934af767'
xd = 'https://finance.yahoo.com/m/8df60ffc-6910-35e9-9d17-086864fb2907/tech-pullback-will-be.htmlhas'
print(hasExternalLink(xd)) """


def get_news_df(ticker):
    # CHANGEABLE var = stock (AAPL, NFLX, TSLA)
    stock = finvizfinance(ticker)
    news_df = stock.ticker_news()
    news_df = news_df.set_index(['Date']) # datetimeindex ipv rangeindex
    news_df.sort_index(inplace=True, ascending=True)
    news_df.attrs['Description'] = stock.ticker_description()
    news_df.attrs['Name'] = ticker
    news_df.sort_values(by='Date', ascending=False, inplace=True)
    return news_df

def get_df_name(df):
    return df.attrs['Name']

def get_df_description(df):
    return df.attrs['Description']

# two functions to get either the summary of an article or the whole text
def returnNewspaperFullText(url):
    try:
        a = Article(url)
        a.download()
        a.parse()
        return a.title + '\n' + clean(a.text, no_urls=True, no_currency_symbols=True, no_punct=True)
    except Exception as e:
        print("Error while processing URL:", url)
        print("Error message:", e)
        return ""

def returnNewspaperSummary(url):
    try:
        a = Article(url)
        a.download()
        a.parse()
        a.nlp()
        txt = a.summary
        return a.title + '\n' + clean(txt, no_urls=True, no_currency_symbols=True, no_punct=True)
    except Exception as e:
        print("Error while processing URL:", url)
        print("Error message:", e)
        return ""

# function to return the polarity
def return_polarity(summaryOrFulltextFunction):
    # now that we extracted the text with newspaper we will feed this into a textblob object
    # i will not use the correct() function as in this example is turns the word 'debuts' into 'debt' and that will have a bad sentiment for no reason
    # print(blob.correct())
    b = TextBlob(summaryOrFulltextFunction)
    return b.polarity

def getPolaritiesFromDf(aDf): #takes a df, makes a copy of it and calculates the polarities, returns a df 
    df = aDf.copy()
    for index, row in df.iterrows():
        url = row['Link']
        if (hasExternalLink(url) == True):
            print(url)
            print('return:' ,hasExternalLink(url))
            #print('Return value is:' , hasExternalLink(url))
            summary = returnNewspaperSummary(getExternalLink(url))
            fulltext = returnNewspaperFullText(getExternalLink(url))
            summary_polarity = return_polarity(summary)
            fulltext_polarity = return_polarity(fulltext)
            print('Summary polarity: ', summary_polarity)
            print('Fulltext polarity:', fulltext_polarity)
            vaderSummary = sia.polarity_scores(summary)['compound']
            vaderFulltext = sia.polarity_scores(fulltext)['compound']
            print('Vader Summary: ', vaderSummary)
            print('Vader Fulltext: ', vaderFulltext)
            df.loc[index,'PolaritySummary'] = summary_polarity
            df.loc[index,'PolarityFulltext'] = fulltext_polarity
            df.loc[index,'Summary'] = summary
            df.loc[index,'Fulltext'] = fulltext
            df.loc[index,'VaderSummary'] = vaderSummary
            df.loc[index,'VaderFulltext'] = vaderFulltext
            msg_tfidf_summary = loaded_vectorizer.transform([summary])
            msg_tfidf_fulltext = loaded_vectorizer.transform([fulltext])
            predicted_sentiment_summary = loaded_model.predict(msg_tfidf_summary)
            predicted_sentiment_fulltext = loaded_model.predict(msg_tfidf_fulltext)
            print('Model score fulltext is: ', predicted_sentiment_fulltext[0])
            print('Model score summary is: ', predicted_sentiment_summary[0])
            df.loc[index, 'model_summary'] = predicted_sentiment_summary[0]
            df.loc[index, 'model_fulltext'] = predicted_sentiment_fulltext[0]
            

            #print(getExternalLink(url))
            print('\n')
        elif (hasExternalLink(url) == 'accessError'):
            print(url)
            print(hasExternalLink(url))
            #print('return:' ,hasExternalLink(url))
            #print('return value is:' , hasExternalLink(url))
            print('no polarity due to no access')
            df.loc[index,'PolaritySummary'] = None
            df.loc[index,'PolarityFulltext'] = None
            df.loc[index,'Summary'] = None
            df.loc[index,'Fulltext'] =None
            df.loc[index,'VaderSummary'] = None
            df.loc[index,'VaderFulltext'] = None
            df.loc[index, 'model_summary'] = None
            df.loc[index, 'model_fulltext'] = None
            print('Model score is: ',  'None')
            print('\n')
        else:
            print(url)
            print('return:' ,hasExternalLink(url))
            #print('return value is:' , hasExternalLink(url))
            summary = returnNewspaperSummary(url)
            fulltext = returnNewspaperFullText(url)
            summary_polarity = return_polarity(summary)
            fulltext_polarity = return_polarity(fulltext)
            print('Summary polarity: ', summary_polarity)
            print('Fulltext polarity:', fulltext_polarity)
            vaderSummary = sia.polarity_scores(summary)['compound']
            vaderFulltext = sia.polarity_scores(fulltext)['compound']
            print('Vader Summary: ', vaderSummary)
            print('Vader Fulltext: ', vaderFulltext)
            df.loc[index,'PolaritySummary'] = summary_polarity
            df.loc[index,'PolarityFulltext'] = fulltext_polarity
            df.loc[index,'Summary'] = summary
            df.loc[index,'Fulltext'] = fulltext
            df.loc[index,'VaderSummary'] = vaderSummary
            df.loc[index,'VaderFulltext'] = vaderFulltext
            msg_tfidf_summary = loaded_vectorizer.transform([summary])
            msg_tfidf_fulltext = loaded_vectorizer.transform([fulltext])
            predicted_sentiment_summary = loaded_model.predict(msg_tfidf_summary)
            predicted_sentiment_fulltext = loaded_model.predict(msg_tfidf_fulltext)
            print('Model score fulltext is: ', predicted_sentiment_fulltext[0])
            print('Model score summary is: ', predicted_sentiment_summary[0])
            df.loc[index, 'model_summary'] = predicted_sentiment_summary[0]
            df.loc[index, 'model_fulltext'] = predicted_sentiment_fulltext[0]
            print('\n')
    return df

def get_polarity_for_ticker_on_date(ticker, begindate, enddate):
    #first we get all the news for a certain day
    #then we take the average polarity for that day
    stock = finvizfinance(ticker)
    news_df = stock.ticker_news()
    news_df = news_df.set_index(['Date']) # datetimeindex ipv rangeindex
    news_df.sort_index(inplace=True, ascending=True)
    df = news_df.loc[pd.to_datetime(begindate):pd.to_datetime(enddate)]
    return df

def get_polarity_per_day_on_df(df):
    #date_list = aDf.index.map(pd.Timestamp.date).unique() #not needed anymore
    # df = pd.DataFrame
    # for i in date_list:
    #     temp_df = aDf.loc[str(i) : str(i)]
    # return temp_df
    df = df.drop(columns=['Title', 'Link', 'Summary', 'Fulltext'])
    # using normalize to get rid of the hours per datetime.date row
    df.index.normalize()
    new_df = df.groupby(df.index.date).mean()
    return new_df

def merge_df_price_and_df_mean_polarities(df1, df2):
    return df1.merge(df2, left_index=True, right_index=True, how='outer')

def get_total_info_on_ticker(ticker, begindate, enddate):
    # 1. we get the data for a certain ticker (OHLC data)
    data_df = get_df_for_ticker_on_date(ticker, beg, end)

    # 2. we get all the news for given ticker for the dates
    news_df = get_polarity_for_ticker_on_date(ticker, beg, end)

    # 3. we calculate the polarities and Vader
    news_df = getPolaritiesFromDf(news_df)

    # 4. we take the average for each day
    news_df = get_polarity_per_day_on_df(news_df)

    # 5. we combine both df's 
    comb_df = merge_df_price_and_df_mean_polarities(data_df, news_df)
    return comb_df

def get_df_for_ticker_on_date(ticker, begindate, enddate):
    #get data for last week of ticker
    APIKEY = "XIBTEAC0WWX5ONGP"
    ts = TimeSeries(APIKEY, output_format='pandas')
    # in this function we use outputsize compact as it gives us the last 100 days and 
    # that is enough when we want a df to merge with the news since the news df 
    # will only be of the last few days
    df, meta = ts.get_daily(ticker, outputsize='compact')
    df.sort_values(by='date', ascending = True, inplace=True)
    columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df.columns = columns
    customdf = df[begindate : enddate]
    customdf.attrs['Name'] = ticker
    return customdf

def get_df_for_ticker_full(ticker):
    #get data for last week of ticker
    APIKEY = "XIBTEAC0WWX5ONGP"
    ts = TimeSeries(APIKEY, output_format='pandas')
    # in this function we use outputsize compact as it gives us the last 100 days and 
    # that is enough when we want a df to merge with the news since the news df 
    # will only be of the last few days
    df, meta = ts.get_daily(ticker, outputsize='full')
    df.sort_values(by='date', ascending = True, inplace=True)
    columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df.columns = columns
    customdf = df[begindate : enddate]
    customdf.attrs['Name'] = ticker
    return customdf

def get_polarity_for_ticker_on_date(ticker, begindate, enddate):
    #first we get all the news for a certain day
    #then we take the average polarity for that day
    stock = finvizfinance(ticker)
    news_df = stock.ticker_news()
    news_df = news_df.set_index(['Date']) # datetimeindex ipv rangeindex
    news_df.sort_index(inplace=True, ascending=True)
    df = news_df.loc[pd.to_datetime(begindate):pd.to_datetime(enddate)]
    return df

def get_intraday(ticker, interval):
    ts = TimeSeries("XIBTEAC0WWX5ONGP", output_format='pandas')
    daily, meta = ts.get_intraday(symbol=ticker, outputsize='full', interval=interval)
    daily.sort_values(by='date', ascending = True, inplace=True)
    columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    daily.columns = columns
    daily.attrs['Name'] = ticker
    return daily