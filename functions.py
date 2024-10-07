from newspaper import Article
import pandas as pd

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




