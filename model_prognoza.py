import itertools

from bs4 import BeautifulSoup
from pandas import DataFrame 
from pandas import read_csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options

from statsmodels.tsa.arima.model import ARIMA
from warnings import filterwarnings


#precizati numele valutei folosite - in cazul meu e 'dummy':
nume_valuta = 'Leul moldovenesc'

# generam un browser 'silently'
chrome_options = Options()
chrome_options.add_argument("--headless")
browser = webdriver.Chrome(options = chrome_options)

#navigare la pagina de interes (in cazul meu e cea de mai jos. in cazul vostru:
# https://www.cursbnr.ro/curs-valutar-bnr)
browser.get('https://www.cursbnr.ro/curs-valutar-bnr')
select = Select(browser.find_element_by_name('currency')).select_by_index(12)
start_date = browser.find_element(By.XPATH,"//input [@class='form-control input-sm']")

# stergem valoare curenta
start_date.clear()

# setam noua valoare
start_date.send_keys("27/03/2005")
browser.find_element(By.ID,"formbuton").click()

# culegel html-ul
html = browser.page_source

# parsam html-ul in cautarea tabele cu seria
soup = BeautifulSoup(html.lower(), "html.parser")

# extragem seria - gasiti id-ul tabelei voastre
lista_tr = soup.find( attrs={'class':'table-responsive'}).find('tbody').find_all('tr')
lista_dict = []

for i in range(len(lista_tr)):
    
    row_dict = {}
    lista_td = lista_tr[i].find_all('td')
    
    # in cazul meu extra coloana 1 si coloana 3, cu nume valuta si curs:       
#     row_dict['valuta'] = lista_td[0].text
#     row_dict['curs vs RON'] = lista_td[2].text
#     lista_dict.append(row_dict)
    
    # VOI trebuie sa culegeti coloanele 1 si 2, cu data si cursul istoric
    row_dict['Data'] = lista_td[0].text
    row_dict['Curs_MDL'] = lista_td[1].text
    lista_dict.append(row_dict)
 
# inchidem browserul
browser.close()

# definim seria
dfSchimb = DataFrame(lista_dict)

# salvati seria - eu comentez pentru ca  ce salvez eu e diferit de ce trebuie
#dfSchimb.to_csv('serie_valutara.csv')
filterwarnings('ignore', 'and will be removed')
# eu aici voi incarca seria lab9.csv redenumita de catre mine serie_valutara pt acest laborator
dfSchimb = read_csv('serie_valutara.csv', header = 0, parse_dates = ['Data'],infer_datetime_format=True)

#indexam dupa data cu frecventa la zi
dfSchimb = dfSchimb.set_index('Data').asfreq('D')

# umplem golurile din weekenduri si sarbatori legale
dfSchimb.Curs_MDL=dfSchimb.Curs_MDL.fillna(method="ffill")

#pregatim antrenarea
p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))

# antrenam pe tot istoricul
#Se cauta printre parametrii acea combinatie care minimizeaza AIC (Akaike Information Criterion) 

dict_list = []

# inhibam mesajul detip warning de non-convergenta - ca sa nu se afiseze 
filterwarnings('ignore', '.*failed to converge.*', )

for param in pdq:
    #for param_seasonal in seasonal_pdq:
        try:
            mod = ARIMA(dfSchimb.Curs_MDL,
                            order=param,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            results = mod.fit()
            
            # lista cu dictionare de valori; aceleasi key inseamna ca la final vom putea transforma lista in dataframe
            dict_list.append({'pdq':param,
                              #'seasonal_pdq':param_seasonal,
                             'AIC':results.aic })
            
            #print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
            
df_results = DataFrame(dict_list).sort_values(by = 'AIC', ascending = True)

# culegem cel mai bun parametru si reantrenam
mod = ARIMA(dfSchimb.Curs_MDL,
            order=df_results.pdq.values[0],

            enforce_stationarity=False,
            enforce_invertibility=False)

results = mod.fit()

#print(results.summary().tables[1])

prognoza_de_maine =  results.get_forecast(steps =1)# + timedelta(days=t)) for t in range(test_horizon)]
print('Data prognoza:',prognoza_de_maine.predicted_mean.index[0])
print('Nume valuta:', nume_valuta)
print('\nCurs prognozat: %.4f' % prognoza_de_maine.predicted_mean.values[0])

print('Interval de incredere 95%: [{} , {}]'.format( round(prognoza_de_maine.conf_int().values[0][0],4),
                                                    round(prognoza_de_maine.conf_int().values[0][1],4))) 