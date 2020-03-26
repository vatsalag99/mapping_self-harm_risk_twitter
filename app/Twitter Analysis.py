import twint
import time 
c = twint.Config()

c = twint.Config()

for i in range (1,5):
    c.Near = "College Park"
    c.Lang = "en"
    c.Location = True
    c.Store_csv = True
    c.Since = "2019-04-14"
    c.Output = "twitter2.csv"
twint.run.Search(c)
time.sleep(1)
