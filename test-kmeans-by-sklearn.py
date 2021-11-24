import datetime
import time
import numpy as np
import pandas as pd
from pandas import Timedelta
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("OnlineRetail.csv")
print(df.agg(['min','max']))
print(df.info())
#######################################################
df["TotalPrice"]=df["Quantity"]*df["UnitPrice"] -200 # remove noise data
df["InvoiceDate"]=pd.to_datetime(df["InvoiceDate"],format="%m/%d/%Y %H:%M")
# print(df.head())
#######################################################
df=df[(df['Country']=='United Kingdom')
      & (df['TotalPrice']<300)
      & (df['TotalPrice']>0)
      ] # remove noise data
#######################################################
df2 = df[["CustomerID","InvoiceDate","TotalPrice"]]
# df2 = df[["CustomerID","InvoiceDatetime","TotalPrice"]]
# df2["Duration"] = df2["InvoiceDate"].apply(lambda d:((datetime.date.today()-datetime.datetime.strptime(d,"%m/%d/%Y %H:%M").date()).days))

# sd = datetime.datetime.today()
sd = datetime.datetime(2011,9,30) # remove noise data
pd.options.mode.chained_assignment = None  # default='warn'
df2['Duration']= sd - df2['InvoiceDate']
df2['Duration'].astype('timedelta64[D]')
# df2['Duration'].astype('int')
df2['Duration']=round(df2['Duration'] / np.timedelta64(1, 'D'))
print(df2.agg(['min','max']))
print(df2.head())
#######################################################
plt.plot(df2['TotalPrice'])
plt.title('TotalPrice')
plt.show()
plt.plot(df2['Duration'])
plt.title('Duration')
plt.show()
#######################################################
# rfm=df2.groupby("CustomerID").agg({"Duration": "min", "CustomerID": "count", "TotalPrice": "sum"})
rfm = df2.groupby('CustomerID').agg({'Duration': lambda x:x.min(), # Recency
                                          'CustomerID': lambda x: len(x),               # Frequency
                                          'TotalPrice': lambda x: x.sum()})          # Monetary Value
rfm.rename(columns={"Duration":"Recency","CustomerID":"Frequency","TotalPrice":"Monetary"},inplace=True)
print(rfm.head())
# print(rfm.info())
#######################################################
rfm = rfm[
    (rfm['Frequency']<50)
    & (rfm['Recency']>0)
] # remove noise data
print(rfm.info())
print(rfm.agg(['min','max']))
#######################################################
plt.plot(rfm['Recency'])
plt.title('Recency')
plt.show()
plt.plot(rfm['Frequency'])
plt.title('Frequency')
plt.show()
plt.plot(rfm['Monetary'])
plt.title('Monetary')
plt.show()
#######################################################
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
model_r = kmeans.fit_predict(rfm[['Recency']])
model_f = kmeans.fit_predict(rfm[['Frequency']])
model_m = kmeans.fit_predict(rfm[['Monetary']])
rfm['R']=model_r
rfm['F']=model_f
rfm['M']=model_m
rfm['score']=rfm['R'].astype(str)+rfm['F'].astype(str)+rfm['M'].astype(str)
# print(rfm[rfm['R']==1].head)
# print(rfm.loc[12346.0:'CustomerID'])
# print(rfm[ (rfm['CustomerID']==12346) | (rfm['CustomerID']==12347)  ])
print(rfm.head())

#######################################################
# plt.plot(rfm[['R']])
plt.plot(rfm.groupby('R').agg('count'))
plt.title('R')
plt.show()
plt.plot(rfm.groupby('F').agg('count'))
plt.title('F')
plt.show()
plt.plot(rfm.groupby('M').agg('count'))
plt.title('M')
plt.show()
#######################################################
# rfm_d = rfm.groupby(['R','F','M']).agg('count')
# rfm_d = rfm.groupby(['score']).agg('count')
# rfm_d = rfm.groupby(['score']).agg({"score":"count"})
rfm_d = rfm.groupby(['R','F','M']).agg({"score":"count"})
print(rfm_d.head())
#######################################################
# use the scatterplot function to build the bubble map
# sns.scatterplot(data=rfm_d, x="gdpPercap", y="lifeExp", size="pop", legend=False, sizes=(20, 2000))
sns.scatterplot(data=rfm_d,x="R",y="F",size="score")
# plt.scatter(rfm_d[["R"]],rfm_d[["F"]],s=rfm_d[["score"]],c='Chartreuse') # not work
plt.show()
#######################################################
### ValueError: Value of 'x' is not the name of a column in 'data_frame'. Expected one of ['score'] but received: R
rfm_d.reset_index(level=[0,1,2], inplace=True)
print(rfm_d.head())
#######################################################
### not work # KeyError: 'R'
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=fig.add_subplot(121,projection='3d')
ax.scatter(rfm_d['R'],rfm_d['F'],rfm_d['M'],c='pink',s=60)
plt.show()
#######################################################
import plotly.express as px
fig = px.scatter_3d(rfm_d,x="R",y="F",z="M",size="score")
fig.update_layout(scene_zaxis_type="log")
fig.show()
#######################################################

