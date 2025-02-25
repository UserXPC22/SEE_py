import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from statsmodels.distributions.empirical_distribution import ECDF


data = pd.read_csv("med_events.csv")


data.columns = ["pnr", "eksd", "dur_original", "perday", "ATC", "CATEGORY_L1", "CATEGORY_L2"]
data['eksd'] = pd.to_datetime(data['eksd'])

def see(arg1):
    C09CA01 = data[data['ATC'] == arg1].copy()
    print(f"Filtered data size: {C09CA01.shape}")
    Drug_see_p0 = C09CA01.copy()
    Drug_see_p1 = C09CA01.copy()
    Drug_see_p1 = Drug_see_p1.sort_values(by=['pnr', 'eksd'])
    Drug_see_p1['prev_eksd'] = Drug_see_p1.groupby('pnr')['eksd'].shift(1)
    Drug_see_p1 = Drug_see_p1.dropna()
    Drug_see_p1 = Drug_see_p1.groupby('pnr').apply(lambda x: x.sample(1)).reset_index(drop=True)
    Drug_see_p1 = Drug_see_p1[['pnr', 'eksd', 'prev_eksd']]
    Drug_see_p1['event_interval'] = (Drug_see_p1['eksd'] - Drug_see_p1['prev_eksd']).dt.days
    print(f"Event interval data size: {Drug_see_p1.shape}")
    per = ECDF(Drug_see_p1['event_interval'])
    x = Drug_see_p1['event_interval']
    y = per(x)
    dfper = pd.DataFrame({'x': x, 'y': y})
    
 
    dfper = dfper[dfper['y'] <= 0.8]
    print(f"Filtered ECDF data size: {dfper.shape}")
    ni = dfper['x'].max()
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='x', y='y', data=dfper, palette='viridis')
    plt.title("80% ECDF")
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=x, y=y, palette='viridis')
    plt.title("100% ECDF")
    plt.show()
    
    m1 = Drug_see_p1['pnr'].value_counts()
    plt.figure()
    m1.plot(kind='bar')
    plt.show()
    
    Drug_see_p2 = Drug_see_p1[Drug_see_p1['event_interval'] <= ni]
    print(f"Filtered event interval data size: {Drug_see_p2.shape}")
    
   
    Drug_see_p2 = Drug_see_p2[Drug_see_p2['event_interval'] > 0]
    
    x1, y1 = [], []  
    if len(Drug_see_p2) > 1:
        d = gaussian_kde(np.log(Drug_see_p2['event_interval']))
        x1 = np.linspace(min(np.log(Drug_see_p2['event_interval'])), max(np.log(Drug_see_p2['event_interval'])), 1000)
        y1 = d(x1)
        plt.figure()
        sns.scatterplot(x=x1, y=y1, palette='viridis')
        plt.title("Log(event interval)")
        plt.show()
    else:
        print("Not enough data points for KDE")
    
    if len(x1) > 0 and len(y1) > 0:
        a = pd.DataFrame({'x': x1, 'y': y1})
        a = (a - a.mean()) / a.std()
        
        
        kmeans = KMeans(n_clusters=2, random_state=1234)
        kmeans.fit(a)
        silhouette_avg = silhouette_score(a, kmeans.labels_)
        print(f'Silhouette Score: {silhouette_avg}')
        
       
        kmeans = KMeans(n_clusters=2, random_state=1234)
        dfper['cluster'] = kmeans.fit_predict(dfper[['x']])
        print(dfper.groupby('cluster')['x'].describe())
        
        ni2 = dfper.groupby('cluster')['x'].min().reset_index()
        ni3 = dfper.groupby('cluster')['x'].max().reset_index()
        ni4 = dfper.groupby('cluster')['x'].median().reset_index()
        ni2.columns = ['Cluster', 'Minimum']
        ni3.columns = ['Cluster', 'Maximum']
        ni4.columns = ['Cluster', 'Median']
        nif = pd.merge(ni2, ni3, on='Cluster')
        nif = pd.merge(nif, ni4, on='Cluster')
        
        results = pd.merge(Drug_see_p1, nif, left_on='pnr', right_on='Cluster', how='left')
        results['Final_cluster'] = np.where((results['event_interval'] >= results['Minimum']) & (results['event_interval'] <= results['Maximum']), results['Cluster'], np.nan)
        results = results.dropna(subset=['Final_cluster'])
        results['Median'] = np.exp(results['Median'])
        results = results[['pnr', 'Median', 'Cluster']]
        
        if not results['Cluster'].empty:
            t1 = results['Cluster'].value_counts().idxmax()
            t1 = pd.DataFrame({'Cluster': [t1]})
            t1_merged = pd.merge(t1, results, on='Cluster')
            t1_merged = t1_merged.iloc[0]
            t1 = t1_merged[['Cluster']]
        else:
            t1 = pd.DataFrame({'Cluster': [0], 'Median': [0]})
        
        Drug_see_p1 = pd.merge(Drug_see_p1, results, on='pnr', how='left')
        Drug_see_p1['Median'] = Drug_see_p1['Median'].fillna(t1['Median'])
        Drug_see_p1['Cluster'] = Drug_see_p1['Cluster'].fillna(0)
        Drug_see_p1['event_interval'] = Drug_see_p1['event_interval'].astype(float)
        Drug_see_p1['test'] = round(Drug_see_p1['event_interval'] - Drug_see_p1['Median'], 1)
        
        Drug_see_p3 = Drug_see_p1[['pnr', 'Median', 'Cluster']]
        
       
        Drug_see_p0 = pd.merge(Drug_see_p0, Drug_see_p3, on='pnr', how='left')
        Drug_see_p0['Median'] = Drug_see_p0['Median'].astype(float)
        Drug_see_p0['Median'] = Drug_see_p0['Median'].fillna(t1['Median'])
        Drug_see_p0['Cluster'] = Drug_see_p0['Cluster'].fillna(0)
    
    return Drug_see_p0

def see_assumption(arg1):
    arg1 = arg1.sort_values(by=['pnr', 'eksd'])
    arg1['prev_eksd'] = arg1.groupby('pnr')['eksd'].shift(1)
    Drug_see2 = arg1.copy()
    Drug_see2 = Drug_see2.sort_values(by=['pnr', 'eksd'])
    Drug_see2['p_number'] = Drug_see2.groupby('pnr').cumcount() + 1
    Drug_see2 = Drug_see2[Drug_see2['p_number'] >= 2]
    Drug_see2 = Drug_see2[['pnr', 'eksd', 'prev_eksd', 'p_number']]
    Drug_see2['Duration'] = (Drug_see2['eksd'] - Drug_see2['prev_eksd']).dt.days
    Drug_see2['p_number'] = Drug_see2['p_number'].astype(str)
    
    print(f"Data for scatter plot: {Drug_see2.shape}")
    print(Drug_see2.head())  
    plt.figure()
    sns.scatterplot(x='p_number', y='Duration', data=Drug_see2, palette='viridis')
    plt.show()
    
    medians_of_medians = Drug_see2.groupby('pnr')['Duration'].median().reset_index()
    print(f"Medians of medians: {medians_of_medians.shape}")
    print(medians_of_medians.head())  
    plt.figure()
    sns.scatterplot(x='p_number', y='Duration', data=Drug_see2, palette='viridis')
    plt.axhline(y=medians_of_medians['Duration'].median(), color='red', linestyle='--')
    plt.show()


medA = see("A02BC02")
medB = see("A09AA02")

see_assumption(medA)
see_assumption(medB)