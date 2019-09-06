
import numpy as np
import pandas as pd
from math import log
from datetime import datetime
import time
import random

eps=10e-8

def fill_zeros(x):
    return '0'*(6-len(x))+x

class Environment:

    def __init__(self):
        self.cost = 0.0025
        # self.cost = 0.000
    def get_repo(self,start_date,end_date,codes_num,market):
        #preprocess parameters

        #read all data
        self.data=pd.read_csv(r'./data/'+market+'.csv',index_col=0,parse_dates=True,dtype=object)
        # self.data["code"]=self.data["code"].astype(str)
        if market=='China':
            self.data["code"]=self.data["code"].apply(fill_zeros)

        sample_flag=True
        while sample_flag:
            codes=random.sample(set(self.data["code"]),codes_num)
            data2=self.data.loc[self.data["code"].isin(codes)]


            date_set=set(data2.loc[data2['code']==codes[0]].index)
            for code in codes:
                date_set=date_set.intersection((set(data2.loc[data2['code']==code].index)))
            if len(date_set)>1200:
                sample_flag=False

        date_set=date_set.intersection(set(pd.date_range(start_date,end_date)))
        self.date_set = list(date_set)
        self.date_set.sort()

        train_start_time = self.date_set[0]
        train_end_time = self.date_set[int(len(self.date_set) / 6) * 5 - 1]
        test_start_time = self.date_set[int(len(self.date_set) / 6) * 5]
        test_end_time = self.date_set[-1]

        return train_start_time,train_end_time,test_start_time,test_end_time,codes

    def get_data(self,start_time,end_time,features,window_length,market,codes):
        self.codes=codes

        self.data = pd.read_csv(r'./data/' + market + '.csv', index_col=0, parse_dates=True, dtype=object)

        if market == 'China':
            self.data["Code"] = self.data["Code"].apply(fill_zeros)

        self.data[features]=self.data[features].astype(float)
        self.data=self.data[start_time.strftime("%Y-%m-%d"):end_time.strftime("%Y-%m-%d")]
        data=self.data


        #Initialize parameters
        self.M=len(codes)+1
        self.N=len(features)
        self.L=int(window_length)
        self.date_set=pd.date_range(start_time,end_time)

        asset_dict=dict()

        for asset in codes:

            asset_data=data[data["Code"]==asset].reindex(self.date_set).sort_index()#加入时间的并集，会产生缺失值pd.to_datetime(self.date_list)
            asset_data=asset_data.resample('D').mean()

            asset_data['close']=asset_data['close'].fillna(method='pad')

            base_price = asset_data.ix[-1, 'close']

            asset_dict[str(asset)]= asset_data
            asset_dict[str(asset)]['close'] = asset_dict[str(asset)]['close'] / base_price

            if 'Fed Rate' in features:
                base_fed = asset_data.ix[-1, 'Fed Rate']
                asset_dict[str(asset)]['Fed Rate'] = asset_dict[str(asset)]['Fed Rate'] / base_fed

            if 'SP500 Price' in features:
                base_SP = asset_data.ix[-1, 'SP500 Price']
                asset_dict[str(asset)]['SP500 Price']=asset_dict[str(asset)]['SP500 Price']/base_SP

            if 'Production' in features:
                base_production = asset_data.ix[-1, 'Production']
                asset_dict[str(asset)]['Production']=asset_dict[str(asset)]['Production']/base_production

            if 'Ratio' in features:
                base_ratio = asset_data.ix[-1, 'Ratio']
                asset_dict[str(asset)]['Ratio']=asset_dict[str(asset)]['Ratio']/base_ratio

            if 'Dollar Index' in features:
                base_dollar = asset_data.ix[-1, 'Dollar Index']
                asset_dict[str(asset)]['Dollar Index']=asset_dict[str(asset)]['Dollar Index']/base_dollar

            if 'Inventory' in features:
                base_inventory = asset_data.ix[-1, 'Inventory']
                asset_dict[str(asset)]['Inventory']=asset_dict[str(asset)]['Inventory']/base_inventory

            asset_data=asset_data.fillna(method='bfill',axis=1)
            asset_data=asset_data.fillna(method='ffill',axis=1)

            #***********************open as preclose*******************#
            #asset_data=asset_data.dropna(axis=0,how='any')
            asset_dict[str(asset)]=asset_data


        #tensor
        self.states=[]
        self.price_history=[]
        t =self.L+1
        length=len(self.date_set)
        while t<length-1:
            V_close = np.ones(self.L)
            if 'Fed Rate' in features:
                V_Fed=np.ones(self.L)
            if 'SP500 Price' in features:
                V_SP500=np.ones(self.L)
            if 'Dollar Index' in features:
                V_Dollar=np.ones(self.L)
            if 'Inventory' in features:
                V_Inventory=np.ones(self.L)
            if 'Production' in features:
                V_Production=np.ones(self.L)
            if 'Ratio' in features:
                V_Ratio=np.ones(self.L)


            y=np.ones(1)

            state=[]
            for asset in codes:

                asset_data=asset_dict[str(asset)]

                V_close = np.vstack((V_close, asset_data.ix[t - self.L - 1:t - 1, 'close']))

                if 'Fed Rate' in features:
                    V_Fed=np.vstack((V_Fed,asset_data.ix[t-self.L-1:t-1,'Fed Rate']))

                if 'SP500 Price' in features:
                    V_SP500=np.vstack((V_SP500,asset_data.ix[t-self.L-1:t-1,'SP500 Price']))

                if 'Dollar Index' in features:
                    V_Dollar=np.vstack((V_Dollar,asset_data.ix[t-self.L-1:t-1,'Dollar Index']))

                if 'Production' in features:
                    V_Production=np.vstack((V_Production,asset_data.ix[t-self.L-1:t-1,'Production']))

                if 'Inventory' in features:
                    V_Inventory=np.vstack((V_Inventory,asset_data.ix[t-self.L-1:t-1,'Inventory']))

                if 'Ratio' in features:
                    V_Ratio=np.vstack((V_Ratio,asset_data.ix[t-self.L-1:t-1,'Ratio']))


                y=np.vstack((y,asset_data.ix[t,'close']/asset_data.ix[t-1,'close']))

            state.append(V_close)


            if 'Fed Rate' in features:

                state.append(V_Fed)

            if 'Production' in features:

                state.append(V_Production)


            if 'SP500 Price' in features:

                state.append(V_SP500)


            if 'Dollar Index' in features:

                state.append(V_Dollar)

            if 'Inventory' in features:

                state.append(V_Inventory)

            if 'Ratio' in features:

                state.append(V_Ratio)
                # state = np.stack((state,V_Dollar), axis=2)


            state=np.stack(state,axis=1)
            state = state.reshape(1, self.M, self.L, self.N)
            self.states.append(state)
            self.price_history.append(y)
            t=t+1
        self.reset()


    def step(self,w1,w2,noise):

        if self.FLAG:
            not_terminal = 1
            price = self.price_history[self.t]

            if noise=='True':
                price[1:]=price[1:]+np.stack(np.random.normal(0,0.002,(1,len(price)-1)),axis=1)
            mu = self.cost * (np.abs(w2[0][1:] - w1[0][1:])).sum()

            risk=0
            r = (np.dot(w2, price)[0] - mu)[0]


            reward = np.log(r + eps)

            w2 = w2 / (np.dot(w2, price) + eps)
            self.t += 1
            if self.t == len(self.states):
                not_terminal = 0
                self.reset()

            price = np.squeeze(price)
            info = {'reward': reward, 'continue': not_terminal, 'next state': self.states[self.t],
                    'weight vector': w2, 'price': price,'risk':risk}
            return info
        else:
            info = {'reward': 0, 'continue': 1, 'next state': self.states[self.t],
                        'weight vector': np.array([[1] + [0 for i in range(self.M-1)]]),
                        'price': self.price_history[self.t],'risk':0}

            self.FLAG=True
            return info

    def reset(self):
        self.t=self.L+1
        self.FLAG = False

    def get_codes(self):
        return self.codes
