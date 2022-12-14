import pandas as pd
import json
import numpy as np
from datetime import datetime
from dateutil.parser import parse
import matplotlib.pyplot as plt 
import seaborn as sns 
import matplotlib.dates as mdates

def get_bp_data(files):
    data = pd.DataFrame([], columns=['date', 'calories', 'distance', 'steps'])
    print(data)
    for i, f in enumerate(files): 
        text_data = pd.read_table(f, sep=' ')
        df = pd.DataFrame(text_data)
        if i==0:
            data['date'] = df.iloc[:,0]
            
        data.iloc[:,i+1] = df.iloc[:,1]

    print(data)
    print(data.info())
    print(data.describe())

def get_data(filename):
    fitbit_data = pd.DataFrame([],columns = ['distance', 'level', 'mets', 'calories', 'steps', 'dates'], dtype=float)
    
    df = pd.read_csv(filename)  
    data = df['data']
    
    for i in range(2, len(data)):
        ex_data = data[i]
        
        if 'Error' in ex_data:
            continue
        
        ex_data = ex_data.replace("'", '"')
        jsonData = json.loads(ex_data)
        
        d_date = jsonData['distance']['activities-distance'][0]['dateTime']
        d_distance = pd.DataFrame(jsonData['distance']['activities-distance-intraday']['dataset'])
        d_calories = pd.DataFrame(jsonData['calories']['activities-calories-intraday']['dataset'])
        d_steps = pd.DataFrame(jsonData['steps']['activities-steps-intraday']['dataset'])
     
        final = pd.merge(d_distance, d_calories, on='time')
        fitbit = pd.DataFrame(pd.merge(final, d_steps, on='time'))
        fitbit.columns = ['time','distance', 'level', 'mets', 'calories', 'steps']
        fitbit['dates'] = d_date
        fitbit.index = pd.DatetimeIndex(str(d_date) + ' ' + fitbit['time'])
        
        fitbit_data = pd.concat([fitbit_data, fitbit])
        
    return fitbit_data
  
def summarize(data):
    print(data.head(200))
    print(data.tail(200))
    print(data.describe())
    print(data.info())

    
    
def get_daily_result(data):
    daily_result = pd.DataFrame(data.groupby(['dates']).sum(), columns=['distance', 'level','mets', 'calories', 'steps'])
    print(daily_result)
    return daily_result

def visualize_daily_data(data):
    data.plot(x_compat=True, rot=90)
    plt.suptitle('2022/10/05 ~ 2022/11/12 Fitbit Data')
    plt.show()
    
def get_group_by_hour(data):
    return data.groupby(pd.Grouper(freq='60Min', base=0, label='left')).sum()

def result_by_time_series(data):
    # print(data.groupby(['dates']).count())
    # mask = data.loc['2022-10-11':'2022-10-13']
    
    arr = [data.loc['2022-10-10'], data.loc['2022-10-11'], data.loc['2022-10-12'], data.loc['2022-10-13']]
    result_arr = []
    for a in arr:
        result_arr.append(get_group_by_hour(a))
        
    visualize_for_time_series(result_arr)
    
def visualize_for_time_series(data):
    fig = plt.figure()    

    for i in range(len(data)):
        ax = fig.add_subplot(2,2,i+1)
        day_data = data[i]

        '''
        ax.plot(mdates.date2num(day_data.index), day_data['calories'], 'r--', label='calories')
        ax.plot(mdates.date2num(day_data.index), day_data['distance'], 'g--', label='distance')
        ax.plot(mdates.date2num(day_data.index), day_data['steps'], 'k--', label='steps')
        ax.plot(mdates.date2num(day_data.index), day_data['mets'], 'b--', label='mets')
        '''
        
        props={
            'title':'2022-10-1' + str(i),
            'xlabel':'Time',
            'ylabel':'Numbers'
        }
        day_data.plot(x_compat=True, rot=90)
        ax.plot(day_data)
        ax.set_xticklabels(['00:00', '03:00', '06:00', '09:00', '12:00', '15:00', '18:00', '21:00', '24:00'], rotation=90, fontsize='small')
        ax.set(**props)
        plt.savefig('00 ????????????/02 Health Data Project/data/image/fitbit_graph_2022-10-1{0}.png'.format(i), dpi=400)
    
    plt.legend(loc='best')
    plt.subplots_adjust(wspace=1, hspace=4)
    plt.show()

def visualize(data):
    fig = plt.figure()
    
    calories_ax = fig.add_subplot(2,2,1)
    steps_ax = fig.add_subplot(2,2,2)
    distance_ax = fig.add_subplot(2,2,3)
    mets_ax = fig.add_subplot(2,2,4)

    calories_ax.plot(data.index, data['calories'],'r--', label='calories')
    steps_ax.plot(data.index, data['steps'], 'g--', label='steps')
    distance_ax.plot(data.index, data['distance'], 'b--', label='distance')
    mets_ax.plot(data.index, data['mets'], 'k--', label='distance')

    calories_props={
        'xlabel': 'Date',
        'xticks':['2022-10-10', '2022-10-20', '2022-10-30', '2022-11-10', '2022-11-20'],
        'ylabel': 'calories'
    }
    steps_props={
        'xlabel': 'Date',
        'xticks':['2022-10-10', '2022-10-20', '2022-10-30', '2022-11-10', '2022-11-20'],
        'ylabel': 'steps'
    }
    distance_props={
        'xlabel': 'Date',
        'xticks':['2022-10-10', '2022-10-20', '2022-10-30', '2022-11-10', '2022-11-20'],
        'ylabel': 'distance'
    }
    mets_props={
        'xlabel': 'Date',
        'xticks':['2022-10-10', '2022-10-20', '2022-10-30', '2022-11-10', '2022-11-20'],
        'ylabel': 'mets'
    }
    
    calories_ax.set(**calories_props)
    steps_ax.set(**steps_props)
    distance_ax.set(**distance_props)
    mets_ax.set(**mets_props)
    
    plt.suptitle('Fitbit Data Graph')
    plt.show()
     
def pairplot(data):
    sns.pairplot(data, hue_order=['calories', 'steps', 'mets', 'distance'])
    plt.show()
    
def origin_data():
    filepath = '00 ????????????/02 Health Data Project/data/fitbit_datasets/1005-1122.csv'
    data = get_data(filepath)
    # summarize(data)
    # get_daily_result(data)
    # visualize_daily_data(data)
    # result_by_time_series(data)
    visualize(data)
    pairplot(data)


def main():
    origin_data()
    
if __name__ == '__main__': # main()
    main()