import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import seaborn as sns 


def simple_graph():
    data = np.arange(10)
    print(data)
    
    plt.plot(data)
    plt.show()
    
# figure와 서브플롯
def figure_suplot():
    fig = plt.figure()
    # 서브플롯 생성
    ax1 = fig.add_subplot(2,2,1) # 2x2 크기의 4개의 서브플롯 중 첫번째 선택
    ax2 = fig.add_subplot(2,2,2) 
    ax3 = fig.add_subplot(2,2,3) 
    ax4 = fig.add_subplot(2,2,4) 
    
    plt.plot([1.5, 3.5, -2, 1.6]) # 가장 최근의 figure와 그 서브플롯을 그림 (4번 서브플롯에 생성)
    plt.plot(np.random.randn(50).cumsum(), 'k--') # 4번 플롯에 실수의 축적 합 그래프 
   
    # 히스토그램 생성
    _ = ax1.hist(np.random.randn(100), bins=20, color='k', alpha=0.3) # 100개의 실수를 20간격의 가로축에 표현
    # scatterplot 생성
    ax2.scatter(np.arange(30), np.arange(30)+3*np.random.randn(30)) # x: 0~29 정수, y: 0~29(x실수 난수)
    
    plt.show()

def subplots():
    fix, axes = plt.subplots(2,2, sharex=True, sharey=True)
    for i in range(2):
        for j in range(2):
            axes[i, j].hist(np.random.randn(500), bins=50, color='k', alpha=0.5)
    plt.subplots_adjust(wspace=2, hspace=3)
    plt.show()
    
# 색상, 마커, 선 스타일
def marker_lineplot():
    from numpy.random import randn 
    # plt.plot(randn(30).cumsum(), 'ko--') # 마커 스타일 지정
    plt.plot(randn(30).cumsum(), color='#ababab', linestyle='dashed', marker='o') # 색깔, 선 스타일, 마커 지정
    plt.show()
    
# 눈금, 라벨, 범례
def label_graph():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(np.random.randn(1000).cumsum())
    
    # x축 눈금 변경
    ticks = ax.set_xticks([0, 250, 500, 750, 1000]) # 전체 데이터 범위를 따라 눈금 배치 지정
    labels = ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'], 
                                rotation=-30, fontsize='small') # 다른 눈금 이름 라벨 지정
    ax.set_title('My Frst matplotlib plot') # 서브플롯의 제목 지정
    ax.set_xlabel('Stages') # x축에 대한 이름 지정
    
    plt.show()

# 그래프 속성 한 번에 지정하기
def graph_properties():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(np.random.randn(1000).cumsum())
    props = {
        'title': 'My First matplotlib plot',
        'xticks': [0, 250, 500, 750, 1000],
        'xticklabels': ['one', 'two', 'three', 'four', 'five'], 
        'xlabel' : 'Stages',
        'yticks' : [-10,-5,0,5,10],
        'yticklabels' : ['minus ten', 'minus five', 'zero', 'five','ten'],
        'ylabel': 'Y Numbers'
    }
    ax.set(**props)    
    plt.show()

# 그래프 범례 지정하기
def graph_legend():
    from numpy.random import randn
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    ax.plot(randn(1000).cumsum(), 'k', label='one')
    ax.plot(randn(1000).cumsum(), 'k--', label='two') # dashed
    ax.plot(randn(1000).cumsum(), 'k.', label='three') # dot
    
    ax.legend(loc = 'best') # 범례 위치 지정 
    plt.show()
    
# 주석과 그림 추가하기
def graph_with_text_picture():
    from datetime import datetime 
    
    fig = plt.figure();
    ax = fig.add_subplot(1,1,1)
    
    data = pd.read_csv('examples/spx.csv', index_col = 0, parse_dates=True)
    spx = data['SPX']
    
    spx.plot(ax=ax, style='k-')
    
    # 재정 위기 중 중요한 날짜를 주석으로 추가하기 (datetime, string 튜플 데이터 리스트)
    crisis_data = [
        (datetime(2007, 10, 11), 'Peak of bull market'),
        (datetime(2008, 3, 12), 'Bear Stearns Fails'),
        (datetime(2008, 9, 15), 'Lehman Bankruptcy')
    ]
    
    for date, label in crisis_data:
        # x,y 좌표로 지정한 위치에 라벨 추가
        ax.annotate(label, xy=(date, spx.asof(date)+75),
                    xytext=(date, spx.asof(date)+225),
                    arrowprops=dict(facecolor='black', headwidth=4, width=2, headlength=4),
                    horizontalalignment='left', verticalalignment='top')
        
    # 2007-2010 구간으로 확대 (그래프 x,y축의 시작과 끝 경계를 직접 지정)
    ax.set_xlim(['1/1/2007', '1/1/2011'])
    ax.set_ylim([600, 1800])
    
    ax.set_title('Important dates in the 2008-2009 financial crisis')
    
    # savefig() 메서드를 이용해 파일 저장 (포맷 지정 가능)
    plt.savefig('saved/financial_crisis_graph.svg')
    plt.savefig('saved/financial_crisis_graph.png', dpi=400, bbox_inches='tight')
    
    # 파일과 유사한 객체에 저장
    from io import BytesIO 
    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()    
    
    plt.show()
    
# 도형 그리기
def figures():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    rect = plt.Rectangle((0.2, 0.75), 0.4, 0.15, color='k', alpha=0.3)
    circ = plt.Circle((0.7, 0.2), 0.15, color='b', alpha=0.3)
    pgon = plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2, 0.6]], color='g', alpha=0.5)
    
    # 서브플롯에 추가하기
    ax.add_patch(rect)
    ax.add_patch(circ)
    ax.add_patch(pgon)
    
    plt.show()
    
def save_to_file():
    from io import BytesIO 
    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    
# 선 그래프 (Series 데이터)
def line_plot_series():
    # 인덱스는 0~90까지 10까지의 간격으로 설정
    s = pd.Series(np.random.randn(10).cumsum(), index=np.arange(0,100,10))
    print(s)
    s.plot() # x축이 인덱스, y축이 값
    plt.show()

# 선 그래프 (DataFrame 데이터)
def line_plot_dataframe():
    # 10행 4열의 총 40개의 실수 난수 생성(누적합으로 계산)
    df = pd.DataFrame(np.random.randn(10,4).cumsum(0),
                      columns=['A','B','C','D'],
                      index=np.arange(0,100,10))
    print(df)
    
    df.plot()
    plt.show()

# 막대 그래프
def bar_graph():
    fig, axes = plt.subplots(2,1)
    data = pd.Series(np.random.randn(16), index=list('abcdefghijklmnop'))
    
    data.plot.bar(ax=axes[0], color='k', alpha=0.7) # 수직 막대 그래프
    data.plot.barh(ax=axes[1], color='k', alpha=0.7) # 수평 막대 그래프
    
    plt.show()
    
# 막대 그래프 (dataframe)
def bar_graph_dataframe():
    df = pd.DataFrame(np.random.randn(6,4),
                      index = ['one','two','three','four','five','six'],
                      columns=pd.Index(['A','B','C','D'], name='Genus')) # 범례의 이름을 Genus로 지정
    print(df)
    # df.plot.bar()
    df.plot.barh(stacked=True, alpha=0.5) # 누적막대그래프로 표현
    plt.show()
    
# 파티 예제
def bar_grah_party_tips():
    tips = pd.read_csv('examples/tips.csv')
    party_counts = pd.crosstab(tips['day'], tips['size']) # day별 size(요일별 사이즈)를 count하여 dataframe 만들기
    
    party_counts = party_counts.loc[:, 2:5] # 1인과 6인 파티는 제외
    print(party_counts)
    
    # 각 로우의 합이 1이 되도록 정규화하고 그래프 그리기
    party_pcts = party_counts.div(party_counts.sum(1), axis=0)
    print(party_pcts)
    
    party_pcts.plot.bar()
    plt.suptitle("Party size for each day")
    # plt.show()

# seaborn으로 팁 데이터 다시 그리기
def seaborn_bar_party():  
    tips = pd.read_csv('examples/tips.csv')
    
    # tips의 비율을 나타내는 새로춘 컬럼 추가하기
    tips['tip_pct'] = tips['tip'] / (tips['total_bill'] - tips['tip'])
    print(tips.head())
     
    sns.barplot(x='tip_pct', y='day', hue='time', data=tips, orient='h')
    sns.set(style='whitegrid')
    plt.show()
    
# 전체 결제금액 대비 팁 비율을 히스토그램으로 표현
def histogram_party():
    tips = pd.read_csv('examples/tips.csv')
    tips['tip_pct'] = tips['tip'] / (tips['total_bill'] - tips['tip'])
    
    # tips['tip_pct'].plot.hist(bins=50) # 간격은 50으로 설정
    
    plt.suptitle('Ratio of Tips Compared to Total bills')
    tips['tip_pct'].plot.density() # 밀도 그래프로 표현
    
    plt.show()
    
# distplot을 이용해 히스토그램과 밀도 그래프를 한 번에 그리기
def bimodal_distribution():
    comp1 = np.random.normal(0,1,size=200)
    comp2 = np.random.normal(10,2,size=200)
    values = pd.Series(np.concatenate([comp1, comp2]))
    sns.distplot(values, bins=100, color='k')
    plt.show()
    
# 산포도 그래프
def scatter_plot():
    macro = pd.read_csv('examples/macrodata.csv')
    data = macro[['cpi', 'm1', 'tbilrate', 'unemp']]
    trans_data = np.log(data).diff().dropna()
    print(trans_data[-5:])
    
    sns.regplot(x='m1', y='unemp', data=trans_data)
    
    plt.title('Changes in log %s versus log %s' % ('m1', 'unemp'))
    
    # sns.pairplot(trans_data, diag_kind='kde', plot_kws = {'alpha':0.2})
    plt.show() 

# 패싯 그리드와 범주형 데이터
def factors_plot():
    tips = pd.read_csv('examples/tips.csv')
    tips['tip_pct'] = tips['tip'] / (tips['total_bill'] - tips['tip'])
    
    sns.catplot(x='day', y='tip_pct', row='time',
                   col='smoker',
                   kind='bar', data=tips[tips.tip_pct<1])
    plt.show()
    
# boxplot으로 중간값, 사분위, 특잇값 표현
def box_plot():
    tips = pd.read_csv('examples/tips.csv')
    tips['tip_pct'] = tips['tip'] / (tips['total_bill'] - tips['tip'])
    
    sns.catplot(x='tip_pct', y='day', kind='box',
                data=tips[tips.tip_pct<0.5])
    plt.show()
    

def main():
   # simple_graph()
   # figure_suplot()
   # subplots()
   # marker_lineplot()
   # label_graph()
   # graph_properties()
   # graph_legend()
   # graph_with_text_picture()
   # figures()
   # line_plot_series()
   # line_plot_dataframe()
   # bar_graph()
   # bar_graph_dataframe()
   # bar_grah_party_tips()
   # seaborn_bar_party()
   # histogram_party()
   # bimodal_distribution()
   # scatter_plot()
   # factors_plot()
   box_plot()

if __name__ == '__main__': # main()
    main()