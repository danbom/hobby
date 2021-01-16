# hobby

https://coredot-ipynb-space.s3.amazonaws.com/1ff23f39-3d20-4c20-8743-954698c70907/20200812/10f94e3e-b1d9-4986-b63b-8ee5e2ba6e78.html?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIATBSTABCP3Y2WLI7V%2F20210115%2Fap-northeast-2%2Fs3%2Faws4_request&X-Amz-Date=20210115T145611Z&X-Amz-Expires=600&X-Amz-SignedHeaders=host&X-Amz-Signature=f470bf5895e4670a4775d1b24c85e59fb6ae8b76a9035fc048f11133d618c505

import pandas as pd
import json
import pickle
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
LoL(리그오브레전드) 데이터 분석 - 높은 티어와 낮은 티어의 차이는?
os.chdir("./data")
with open("champion.json") as f:
    champs = json.load(f)
champs=champs['data']
champNames=[x['name'] for x in champs.values()]
champDfs = {}
for cn in champNames:
    temp = pd.read_csv(f'./champ/{cn}.csv')
    champDfs[cn] = temp
dfTemp = list(champDfs.values())[0]
dfTemp['champ'] = list(champDfs.keys())[0]
데이터 요약
gameDuration : 게임 길이(초)
~Ratio : 팀내 ~ 비율
tier : 아이언(IRON) ~ 다이아몬드(DIAMOND) 총 6개 티어가 존재
lane : 실제 경기에서는 탑, 미드, 정글, 원딜, 서폿 총 5가지 포지션이 있지만 데이터에는 미드듀오, 봇 솔로 등 이상한 데이터도 일부 존재한다.
np.random.seed(2020)
dfTemp.sample(5)
gameDuration	goldRatio	killRatio	deathRatio	damageRatio	tier	lane	win	champ
1305	1922	0.256583	0.384615	0.146341	0.370749	SILVER IV	TOP SOLO	False	Aatrox
4114	1539	0.197033	0.212121	0.260870	0.178559	DIAMOND II	TOP SOLO	True	Aatrox
2445	1728	0.169127	0.052632	0.270270	0.174093	GOLD II	TOP SOLO	False	Aatrox
2744	1313	0.220003	0.214286	0.222222	0.239557	PLATINUM IV	TOP DUO	False	Aatrox
1287	1854	0.279472	0.379310	0.210526	0.302056	BRONZE I	MIDDLE DUO	True	Aatrox
Abstract
분석에는 라이엇 API에서 제공하는 데이터를 이용했으며, 2020-07-20~2020-08-03 총 2주간의 랭크게임 데이터를 사용했다. 또 각 티어별 8974개의 경기 데이터를 이용해 총 538440(8974x6x10)개의 챔피언 데이터를 사용했다.
이를통해 낮은 티어(아이언, 브론즈, 실버)와 높은 티어(골드, 플래티넘, 다이아)의 게임, 챔피언 등의 차이에 대해 살펴보고자 한다.

세부 내용
저티어와 고티어에서 챔피언 선호도, 승률의 차이
경기 시간 및 역전 가능성의 차이
기타 요소들(처음 스킬을 찍는 시간 등)의 차이
#티어별 챔피언
lowTier = ['IRON', 'BRONZE', 'SILVER']
highTier = ['GOLD', 'PLATINUM', 'DIAMOND']
byTier = []
for key,val in champDfs.items():
    temp={}
    val['tier']=val['tier'].str.split().str[0]

    valLow = val[val['tier'].isin(lowTier)]

    valHigh = val[val['tier'].isin(highTier)]
    temp = [key,len(valLow),len(valHigh),valLow['damageRatio'].mean(),valHigh['damageRatio'].mean(),
            valLow.mean()['win'],valHigh.mean()['win']]
    byTier.append(temp)
byTiDf = pd.DataFrame(byTier)

byTiDf.columns = ['champ','lowNum','highNum','lowDamageMean','highDamageMean','lowWinRatio','highWinRatio']

byTiDf['winGap'] = byTiDf['highWinRatio'] - byTiDf['lowWinRatio']

byTiDf['numGap'] = byTiDf['highNum']/byTiDf['lowNum']
가장많이 등장한 챔피언
dfTemp = byTiDf.copy()

dfTemp['lowNum'] = 10*dfTemp['lowNum']/dfTemp['lowNum'].sum()
fig = px.bar(dfTemp.sort_values("lowNum",ascending=False)[:10],x='champ',y='lowNum',
      color_discrete_sequence=['#0011aa'],opacity=0.7)
fig.update_layout(title_text="아이언4~실버1", template='plotly_white', yaxis_tickformat='%',
                 yaxis_title="등장비율")
Ezreal
Ashe
Caitlyn
Yasuo
Sylas
Sett
Lux
Ekko
Garen
Lee Sin
0%
5%
10%
15%
20%
25%
30%
35%
아이언4~실버1
champ
등장비율
dfTemp = byTiDf.copy()

dfTemp['highNum'] = 10*dfTemp['highNum']/dfTemp['highNum'].sum()
fig = px.bar(dfTemp.sort_values("highNum",ascending=False)[:10],x='champ',y='highNum',
      color_discrete_sequence=['#aa0011'],opacity=0.7)
fig.update_layout(title_text="골드4~다이아1", template='plotly_white', yaxis_tickformat='%',
                 yaxis_title="등장비율")
Ezreal
Caitlyn
Ashe
Sylas
Lee Sin
Thresh
Ekko
Sett
Kai'Sa
Karma
0%
5%
10%
15%
20%
25%
30%
35%
40%
골드4~다이아1
champ
등장비율
이즈리얼이 저티어/고티어 모두 가장 많이 픽하는 챔피언이었다. 특히 상위티어 게임에서는 100게임중 40번의 게임에 등장할정도로 빈도수가 높았다.
야스오의 경우 상위티어(18위)보다 하위티어(4위)에서 자주 등장했다.
하위티어에서 등장횟수가 높은 챔피언
dfTemp = byTiDf.sort_values("numGap").head(10).copy()
dfTemp=dfTemp.rename(columns = {"lowNum":"하위티어","highNum":"상위티어"})
fig = px.bar(dfTemp, x='champ', y=['하위티어','상위티어'])
fig.update_layout(yaxis_title="등장횟수", template='plotly_white')
Master Yi
Amumu
Teemo
Miss Fortune
Veigar
Lux
Yorick
Dr. Mundo
Udyr
Garen
0
1000
2000
3000
4000
5000
6000
7000
champ
등장횟수
x축은 (하위티어 등장비율) = (하위티어에서 등장 횟수)/(상위티어에서 등장 횟수) 의 순위이다.(왼쪽이 1등)

마스터이, 티모와 같이 재미는 있지만 팀의 승리 기여도가 비교적 낮은 챔피언(소위말해 '충' 챔피언)의 등장비율은 하위티어에서 높았다.
아무무, 문도, 가렌과 같이 난이도가 쉬운 캐릭터들도 높은 순위를 차지했다.
상위티어에 비해 하위티어에서 승률이 낮은 챔피언들
dfTemp = byTiDf.sort_values("winGap",ascending = False).head(10)
fig = go.Figure()
fig.add_trace(go.Bar(x=dfTemp['champ'], y=dfTemp['lowWinRatio'],
                    name='하위티어', marker_color='indianred'))
fig.add_trace(go.Bar(x=dfTemp['champ'], y=dfTemp['highWinRatio'],
                    name='상위티어',marker_color='lightsalmon'))

fig.update_layout(title_text="챔피언 승률",yaxis_title="승률", template='plotly_white',
                 yaxis_tickformat='%')
fig.show()
Ivern
Anivia
Qiyana
Gnar
Sivir
Nunu & Willump
Karthus
Udyr
Pantheon
Gangplank
0%
10%
20%
30%
40%
50%
챔피언 승률
승률
dfTemp = byTiDf.sort_values("winGap",ascending = False)
dfTemp=dfTemp.reset_index(drop=True)
아이번의 상위티어와 하위티어 승률차이는 11%p로 매우 높은 수치를 기록했다.
아이번, 누누, 우디르의 경우 필자의 경험상 트롤링(고의적으로 팀의 승리를 저해하는 행위)을 할때 자주 픽하는 챔피언이었으며 보통 트롤링은 하위티어에서 많이 발생하기 때문에 이와같은 행위가 승률에 영향을 줬을 것이라고 판단된다. 그래서 이를 확인해보고자 한다.
trollChamps =  byTiDf.sort_values("winGap",ascending = False).head(20)['champ'].values

tcDic = {}

for tc in trollChamps:

    troll = champDfs[tc]

    trollLow = troll[troll['tier'].isin(lowTier)]
    trollHigh = troll[troll['tier'].isin(highTier)]

    tcDic[tc] = [sum(trollLow['damageRatio']<=0.01)/len(trollLow),
                 sum(trollHigh['damageRatio']<=0.01)/len(trollHigh)]
dfTemp = pd.DataFrame(tcDic).T

dfTemp.columns = ["하위티어",'상위티어']

fig = px.bar(dfTemp)
fig.update_layout(template = "plotly_white", yaxis_title="트롤 비율",
                 yaxis_tickformat='.2%',title_text = "팀내 데미지 비율이 1%이하인 경우")
Ivern
Anivia
Qiyana
Gnar
Sivir
Nunu & Willump
Karthus
Udyr
Pantheon
Gangplank
Tristana
Evelynn
Bard
Ahri
Talon
Gragas
Rakan
Volibear
Warwick
Nidalee
0.00%
1.00%
2.00%
3.00%
4.00%
5.00%
6.00%
팀내 데미지 비율이 1%이하인 경우
index
트롤 비율
트롤링의 판단기준을 필자는 팀내 데미지 비율이 아주 낮은 경우로 규정했다. 데미지 비율이 1%이하라는 뜻은 팀원이 적 챔피언들에게 준 전체 데미지의 총합이 10만이었을때 특정 챔피언이 1000이하의 데미지를 넣었다는 것을 의미한다. 팀내 데미지 비율이 가장 낮은 타릭의 경우에도 평균 7%정도의 데미지 비중을 차지했다.

위의 barplot에서 아이번과 애니비아가 유독 트롤링이 많았음을 알 수 있다. 애니비아의 경우 W스킬인 벽으로 팀의 귀환을 끊는 등의 행위가 가능하기때문에 트롤링에 많이 사용된다고 판단된다. 이 외에 챔피언의 경우 트롤비율이 1%이하로 100경기당 1번꼴로 트롤링이 행해지며 높은 비율로 보기는 어렵다. 특히 악명높은 위상을 가진 누누나 우디르의 트롤비율은 생각보다 높지않았다.

시비르, 갱플랭크, 나르 등 챔피언이 하위티어와 상위티어에서 승률차이가 많이 나는것은 예상밖이었는데 이들은 난이도가 그렇게 어렵지도 않고 야스오와 같이 망하면 복구하기 힘든(멘탈의 영향을 많이 받는)챔피언이 아니기 때문이다. 그래서 데이터를 살펴본 결과 이들은 공통적으로 표본수가 상대적으로 적었음을 확인할 수 있었다.(상위티어, 하위티어 각각 1000개 이하/아이번, 애니비아도 포함됨)
그래서 상위티어, 하위티어에서 1000번이상 플레이된 챔피언 중에서 승률차이가 많이 나는 경우를 살펴봤다.

#dfTemp
dfTemp = byTiDf.query('lowNum>=1000 and highNum>=1000').sort_values("winGap",ascending = False).head(10)

fig = go.Figure()
fig.add_trace(go.Bar(x=dfTemp['champ'], y=dfTemp['lowWinRatio'],
                    name='하위티어', marker_color='indianred'))
fig.add_trace(go.Bar(x=dfTemp['champ'], y=dfTemp['highWinRatio'],
                    name='상위티어',marker_color='lightsalmon'))

fig.update_layout(title_text="하위티어에서 승률이 낮은 챔피언",yaxis_title="승률", template='plotly_white',
                 yaxis_tickformat='%')
fig.show()
Nunu & Willump
Karthus
Pantheon
Evelynn
Ahri
Talon
Volibear
Nidalee
Twisted Fate
Morgana
0%
10%
20%
30%
40%
50%
하위티어에서 승률이 낮은 챔피언
승률
눈에 띄는 것은 카서스, 판테온, 트페와 같이 글로벌 궁극기가 있는 챔피언이 하위티어에서 승률이 낮았다는 것이다. 이는 비교적 하위티어에서는 팀원의 상황을 체크해 적재적소에 궁극기를 잘 활용하지 못하기 때문인것으로 생각된다.

dfTemp = byTiDf.query('lowNum>=1000 and highNum>=1000').sort_values("winGap").head(10)

fig = go.Figure()
fig.add_trace(go.Bar(x=dfTemp['champ'], y=dfTemp['lowWinRatio'],
                    name='하위티어', marker_color='indianred'))
fig.add_trace(go.Bar(x=dfTemp['champ'], y=dfTemp['highWinRatio'],
                    name='상위티어',marker_color='lightsalmon'))

fig.update_layout(title_text="하위티어에서 승률이 높은 챔피언",yaxis_title="승률", template='plotly_white',
                 yaxis_tickformat='%')
fig.show()
Swain
Fizz
Xerath
Aphelios
Mordekaiser
Zac
Fiora
Yasuo
Nocturne
Jarvan IV
0%
10%
20%
30%
40%
50%
하위티어에서 승률이 높은 챔피언
승률
야스오, 피즈, 제라스, 아펠리오스 등과 같이 다루기 까다롭다고 생각되는 챔피언들 일부가 오히려 신기하게도 하위티어에서 승률이 높았다.

참고로 상위 티어에서 승률이 높은 챔피언은 바드(55%), 카서스(55%), 소나(54%)등이 있었고, 하위티어에서 승률이 높은 챔피언은 퀸(56%), 질리언(55%), 스웨인(55%) 등이 있었다.(코드참조)

#byTiDf.sort_values("highWinRatio",ascending = False)

#byTiDf.sort_values("lowWinRatio",ascending = False)
# lowDamage = np.array([])
# highDamage = np.array([])

# for val in champDfs.values():
#     low = val[val['tier'].isin(lowTier)]
#     high = val[val['tier'].isin(highTier)]
    
#     lowDamage = np.append(lowDamage,low['damageRatio'].values)
#     highDamage = np.append(highDamage,high['damageRatio'].values)

# fig = go.Figure()
# fig.add_trace(go.Histogram(x=lowDamage))
# fig.add_trace(go.Histogram(x=highDamage))

# fig.update_traces(opacity=0.55)
# fig.show()
# # 티어별 플레이타임
# playTime = {}

# for ti in tiers:
#     playTime[ti] = np.array([])

# for val in champDfs.values():
#     for ti in tiers:
#         playTime[ti] = np.append(playTime[ti],val[val['tier']==ti]['gameDuration'].values)
    

# playTiDf = pd.DataFrame(playTime)
lowTime = []
highTime = []
for val in champDfs.values():
    lowTime.extend(val[val['tier'].isin(lowTier)]['gameDuration'].tolist())
    
    highTime.extend(val[val['tier'].isin(highTier)]['gameDuration'].tolist())
    
2. 경기 시간분포 및 역전가능성
fig = go.Figure()
fig.add_trace(go.Histogram(x=lowTime,name='하위티어'))
fig.add_trace(go.Histogram(x=highTime,name='상위티어'))

fig.update_traces(opacity=0.55)
fig.update_layout(xaxis_title="경기시간(초)",template = 'plotly_white', title_text="경기시간 분포")
fig.show()
500
1000
1500
2000
2500
3000
3500
4000
4500
0
500
1000
1500
2000
경기시간 분포
경기시간(초)
평균경기시간은 하위티어가 1672초(약 28분), 상위티어가 1572초(약 26분)으로 상위티어가 좀 더 짧았다.
또한 경기시간은 900초근방, 1200초 근방에서 빈도가 급격히 늘어남을 알 수 있는데 이는 최초 서랜(항복)시간과 관련이 있어 보인다.(15분에서 만장일치 서랜 시작, 20분에서 4/1표 서랜 시작)
900초근방, 1200초 근방에서 빈도는 하위티어보다 상위티어에서 더 많은 변화를 보이며 이를통해 상위티어가 하위티어보다 서랜을 많이 침을 유추할 수 있다.
그리고 900초 미만 게임의 경우에도 상위티어의 빈도가 하위티어에 비해 많이 높은데 이는 상위티어일수록 매우 유리한게임을 빨리 끝내는법을 잘 알고있거나, 오픈(경기를 포기해 아무것도 하지 않는 것)이 많이 행해지기 때문인 것으로 생각된다.
30분이 넘어가는 경기의 경우 하위티어에서 발생빈도가 높다.

# np.mean(lowTime)

# np.mean(highTime)
from sklearn.linear_model import LogisticRegression, LinearRegression
df = pd.read_csv("match_15m_0806.csv")
df = df[df['gameDuration']>900]
np.random.seed(2020)
df=df.sample(frac=1)
df = df.groupby('tier').head(8976)
df['gold_gap'] = df['blue_gold'] - df['red_gold']
dfLow = df[df['tier'].isin(lowTier)]
dfHigh = df[df['tier'].isin(highTier)]
model = LogisticRegression(fit_intercept=False)
model.fit(df[['gold_gap']].values, df['blue_win'])


modelLow = LogisticRegression(fit_intercept=False)
modelLow.fit(dfLow[['gold_gap']].values, dfLow['blue_win'])

modelHigh = LogisticRegression(fit_intercept=False)
modelHigh.fit(dfHigh[['gold_gap']].values, dfHigh['blue_win']);
def realProb(df,x,term):
    term = term/2
    df = df.copy()
    res = df.query(f'gold_gap<{x+term} and gold_gap>={x-term}')
    return (sum(res['blue_win'])/len(res))
x = np.sort(dfLow['gold_gap'].values)
realX = list(range(-10000,10001,500))
realY = [realProb(dfLow,x,500) for x in realX]
realY_ = [realProb(dfHigh,x,500) for x in realX]
역전 가능성
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['gold_gap'],y=df['blue_win'],mode='markers', name='승패 결과(100%=승리)'))
fig.add_trace(go.Scatter(x=x, y=model.predict_proba(x.reshape(len(x),-1))[:,1],
                        mode='lines',name = '회귀곡선'))
fig.add_trace(go.Scatter(x=realX, y=realY, mode='markers', name='실제 승패 빈도'))
fig.update_layout(template = 'plotly_white',yaxis_tickformat='%' ,title_text="골드차이와 승패결과 및 로지스틱 회귀 곡선",
                 xaxis_title = "15분 골드 차이")
fig.show()
−20k
−15k
−10k
−5k
0
5k
10k
15k
20k
0%
20%
40%
60%
80%
100%
골드차이와 승패결과 및 로지스틱 회귀 곡선
15분 골드 차이
15분 골드차이와 경기 승률 분석(상위+하위티어)

파란색 점의 경우 실제 승패를 나타낸다. 골드차이가 -8000에서 8000정도 까지는 점들의 밀도가 높아 알아보기 힘들지만 골드차이가 10000이상이 되면 패배(0%)는 거의 없고 승리(100%)만 했음을 알 수 있다. 마찬가지로 골드차이가 -10000이하인 경우 대부분의 경기에서 패배했음을 알 수 있다.
초록색 점의 경우 특정 구간(예컨대 15분 골드격차가 750 ~ 1250일 때)에서 승리한 비율을 나타낸다. 이 값이 3000일때 값이 81%라는 뜻은 골드격차가 2750 ~ 3250일때 100번중 81번의 경우 승리했다는 뜻이며 확률의 관점에서 본다면 '골드격차가 3000정도일때 81%확률로 승리할 것이다' 라고 해석할 수 있다.(물론 타워, 용 등 다른 변수들도 존재하지만 골드격차만으로도 충분히 잘 적합된 회귀곡선을 얻을 수 있었다.)
빨강색 선은 로지스틱 회귀곡선을 의미하며 쉽게 말해 초록색점을 가장 잘 fitting하는 곡선임을 의미한다.(곡선은  11+e−(ax+b) 형태임. 이때 골드차이가 0이면 승률은 50%일 것이므로 b=0으로 설정했다.)
fig = go.Figure()
fig.add_trace(go.Scatter(x=realX, y=realY, mode='markers',name='하위티어'))

fig.add_trace(go.Scatter(x=realX, y=realY_, mode='markers',name='상위티어'))
fig.add_trace(go.Scatter(x=x, y=modelHigh.predict_proba(x.reshape(len(x),-1))[:,1],
                        mode='lines',name='로지스틱 회귀곡선'))
fig.update_layout(template = 'plotly_white',yaxis_tickformat='%',xaxis_title = "15분 골드차이" ,
                  title_text="상위+하위티어")

fig.show()
−15k
−10k
−5k
0
5k
10k
15k
0%
20%
40%
60%
80%
100%
상위+하위티어
15분 골드차이
예상과 달리 상위, 하위티어에서 15분 골드차이에따른 승률은 크게 차이가 없었다. 보통 3000골드차이(abs)정도가 quantile 50에 위치하는데(아래 histogram참조) 쉽게말해 100번의 경기에서 50번의 경기는 15분에 골드차이가 3000이상 난다는 뜻이다.
골드차이가 +3000일때(2750~3250) 하위티어, 상위티어 모두 승리확률은 80%정도라고 보여진다. 이는 15분, 20분서랜 등을 모두 포함하는 결과로, 만약 경기시간이 길어진다면 승률은 어떤 차이를 보일까?? 다시말해 골드격차가 좀 나더라도 항복하지않고 경기를 이어나간다면??

fig = go.Figure()
fig.add_trace(go.Histogram(x=dfLow['gold_gap'].values,name='하위티어'))
fig.add_trace(go.Histogram(x=dfHigh['gold_gap'].values,name='상위티어'))
#fig.add_trace(go.Histogram(x=playTiDf['DIAMOND'].values))
fig.update_traces(opacity=0.55)
fig.update_layout(template = 'plotly_white',xaxis_title = "15분 골드차이" ,
                  title_text="15분 골드차이 분포")
fig.show()
−15k
−10k
−5k
0
5k
10k
15k
0
100
200
300
400
500
15분 골드차이 분포
15분 골드차이
30분 이상 경기
dfLow_=dfLow[dfLow['gameDuration']>=1800]
dfHigh_=dfHigh[dfHigh['gameDuration']>=1800]
modelLow_ = LogisticRegression(fit_intercept=False)
modelLow_.fit(dfLow_[['gold_gap']].values, dfLow_['blue_win']);
modelHigh_ = LogisticRegression(fit_intercept=False)
modelHigh_.fit(dfHigh_[['gold_gap']].values, dfHigh_['blue_win']);
realX = list(range(-5000,5001,500))
realY = [realProb(dfLow_,x,500) for x in realX]
realY_ = [realProb(dfHigh_,x,500) for x in realX]
fig = go.Figure()
fig.add_trace(go.Scatter(x=realX, y=realY, mode='markers',name='하위티어'))

fig.add_trace(go.Scatter(x=realX, y=realY_, mode='markers',name='상위티어'))

fig.add_trace(go.Scatter(x=x, y=modelLow_.predict_proba(x.reshape(len(x),-1))[:,1],
                        mode='lines',name='하위티어 회귀곡선'))
fig.add_trace(go.Scatter(x=x, y=modelHigh_.predict_proba(x.reshape(len(x),-1))[:,1],
                        mode='lines',name='상위티어 회귀곡선'))

fig.update_layout(template = 'plotly_white',yaxis_tickformat='%', title_text="30분 이상 게임",
                 xaxis_title = "15분 골드차이")

fig.show()
−15k
−10k
−5k
0
5k
10k
15k
0%
20%
40%
60%
80%
100%
30분 이상 게임
15분 골드차이
30분 이상 지속된 게임에 한해서 15분 골드차이로 승률을 살펴보자.
골드차이가 5000이상(abs)나는 경우에서 점들의 분산이 커짐을 확인할 수 있는데 이는 15분 골드차이가 많이난 경우 30분이상 경기를 지속하지 못했기 때문에 표본수가 작아서 발생한 일이다.
승률의 경우 경기시간 조건이 없을 때는 3000골드의 격차가 발생했을 때 80%의 승률을 보였다면 30분 이상 지속된 게임의 경우 하위티어는 67%의 승률, 상위티어는 60%의 승률밖에 기록하지 못했다. 이는 또한 상위티어에서 역전경기가 빈번하게 발생했음을 보여준다.
5000골드의 격차의 경우에도 경기시간이 30분이상 지속된 경기에서의 역전 가능성은 하위티어에서 20%p, 상위티어에서 24%p 상승했다.
이는 당연한 결과인데, 15분당시 불리한 상황에서 단시간(15분 내)에 역전해 경기를 마무리하는 것은 장시간(15분 이후)에 역전하는것보다 빈도가 훨씬 낮기 때문이다.
참고로 이결과를 통해 단순히 '지금(15분) 골드격차가 5000이니까 게임을 30분 이상으로 억지로 끌고가보자 그러면 역전확률이 20%p 올라갈 것이다'와 같은 해석을 하는 것은 옳지 않다. '게임을 30분이상 끌고간다'라는 변수가 결과를 편향시킬 수 있기 때문이다.

첫 아이템, 스킬, 와드
with open("item_0810.pkl","rb") as f:
    firstItem = pickle.load(f)

with open("skill_0810.pkl","rb") as f:
    firstSkill = pickle.load(f)

with open("ward_0810.pkl","rb") as f:
    firstWard = pickle.load(f)### 2. 챔피언별 특징
tiers = ['IRON','BRONZE','SILVER','GOLD','PLATINUM','DIAMOND']

items = [np.mean(firstItem[x]) for x in tiers]

skills=[np.mean(firstSkill[x]) for x in tiers]

wards = [np.mean([len(x) for x in firstWard[y]]) for y in tiers]
firstDf = pd.DataFrame([tiers,items,skills,wards]).T

firstDf.columns = ['tier','item','skill','wardNum']

firstDf[['item','skill']]=firstDf[['item','skill']]/1000
firstDf
tier	item	skill	wardNum
0	IRON	14.8289	46.0553	3.18082
1	BRONZE	14.4673	49.8119	3.13525
2	SILVER	15.157	55.3914	3.22115
3	GOLD	15.4345	59.3081	3.37188
4	PLATINUM	14.8982	61.3735	3.75635
5	DIAMOND	14.0512	63.163	4.76916
마지막으로 티어별 처음 아이템 구매 시점(초)/처음 스킬을 찍은 시간(초)/2분안에 와드를 박은 개수 를 확인해보자.
처음 아이템 구매시각은 골드에서 가장 늦었고(15.4초) 다이아몬드에서 가장 빨랐다(14초). 결과를 확인하기 전에는 티어가 낮을수록 첫아이템을 늦게 살것으로 생각되었는데(티어가 올라갈수록 인베방어, 와드설치등을 중요시하기 때문에) 예상과 다른 결과였다.
처음 스킬을 찍은 시각은 확실히 티어가 올라갈수록 늦어졌는데 인베, 라인전 등 여러 상황에 따라 1레벨때 찍어야할 스킬은 다르므로 높은 티어일수록 이와같은 요소를 중요시함을 알 수 있다.
게임시작 2분동안 와드를 설치한 갯수또한 티어가 올라갈수록 많았다. 특히 아이언부터 골드까지의 격차는 크지않았지만 골드와 플래티넘, 플래티넘과 다이아에서 초반와드설치 개수차이가 큼을 알 수 있다. 하위티어 유저들은 상위티어에서 초반 와드를 설치하는 위치만 참고해도 유의미한 정보를 얻을 수 있을 것이다.
