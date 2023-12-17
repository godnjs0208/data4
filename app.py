#!/usr/bin/env python
# coding: utf-8

# In[10]:


import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import geopandas as gpd
import folium
from dash.exceptions import PreventUpdate
import plotly.express as px
from dash.dash_table.Format import Group
from dash import dash_table


c = gpd.read_file('data/연도별.shp')
co = c.to_crs(4326)

df_table = pd.DataFrame({
    '서울시 용도별 건물 정보': ['국가공간정보포털 국가중점데이터'],
    '서울시 침수 흔적도': ['서울 열린데이터광장'],
    '하천/용도구역': ['국가공간정보포털 오픈마켓']
})

gdf_river = gpd.read_file('data/LSMD_CONT_UJ201_11_202311.shp')

one = pd.read_csv("data/서울172022-1.csv", encoding='euc-kr')
onee = pd.read_csv("data/서울172022-2.csv", encoding='euc-kr')
oneee = pd.read_csv("data/서울172022-3.csv", encoding='euc-kr')

two = pd.read_csv("data/서울1819-1.csv", encoding='euc-kr')
twoo = pd.read_csv("data/서울1819-2.csv", encoding='euc-kr')
twooo = pd.read_csv("data/서울1819-3.csv", encoding='euc-kr')

three = pd.read_csv("data/서울22-1.csv", encoding='euc-kr')
threee = pd.read_csv("data/서울22-2.csv", encoding='euc-kr')

data = pd.concat([one, onee, oneee], axis=0)
dataa = pd.concat([two, twoo, twooo], axis=0)
dataaa = pd.concat([three, threee], axis=0)

residential_buildings = data[data['건물용도분류명'] == '주거용']
residential_buildingss = dataa[dataa['건물용도분류명'] == '주거용']
residential_buildingsss = dataaa[dataaa['건물용도분류명'] == '주거용']
year1 = residential_buildings[['Year']]
year2 = residential_buildingss[['Year']]
year3 = residential_buildingsss[['Year']]

merge = pd.merge(year1, year2, on='Year', how='outer')
merge = pd.merge(merge, year3, on='Year', how='outer')
filtered_rows = merge['Year'] != 1900
merge1 = merge[filtered_rows]

floor = residential_buildings[residential_buildings['지하층수'] > 0]
floorr = residential_buildingss[residential_buildingss['지하층수'] > 0]
floorrr = residential_buildingsss[residential_buildingsss['지하층수'] > 0]

floor1 = floor[['Year']]
floor2 = floorr[['Year']]
floor3 = floorrr[['Year']]

df_building = gpd.read_file('data/최최종point.shp')
gdf_seoul_emd = gpd.read_file('data/서울시경계.shp')
df_building['Year'] = df_building['F_DISA_NM'].str[:4].astype(int)

app = dash.Dash(__name__)
app.title = '서울시 지하주택 침수 현황'
server = app.server

app.layout = html.Div([
    html.H2('서울시 지하주택 침수 현황',
            style={'textAlign': 'center',
                   'marginBottom': 30,
                   'marginTop': 30}),

    dcc.Tabs([
        dcc.Tab(label='분석 개요', children=[
            html.Div([
                # 이미지와 텍스트, 표를 감싸는 div
                html.Div([
                    # 이미지 부분 (float: left로 설정하여 왼쪽 정렬)
                    html.Img(src='http://file3.instiz.net/data/file3/2022/08/10/7/2/e/72e57de99732b906b3da296b58f5861d.jpg',
                             width=750, height=865, style={'border-radius': '10px','border': '1px solid #ddd','margin': '20px','float': 'left'}),

                    # 텍스트와 표를 포함하는 div
                    html.Div([
                        # 텍스트 부분 (가운데 정렬)
                        html.Div([
                            html.H3('▽분석 필요성', style={'border-radius': '10px','fontSize': 36, 'fontWeight': 'bold', 'textAlign': 'center'}),
                            dcc.Markdown("""
                                    최근 기후 변화로 인하여 침수 피해가 증가하고 있음.
                                    
                                    기록적인 폭우로 많은 피해가 있었던 지난해 서울시에서는
                                    
                                    반 지하에 거주하던 일가족이 사망하는 사고까지도 발생하여
                                    
                                    따라서 앞으로의 침수 피해가 우려되는 구역을 분석하여
                                    
                                    상습침수구역을 우선적으로 개선할 수 있도록 하는 방안이 필요함.
                            """, dangerously_allow_html=True, style={'margin': '0', 'text-align': 'center', 'fontSize': 20})
                        ], style={'border-radius': '10px','border': '1px solid #ddd','margin': '20px', 'white-space': 'pre-line', 'width':'100%', 'textAlign': 'center', 'margin-left': '5px','margin-bottom': '5px'}),

                        # Title above the table
                        html.Div([ 
                            html.H3('▽데이터 가공', style={'fontSize': 36, 'fontWeight': 'bold', 'textAlign': 'center'}),
                        ], style={'margin-top': '20px'}),

                        # 표 부분
                        html.Div([
                            dash_table.DataTable(
                                id='table',
                                columns=[{'name': col, 'id': col} for col in df_table.columns],
                                data=df_table.to_dict('records'),
                                style_cell={'fontSize': 18, 'fontWeight': 'bold', 'textAlign': 'center'},
                                style_data_conditional=[
                                    {
                                        'if': {'row_index': 0},
                                        'backgroundColor': '#ff69b4',
                                        'color': 'white',
                                    }
                                ],
                            )
                        ], style={'flex': 1, 'margin-left': '20px'}),

                        # 추가 텍스트 부분
                        html.Div([
                            dcc.Markdown("""
                                    용도별 건물정보 데이터에서 주거용, 지하층수가 있는 건물 데이터를 뽑아
                                    
                                    지하주택 데이터를 만들고 서울시 침수흔적도를 위 데이터와 합쳐
                                    
                                    침수 기록이 있는 지하주택의 좌표 데이터를 만들어 이를 각 연도별로 가공함.
                                    
                                    2017년도~2022년도의 데이터를 가공한 결과 2021년도 데이터에는 해당되는 침수 주택이 없어
                                    
                                    2021년도를 제외한 총 5개년도로 분석을 진행함.
                            """)
                        ], style={'margin-top': '26px', 'text-align': 'center', 'fontSize': 20})

                    ], style={'border-radius': '10px','display': 'flex', 'flex-direction': 'column', 'justify-content': 'space-between',
                              'border': '1px solid #ddd', 'padding': '10px'})  # 테두리 스타일 추가
                ])
            ])
        ],
        selected_style={'backgroundColor': '#ff5733', 'color': 'white'}),
        
        dcc.Tab(label='연도별 분석', children=[
            html.Div([
                dcc.Dropdown(
                    id='year-dropdown',
                    options=[{'label': str(year), 'value': year} for year in sorted(df_building['Year'].unique())],
                    value=df_building['Year'].min(),
                    style={'width': '50%'}
                ),
                dcc.Graph(id='pie-chart', style={'border-radius': '10px','width': '100%', 'display': 'inline-block', 'border': '1px solid #ddd', 'padding': '10px'}),
                dcc.Graph(id='choropleth-map', style={'border-radius': '10px','width': '48%', 'display': 'inline-block', 'border': '1px solid #ddd', 'padding': '10px'}),
                dcc.Graph(id='bar-chart1', style={'border-radius': '10px','width': '48%', 'display': 'inline-block', 'border': '1px solid #ddd', 'padding': '10px'})
            ])
        ],
        selected_style={'backgroundColor': '#ff5733', 'color': 'white'}),

        dcc.Tab(label='종합 분석', children=[
            html.Div([
                html.Div([
                    dcc.Graph(id='bar-chart'),
                    html.Div(id='statistics-container')
                ], style={'border-radius': '10px','border': '1px solid #ddd', 'padding': '10px'}),  # 테두리 스타일 추가
                html.Div([
                    dcc.Graph(id='choropleth-map-left',
                              style={'border-radius': '10px','width': '50%', 'display': 'inline-block', 'border': '1px solid #ddd', 'padding': '10px'}),  # 왼쪽 단계구분도
                    dcc.Graph(id='choropleth-map-right',
                              style={'border-radius': '10px','width': '50%', 'display': 'inline-block', 'border': '1px solid #ddd', 'padding': '10px'})  # 오른쪽 단계구분도
                ], style={'border-radius': '10px','display': 'flex'})  # 테두리 스타일 추가
            ])
        ],
        selected_style={'backgroundColor': '#ff5733', 'color': 'white'}),
        
        dcc.Tab(label='분석 결과', children=[
            html.Div([
                html.Div([
                    html.H6('▽침수된 지하주택과 하천 위치의 연관성', style={'fontSize': 36, 'fontWeight': 'bold', 'textAlign': 'center'}),
                    html.Div(id='map-container', style={'width': '95%', 'padding': '10px'}),
                ], style={'border': '1px solid #ddd', 'border-radius': '10px', 'padding': '10px'}),
                
                html.Div([
                    html.Div([
                        html.H3('▽분석 결과', style={'fontSize': 36, 'fontWeight': 'bold', 'textAlign': 'center'}),
                        dcc.Markdown("""
                        분석한 결과를 보았을 때 많이 분포하는 곳을 위주로 확대해보면 주변에 하천이 있다는 것을 알 수 있었음.

                        이를 확인하기 위해, 왼쪽 지도와 같이 서울시의 하천 위치를 확인하여 포인트와 비교해 보았음.

                        왼쪽 지도에서 볼 수 있듯 포인트는 한강 중심으로 분포하여 있는 것을 확인할 수 있었으며

                        한강과 연결되는 하천 주위로 포인트들이 밀집되어 분포된 것을 볼 수 있었음.
                        """, dangerously_allow_html=True, style={'margin': '0', 'fontSize': 19}),
                    ], style={'border-radius': '10px','border': '1px solid #ddd', 'padding': '10px', 'margin-left': '5px', 'margin-bottom': '5px'}),
                    
                    html.Div([
                        html.H3('▽결론 및 제언', style={'fontSize': 36, 'fontWeight': 'bold', 'textAlign': 'center'}),
                        dcc.Markdown("""
                        서울시 내에서 한강 및 하천 인근 지역에

                        많은 침수 피해가 일어남.
                        
                        따라서 앞서 보았던 총 침수 주택 수 바 차트의 결과처럼
                        
                        영등포구, 강동구, 강북구, 도봉구를 우선적으로 개선할 수 있도록 하는 방안을 마련해야 함.
                        
                        반지하를 없애는 방안은 가장 근본적인 방안이지만 현실적으로 어려움이 큰 대안이기 때문에
                        
                        위의 분석 내용을 바탕으로 하여서 지속적으로 배수시설을 점검하고
                        
                        지하에 거주하고 있는 주민들 위한 지원정책이 필요함.
                        """, dangerously_allow_html=True, style={'margin': '0', 'fontSize': 19}),
                    ], style={'border-radius': '10px','border': '1px solid #ddd', 'padding': '10px', 'margin-left': '5px', 'margin-bottom': '5px'}),
                
                ], style={'border-radius': '10px','width': '60%', 'display': 'inline-block'}),
            
            ], style={'border-radius': '10px','display': 'flex', 'justify-content': 'space-between', 'border': '1px solid #ddd', 'margin': '20px', 'white-space': 'pre-line', 'width':'100%', 'textAlign': 'center'}),
        ],
                selected_style={'backgroundColor': '#ff5733', 'color': 'white'})
                
    ])
])

    

@app.callback(
    Output('map-container', 'children'),
    [Input('year-dropdown', 'value')]
)
def update_map(selected_year):
    if selected_year is None:
        raise PreventUpdate

    filtered_df = df_building

    m = folium.Map(location=[37.5502, 126.982], zoom_start=10.5, tiles="cartodb positron", width="100%", height="100%")
    folium.GeoJson(gdf_seoul_emd,
                   name='geojson',
                   style_function=lambda x: {'fillColor': '#00000000', 'color': 'black', 'weight': '1'},
                   tooltip=folium.GeoJsonTooltip(fields=['SGG_NM'], labels=False, sticky=True)).add_to(m)
    
    folium.GeoJson(gdf_river,
               name='river_geojson',
               style_function=lambda x: {'fillColor': '#0000FF', 'color': 'blue', 'weight': '1'},
               tooltip='River').add_to(m)

    m.get_root().html.add_child
    for i in range(filtered_df['x'].count()):
        popup_text = f"주소: {filtered_df['address'].iloc[i]}<br>지하층수: {filtered_df['U_F_NM'].iloc[i]}<br>건물용도: {filtered_df['M_P_NAM'].iloc[i]}<br>원인: {filtered_df['F_DISA_NM'].iloc[i]}"
        folium.Circle([filtered_df['y'].iloc[i], filtered_df['x'].iloc[i]],
                      popup=folium.Popup(html=popup_text, max_width=300),
                      radius=50,
                      color="red",
                      fill=True,
                      fill_color="red",
                      fill_opacity=0.4).add_to(m)
        
    return [html.Iframe(srcDoc=m._repr_html_(), width='800', height='500')]





# 연도별 파이 차트 업데이트 함수
@app.callback(
    Output('pie-chart', 'figure'),
    [Input('year-dropdown', 'value')]
)
def update_pie_chart(selected_year):
    merge1_filtered = merge1[merge1['Year'] == selected_year]
    df_count = len(merge1_filtered)
    total_count = len(merge1) - df_count
    
    ratio = df_count / total_count * 100 if total_count > 0 else 0

    fig = px.pie(names=[f'{selected_year}년 지상층만 있는 건물 비율', f'{selected_year}년 지하층이 있는 건물 비율'],
                 values=[100 - ratio, ratio],
                 title=f'{selected_year}년 지하건물 비율',
                 labels={'Names': '데이터', 'Values': '비율'},
                 color_discrete_sequence=['pink', 'brown'])

    return fig


@app.callback(
    Output('choropleth-map', 'figure'),
    [Input('year-dropdown', 'value')]
)
def update_choropleth_map(selected_year):
    co_filtered = co[co['Year'] == selected_year]
    geojson3 = co.__geo_interface__
    fig = px.choropleth_mapbox(co_filtered,
                               geojson=geojson3,
                               locations=co_filtered.index,
                               color='pop',
                               color_continuous_scale="YlOrRd",
                               mapbox_style="carto-positron",
                               zoom=10,
                               center={"lat": co_filtered.geometry.centroid.y.mean(), "lon": co_filtered.geometry.centroid.x.mean()},
                               opacity=0.5,
                               labels={'pop': 'count', 'GU_NAM': 'GU_NAM'},
                               custom_data=[co_filtered['GU_NAM']]
                               )
    fig.update_traces(hovertemplate="<b>%{customdata[0]}</b><br>침수건수: %{z}<extra></extra>")
    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center={"lat": co_filtered.geometry.centroid.y.mean(), "lon": co_filtered.geometry.centroid.x.mean()},
            zoom=9,
        ),
        mapbox_style="carto-positron",
        height=800,
        width=800,
        title=f"{selected_year}년 침수된 지하주택 단계구분도"
    )
    return fig

@app.callback(
    Output('bar-chart1', 'figure'),  # Update the Output ID to 'bar-chart'
    [Input('year-dropdown', 'value')]
)
def update_bar_chart1(selected_year):
    co_filtered = co[co['Year'] == selected_year]
    
    # Create a bar chart using px.bar
    fig = px.bar(
        co_filtered,
        x='GU_NAM',
        y='pop',
        color='pop',
        color_continuous_scale="YlOrRd",
        labels={'pop': 'count', 'GU_NAM': 'GU_NAM'},
        title=f"{selected_year}년 침수된 지하주택 수",
        hover_name='GU_NAM',  # Hover name for labels
        height=800,
        width=800,
    )

    return fig



# 통계 및 단계구분도 업데이트
# 통계 업데이트
@app.callback(
    [Output('bar-chart', 'figure'),
     Output('statistics-container', 'children')],
    [Input('year-dropdown', 'value')]
)
def update_statistics(selected_year):
    # 드랍다운 값을 사용하지 않고 전체 데이터를 사용
    statistics = df_building['GU_NAM'].value_counts()

    bar_chart = {
        'data': [{'x': statistics.index, 'y': statistics.values, 'type': 'bar', 'marker': {'color': 'red'}}],
        'layout': {'title': '총 침수 지하주택 수'}
    }

    statistics_text = f"전체 데이터 기준 총 침수 지하주택 수:\n{statistics}"

    return bar_chart, statistics_text


# 단계구분도 업데이트
@app.callback(
    [Output('choropleth-map-left', 'figure'),
     Output('choropleth-map-right', 'figure')],
    [Input('year-dropdown', 'value')]
)
def update_choropleth_map(selected_year):
    if selected_year is None:
        raise PreventUpdate

    count1 = gpd.read_file('data/영등포구없.shp')
    count2 = gpd.read_file('data/최종카운트.shp')

    # GeoDataFrame을 GeoJSON으로 변환
    geojson1 = count1.__geo_interface__
    geojson2 = count2.__geo_interface__

    # Plotly Choropleth 그리기
    fig_left = px.choropleth_mapbox(count2,
                                    geojson=geojson2,
                                    locations=count2.index,
                                    color='pop',
                                    color_continuous_scale="YlOrRd",
                                    mapbox_style="carto-positron",
                                    zoom=10,
                                    center={"lat": count2.geometry.centroid.y.mean(), "lon": count2.geometry.centroid.x.mean()},
                                    opacity=0.5,
                                    labels={'pop': 'count', 'GU_NAM': 'GU_NAM'},
                                    custom_data=[count2['GU_NAM']]
                                    )

    fig_right = px.choropleth_mapbox(count1,
                                     geojson=geojson1,
                                     locations=count1.index,
                                     color='pop',
                                     color_continuous_scale="YlOrRd",
                                     mapbox_style="carto-positron",
                                     zoom=10,
                                     center={"lat": count1.geometry.centroid.y.mean(), "lon": count1.geometry.centroid.x.mean()},
                                     opacity=0.5,
                                     labels={'pop': 'count', 'GU_NAM': 'GU_NAM'},
                                     custom_data=[count1['GU_NAM']]
                                     )

    fig_left.update_traces(hovertemplate="<b>%{customdata[0]}</b><br>침수건수: %{z}<extra></extra>")
    fig_right.update_traces(hovertemplate="<b>%{customdata[0]}</b><br>침수건수: %{z}<extra></extra>")
 
    fig_left.update_layout(        mapbox=dict(
            style="carto-positron",
            center={"lat": count2.geometry.centroid.y.mean(), "lon": count2.geometry.centroid.x.mean()},
            zoom=9,
        ),
        mapbox_style="carto-positron",
        height=600,
        width=600,
        title="침수된 지하주택 단계구분도"
                           
    )

    fig_right.update_layout(
        mapbox=dict(
            style="carto-positron",
            center={"lat": count1.geometry.centroid.y.mean(), "lon": count1.geometry.centroid.x.mean()},
            zoom=9,
        ),
        mapbox_style="carto-positron",
        height=600,
        width=600,
        title="영등포구를 제외한 침수된 지하주택 단계구분도"
    )

    return fig_left, fig_right


if __name__ == '__main__':
    app.run_server(debug=True, port=8088)


# In[2]:


get_ipython().system(' pip install gunicorn')

