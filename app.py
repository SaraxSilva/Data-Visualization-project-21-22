import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import numpy as np
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objs as go
from datetime import timedelta
import plotly.express as px
import plotly.figure_factory as ff
import pickle
import zipfile
from zipfile import ZipFile

file_to_read = open("geo.pkl", "rb")
geo_world_ok = pickle.load(file_to_read)
zf = ZipFile('data/final.zip') 
spotify = pd.read_csv(zf.open('final.csv'))

spotify['date'] = pd.to_datetime(spotify['date'])
spotify['top_tracks'] = spotify['top_tracks'].str.rstrip()
spotify['country'] = spotify['country'].replace(['USA'],'United States')
spotify['country'] = spotify['country'].replace(['UK'],'United Kingdom')
df_countries = pd.DataFrame(columns = ['country'], data= spotify['country'].unique())
continent=[]
for i in df_countries['country']:
    if i in ('United States','Canada','Mexico'):
        continent.append('north america') 
    elif i in ('United Kingdom','Austria','Belgium','Switzerland','Germany','Denmark','Spain','Finland','Ireland','Italy','France','Netherlands','Norway','Poland','Portugal','Sweden','Turkey'):
        continent.append('europe')
    elif i in ('Argentina','Brazil','Chile','Colombia','Costa Rica','Ecuador','Peru'):
        continent.append('south america')
    elif i in ('Indonesia','Malaysia','Philippines','Singapore','Taiwan'):
        continent.append('asia')
    else:
        continent.append('world')
df_countries['continente']=continent
df_countries['color1']=0.57

####################################################################################################################################

app = dash.Dash(__name__)
server = app.server

############# talvez fazer aparte #################

start_date = spotify['date'].min()
end_date = spotify['date'].max()

delta = end_date - start_date  # returns timedelta
dates = []
for i in range(delta.days + 1):
    day = start_date + timedelta(days=i)
    dates.append(day)

existing_dates = spotify['date'].unique()

disabledDates = []
for i in dates:
    if i not in existing_dates:
        disabledDates.append(i)

###################################################

spotify1 = spotify.copy()


def lista(row):
    return row['artist'].split(',')


spotify1['artist'] = spotify1.apply(lambda row: lista(row), axis=1)
spotify1 = spotify1.explode('artist')
spotify1 = spotify1[~(spotify1['artist'] == ' ')]
spotify1['artist'] = spotify1['artist'].str.strip()


###################################################

def map_countries(c):
    if c == 'Global':
        return ~(df_countries['country'] == c)
    elif c == 'Singapore':
        return (df_countries['country'] == 'Malaysia')
    else:
        return (df_countries['country'] == c)


#######################################################
tab1_content = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Br(),
            dbc.Container([
                html.Label('Choose a year:'),
                html.Br(),
                dcc.Slider(id='year_slider',
                           min=2017,
                           max=2020,
                           marks={str(i): '{}'.format(str(i)) for i in [2017, 2018, 2019, 2020]},
                           value=2017,
                           step=1)
            ]),
            html.Br(),
            html.Br(),
            dbc.Container([
                html.Label('Choose a week:'),
                html.Br(),
                dcc.DatePickerSingle(id='date_picker',
                                     min_date_allowed=spotify['date'].min(),
                                     max_date_allowed=spotify['date'].max(),
                                     date=spotify['date'].min(),
                                     disabled_days=disabledDates,
                                     display_format='MMM Do')
            ]),
            html.Br(),
            html.Br(),
            dbc.Container([
                html.Label('Choose a country:'),
                html.Br(),
                dcc.Dropdown(id='dropdown_country',
                             options=[{'label': x, 'value': x} for x in df_countries['country'].unique()],
                             value='Global',
                             style= {'color': '#000000', 'background-color': '#08BA14', 'border': '3px solid #FFFFFF'}),
                html.Br(),
                dcc.Graph(id='map', figure={}),
                html.Br()
                ])
            ], width=5, className='pretty_box'),
        dbc.Col([
            dbc.Container([
                html.H5('Top 10 songs this week:'),
                html.Label('(Choose a song)'),
                html.Br(),
                dcc.Dropdown(id='music_dropdown',
                             options=[{'label': x, 'value': x} for x in spotify['top_tracks'][
                                 (spotify['country'] == 'Global') & (spotify['date'] == '2017-01-02') & (spotify['position'] <= 10)]],
                             value=spotify['top_tracks'][(spotify['country'] == 'Global') & (spotify['date'] == '2017-01-02') & (spotify['position'] <= 10)].tolist()[0],
                             style= {'color': '#000000', 'background-color': '#08BA14', 'border': '3px solid #FFFFFF'}),
                html.Br(),
                html.Div(id='music')
            ], className='pretty_box'),
            dbc.Container([
                dcc.Graph(id='bar_gra', figure={}),
                dbc.Row([
                    html.Center("For most artists having one song in the top 50 it's amazing. Now imagine having more than 1!")
                ], className='pretty_box2'),
            ], className='pretty_box')

        ], width={'size':6})
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Container([
                dcc.Graph(id='scatter_gra', figure={}),
                dbc.Row([
                    html.Center("Overall, songs with high energy levels are more likely to chart. When the energy is lower, charting songs tend to have a lower positivity as well.")
                ], className='pretty_box2'),
            ])
        ], width={'size':5, 'order':1}, className='pretty_box'),
        dbc.Col([
            html.Br(),
            dbc.Container([
                html.Center(dcc.Graph(id='sunburst_gra', figure={})),
                html.Br(),
                dbc.Row([
                    html.Center("Undoubtedly, genres preferences change overtime. Pop being the timeless favorite, Hip Hop, Rap and Latin are among the top genres.")
                ], className='pretty_box2'),
            ])
        ], width={'size':6,'order':2}, className='pretty_box')
    ])
])

tab2_content = dbc.Container([
    dbc.Row([
        dbc.Container([
            dbc.Col([
                html.Br(),
                html.Label('Choose an audio feature:'),
                dbc.RadioItems(id='audio_feat_picker',
                options=[{"label": "acousticness", "value": "acousticness"},
                         {"label": "danceability", "value": "danceability"},
                         {"label": "energy", "value": "energy"},
                         {"label": "instrumentalness", "value": "instrumentalness"},
                         {"label": "liveness", "value": "liveness"},
                         {"label": "loudness", "value": "loudness"},
                         {"label": "speechiness", "value": "speechiness"},
                         {"label": "valence", "value": "valence"},
                         {"label": "tempo", "value": "tempo"}],
                value='danceability',
                input_checked_style={"backgroundColor": "#08BA14"},
                inline=True,
                labelClassName='mr-3'),
                html.Br(),
                html.Br(),
                dcc.Graph(id='box_plots', figure={})
            ])]),
        html.Br(),
        dbc.Row(html.Br()),
        dbc.Container(id='comments'),
        html.Br(),
        dbc.Row(html.Br()),
        dbc.Row(html.Br()),
        dbc.Row(html.Br()),
        dbc.Row([
            dbc.Col(html.Div(dbc.Checklist(id='genre_checklist',
                                           options=[{'label': x, 'value': x} for x in spotify['Genre_new'].unique()],
                                           value=['latin', 'pop', 'hip hop', 'dance/electronic'],
                                           switch=True,
                                           input_checked_style={"backgroundColor": "#08BA14"})),width={'size':2}),
            dbc.Col(html.Div(dcc.Graph(id='distplot', figure={})), width={'size':10}),
            dbc.Row(dbc.Container([html.Center('How audio features are distributed within Genres? To represent and compare those distributions Kernel Density Estimation was used, so it is possible to visualize density and distribution of audio features for every genre choosen.')], className='pretty_box2'))
        ] 
    ),

    ], className='pretty_box')

])
app.layout = dbc.Container([
    html.Br(),
    dbc.Row([
        dbc.Col(html.Img(src='/assets/spotify-logo.png', height="100px"), width=1),
        dbc.Col(html.H1('Spotify: Worldwide Visual Chart Analysis',
                        className='text-center mb-4'),
                width=10),
        dbc.Col(html.Img(src='/assets/ims-logo.png', height='100px'), width=1)
    ]),
    dbc.Row([
        dbc.Col(html.H4('Get deep into music!',
                        className='text-center mb-4'),
                width=12),
        dbc.Col(html.Center('Authors: Beatriz Neto (20210608), Sara Silva (20210619), Yuriy Perezhohin (20210767)'),
                width=12),
        html.Br(),
    dbc.Row(dbc.Container([html.Center('Spotify is the most used service for music streaming all around the world, our dashboard provides visual analysis about users preferences and statistical track exploration.')], className='pretty_box2'))
    ]),
    dbc.Tabs([dbc.Tab(tab1_content, label="Chart Analysis", active_label_style={"color": "#08BA14"}),
              dbc.Tab(tab2_content, label="Audio Feature Statistics", active_label_style={"color": "#08BA14"})]),
], fluid=True)


@app.callback(
    Output('date_picker', 'min_date_allowed'),
    Output('date_picker', 'max_date_allowed'),
    Output('date_picker', 'date'),
    Input('year_slider', 'value'))
def update_datepicker(y):
    min_date_allowed = spotify[spotify['date'].dt.year == y]['date'].min()
    max_date_allowed = spotify[spotify['date'].dt.year == y]['date'].max()
    return min_date_allowed, max_date_allowed, min_date_allowed


@app.callback(
    Output('music_dropdown', 'options'),
    Output('music_dropdown', 'value'),
    Input('dropdown_country', 'value'),
    Input('date_picker', 'date'))

def update_musicdropdown(country, date):
    options = [{'label': x, 'value': x} for x in spotify['top_tracks'][(spotify['country'] == country) & (spotify['date'] == date) & (spotify['position'] <= 10)]]
    value = spotify['top_tracks'][(spotify['country'] == country) & (spotify['date'] == date) & (spotify['position'] <= 10)].tolist()[0]
    return options, value


@app.callback(Output('music', 'children'),
              Input('music_dropdown', 'value'),
              Input('dropdown_country', 'value'),
              Input('date_picker', 'date'))

def embed_iframe(value, country, date):
    mask5 = ((spotify['country'] == country) & (spotify['date'] == date) & (spotify['position'] <= 10))

    music = dict(zip(spotify['top_tracks'][mask5], spotify['uri'][mask5]))
    return html.Iframe(
        style={"borderRadius": "12px"},
        src=f"https://open.spotify.com/embed/track/{music[value]}?utm_source=generator",
        width="100%",
        height="80",
        allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture")




@app.callback(
    Output('map', 'figure'),
    Output('bar_gra', 'figure'),
    Output('scatter_gra', 'figure'),
    Output('sunburst_gra', 'figure'),
    Output('box_plots', 'figure'),
    Output('distplot', 'figure'),

    Input('dropdown_country', 'value'),
    Input('date_picker', 'date'),
    Input('audio_feat_picker', 'value'),
    Input('genre_checklist', 'value'))


def update_charts(country, date, audio_feat, genres):
    # map
    mask1 = map_countries(country)

    fig1 = px.choropleth(
        df_countries[mask1],
        geojson=geo_world_ok,
        locations='country',
        color='color1',
        range_color=[0, 1],
        color_continuous_scale='Rainbow'
    )
    fig1.update_geos(scope=df_countries.loc[df_countries['country'] == country, 'continente'].iloc[0],
                     showland=True, landcolor="#ffffff")
    fig1.update_geos(showcountries=True,
                     showocean=True, oceancolor="#424140")
    fig1.update_layout(height=400, width=490)
    fig1.update_traces(marker_line_width=0.5)
    fig1.update_layout(showlegend=False)
    fig1.update_layout(geo_bgcolor="#424140")
    fig1.update_layout(modebar_activecolor='#3aa405')
    fig1.update_layout(activeshape_fillcolor='#3aa405')
    fig1.update_layout(margin=dict(l=0, r=10, t=0, b=0), )
    fig1.update_layout({
        'plot_bgcolor': '#424140',
        'paper_bgcolor': '#424140'})
    fig1.update_layout(showlegend=False)
    fig1.update_layout(coloraxis_showscale=False)
    #####################
    #  bar chart
    df = spotify1[(spotify1['country'] == country) & (spotify1['date'] == date)]
    spotify2 = pd.DataFrame(df.groupby('artist')['title'].count().sort_values(ascending=False)).reset_index()
    spotify2['color1'] = 0.57
    fig2 = px.bar(data_frame=spotify2[spotify2['title'] >= 2], x='artist', y='title', color='color1',
                  range_color=[0, 1], color_continuous_scale='Rainbow', title='Artists with more than 1 song in the top 50')
    fig2.update_layout({
        'plot_bgcolor': '#424140',
        'paper_bgcolor': '#424140', })
    fig2.update_layout(coloraxis_showscale=False)
    fig2.update_xaxes(showgrid=False, gridcolor='#ffffff', color='#ffffff')
    fig2.update_xaxes(showline=True, linewidth=2, linecolor='#ffffff')
    fig2.update_yaxes(showline=True, linewidth=2, linecolor='#ffffff', mirror=True, color='#ffffff')
    fig2.update_xaxes(title_font=dict(size=18, family='Helvetica', color='#ffffff'))
    fig2.update_yaxes(title_font=dict(size=18, family='Helvetica', color='#ffffff'))
    fig2.update_layout(title_font=dict(family='Helvetica', size=20, color='#ffffff'), title_x=0.5, xaxis_title="Artist",
                       yaxis_title="Nº of Songs", modebar_activecolor='#3aa405')
    fig2.update_yaxes(tick0=1, dtick=1)
    ###################
    # scatter plot
    mask2 = (spotify['country'] == country) & (spotify['date'] == date)
    fig3 = px.scatter(spotify[mask2], x='valence', y='energy', color='position', color_continuous_scale='YlGn',
                      range_color=[1, 50], title='Energy vs. Positivity')
    fig3.update_xaxes(range=[0, 1])
    fig3.update_yaxes(range=[0, 1])
    fig3.update_xaxes(showgrid=False)
    fig3.update_yaxes(showgrid=False)
    fig3.update_xaxes(title_font=dict(size=18, family='Helvetica', color='#ffffff'), color='#ffffff')
    fig3.update_yaxes(title_font=dict(size=18, family='Helvetica', color='#ffffff'), color='#ffffff')
    fig3.update_layout({
        'plot_bgcolor': '#424140',
        'paper_bgcolor': '#424140', }, modebar_activecolor='#3aa405')
    fig3.update_layout(xaxis_title="Positivity", yaxis_title="Energy")
    fig3.add_trace(go.Scatter(
        x=[0.5, 0.5],
        y=[1.5, -0.5],
        line=dict(color='#ffffff', width=2),
        showlegend=False
    ))
    fig3.add_trace(go.Scatter(
        x=[-0.5, 1.5],
        y=[0.5, 0.5],
        line=dict(color='#ffffff', width=2),
        showlegend=False,
    ))
    fig3.add_annotation(x=0.12, y=0.95,
                        text="Agressive",
                        showarrow=False,
                        yshift=10,
                        font=dict(
                            family="Helvetica",
                            size=12,
                            color="#ffffff"),
                        bordercolor="#545454",
                        borderwidth=2,
                        borderpad=4,
                        opacity=0.8
                        )
    fig3.add_annotation(x=0.95, y=0.95,
                        text="Joyfull",
                        showarrow=False,
                        yshift=10,
                        font=dict(
                            family="Helvetica",
                            size=12,
                            color="#ffffff"),
                        bordercolor="#545454",
                        borderwidth=2,
                        borderpad=4,
                        opacity=0.8
                        )
    fig3.add_annotation(x=0.08, y=0.05,
                        text="Sad",
                        showarrow=False,
                        yshift=10,
                        font=dict(
                            family="Helvetica",
                            size=12,
                            color='#ffffff'),
                        bordercolor="#545454",
                        borderwidth=2,
                        borderpad=4,
                        opacity=0.8
                        )
    fig3.add_annotation(x=0.95, y=0.05,
                        text="Chill",
                        showarrow=False,
                        yshift=10,
                        font=dict(
                            family="Helvetica",
                            size=12,
                            color="#ffffff"),
                        bordercolor="#545454",
                        borderwidth=2,
                        borderpad=4,
                        opacity=0.8
                        )

    fig3.update_yaxes(nticks=10)
    fig3.update_yaxes(nticks=10)
    fig3.update_layout(title_font=dict(family='Helvetica', size=20, color='#ffffff'), title_x=0.5,
                       height=450, width=450,
                       margin=dict(l=0, r=0, t=40, b=0))
    fig3.update_yaxes(tick0=0.25, dtick=0.25)
    fig3.update_xaxes(tick0=0.25, dtick=0.25)
    fig3.update_coloraxes(colorbar_tickfont_color='#ffffff')
    fig3.update_coloraxes(colorbar_title_font_color='#ffffff')
    fig3.update_coloraxes(colorbar_title_text='Positions')
    ###############
    # sunburst plot
    mask3 = (spotify['country'] == country) & (spotify['date'] == date)
    fig4 = px.sunburst(spotify[mask3], path=["Genre_new", "Genre"], color_discrete_sequence = px.colors.sequential.YlGn,
                       title='Songs Genres per number of occurrences')
    fig4.update_layout({
        'plot_bgcolor': '#424140',
        'paper_bgcolor': '#424140', },
        title_font=dict(family='Helvetica', size=20, color='#ffffff'), title_x=0.5)
    fig4.update_layout(margin=dict(l=0, r=0, t=40, b=0), modebar_activecolor='#3aa405', height=400, width=400)

    # box plots
    fig5 = px.box(spotify, x='Genre_new', y=audio_feat, color='Genre_new', color_discrete_sequence=['#08ba14'],
                  title=f'{audio_feat}'.capitalize() + ' Box Plots Chart')
    fig5.update_layout(margin=dict(l=0, r=0, t=40, b=0), modebar_activecolor='#3aa405', showlegend=False)
    fig5.update_layout({
        'plot_bgcolor': '#424140',
        'paper_bgcolor': '#424140', },
        title_font=dict(family='Helvetica', size=20, color='#ffffff'), title_x=0.5, xaxis_title="Genre",
        yaxis_title=audio_feat.capitalize(), modebar_activecolor='#3aa405')
    fig5.update_xaxes(title_font=dict(size=18, family='Helvetica', color='#ffffff'), color='#ffffff')
    fig5.update_yaxes(title_font=dict(size=18, family='Helvetica', color='#ffffff'), color='#ffffff',
                      showgrid=True, gridcolor='#adadad')
    # displot
    hist_data = []
    group_labels = []

    for genre in genres:
        hist_data.append(np.array(spotify[spotify['Genre_new'] == genre][audio_feat]))
        group_labels.append(genre)

    fig6 = ff.create_distplot(hist_data, group_labels, bin_size=.2, histnorm='probability density', curve_type='kde',
                              show_hist=False,
                              show_rug=False)
    fig6.update_layout(margin=dict(l=0, r=0, t=40, b=0), modebar_activecolor='#3aa405', showlegend=True,title=f'{audio_feat}'.capitalize() + ' Distplot',height=600)
    fig6.update_layout({
        'plot_bgcolor': '#424140',
        'paper_bgcolor': '#424140', },
        title_font=dict(family='Helvetica', size=20, color='#ffffff'),
        xaxis_title=audio_feat.capitalize(), yaxis_title='Probability Density', modebar_activecolor='#3aa405')
    fig6.update_xaxes(title_font=dict(size=18, family='Helvetica', color='#ffffff'), color='#ffffff')
    fig6.update_yaxes(showgrid=True, gridcolor='#adadad',
                      title_font=dict(size=18, family='Helvetica', color='#ffffff'), color='#ffffff')
    fig6.update_xaxes(showgrid=False)
    fig6.update_layout(
    font_family="Helvetica",
    font_color="#ffffff",
    title_font_family="Helvetica",
    title_x=0.5,
    title_font_color="#ffffff",
    legend_title_font_color="#ffffff",
    legend_title="Genre"
)

    return fig1, fig2, fig3, fig4, fig5, fig6

@app.callback(
    Output('comments', 'children'),
    Input('audio_feat_picker', 'value')
)

def update_comment(audio_feat):
    if audio_feat == 'acousticness':
        comment = f'Jazz and Opm are the genres with higher levels of {audio_feat}. On the contrary, Metal and Boy Band have the lower.'
    elif audio_feat == 'danceability':
        comment = f'Reggae have, without doubt, the higher level of {audio_feat}. On the contrary, Metal and Opm have the lower.'
    elif audio_feat == 'energy':
        comment = f'Metal, Boy Band and K-Pop are the genres with higher levels of {audio_feat}. On the contrary, Jazz and Opm have the lower.'
    elif audio_feat == 'instrumentalness':
        comment = f'For the exception of some outliers, all the genres have lower levels of {audio_feat}.'
    elif audio_feat == 'liveness':
        comment = f'Although all the genres having lower levels of {audio_feat}, House and Reggaeton seem to have the higher levels among all.'
    elif audio_feat == 'loudness':
        comment = f'Although all the genres having negative levels of {audio_feat}, Latin, Reggaeton and Funk seem to have the higher levels among all.'
    elif audio_feat == 'speechiness':
        comment = f'Trap and Hip Hop are the genres with higher levels of {audio_feat}. On the contrary, Country, Bolero, Reggae and Jazz have the lower.'
    elif audio_feat == 'valence':
        comment = f'Reggae and Reggaeton are the genres with higher levels of {audio_feat} (positivity). On the contrary, R&B, House and Metal have the lower.'
    else:
        comment = f'R&B, Rock and Funk are the genres with higher levels of {audio_feat}. On the contrary, Country and Jazz have the lower.'

    return dbc.Row([html.Center(comment)], className='pretty_box2')


if __name__ == '__main__':
    app.run_server(debug=True,port=1337)
