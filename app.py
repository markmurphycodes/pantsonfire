import pandas as pd
import numpy as np
import sklearn
import plotly.express as px
import plotly.graph_objects as go

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pickle


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])
server = app.server


# Import datasets

fire_geo_data = pd.read_csv("fire_data_combined.csv")
fire_geo_data['year'] = pd.DatetimeIndex(fire_geo_data['date']).year.astype(int)

# Import model
model = pickle.load(open("rand_forest.sav", "rb"))

# Lengths greater than 250 days are extreme outliers
fire_geo_data = fire_geo_data[fire_geo_data['len'] <= 250]

# Prepare graph

pred_data = pd.DataFrame(columns=['doy', 'pred'])

severity = px.scatter(fire_geo_data,
           x="lon",
           y="lat",
           color="class",
           size="size",
           title="GPS locations VS fire severity")

temp = px.scatter(fire_geo_data,
           x="year",
           y="tavg",
           color="class",
           size="size",
           title="Average temp VS fire severity")

length = px.box(fire_geo_data,
           x="year",
           y="len",
           color="year",
           title="concentration of fire lengths by year")

pred = px.line(pred_data,
               x='doy',
               y='pred',
               title='Risk For Selected Area Over 1 Year')




min_lat = fire_geo_data['lat'].min()
max_lat = fire_geo_data['lat'].max()
mean_lat = fire_geo_data['lat'].mean()

min_lon = fire_geo_data['lon'].min()
max_lon = fire_geo_data['lon'].max()
mean_lon = fire_geo_data['lon'].mean()


min_len = fire_geo_data['len'].min()
max_len = fire_geo_data['len'].max()
mean_len = fire_geo_data['len'].mean()


min_tavg = fire_geo_data['tavg'].min()
max_tavg = fire_geo_data['tavg'].max()
mean_tavg = fire_geo_data['tavg'].mean()


min_tmin = fire_geo_data['tmin'].min()
max_tmin = fire_geo_data['tmin'].max()
mean_tmin = fire_geo_data['tmin'].mean()


min_tmax = fire_geo_data['tmax'].min()
max_tmax = fire_geo_data['tmax'].max()
mean_tmax = fire_geo_data['tmax'].mean()


min_wavg = fire_geo_data['wavg'].min()
max_wavg = fire_geo_data['wavg'].max()
mean_wavg = fire_geo_data['wavg'].mean()


min_wmax = fire_geo_data['wmax'].min()
max_wmax = fire_geo_data['wmax'].max()
mean_wmax = fire_geo_data['wmax'].mean()



lat_slider = html.Div(children=[html.P("Lat"),
    dcc.Slider(
        id='slider-lat',
        marks= {
            min_lat: str(min_lat),
            max_lat: str(max_lat)},
        min=min_lat,
        max=max_lat,
        value=mean_lat,
        step=0.01
    )
])

lon_slider = html.Div(children=[html.P("Lon"),
    dcc.Slider(
        id='slider-lon',
        marks= {
            min_lon: str(min_lon),
            max_lon: str(max_lon)},
        min=min_lon,
        max=max_lon,
        value=mean_lon,
        step=0.01
    )
])

len_slider = html.Div(children=[html.P("Length"),
    dcc.Slider(
        id='slider-len',
        marks= {
            min_len: str(min_len),
            max_len: str(max_len)},
        min=min_len,
        max=max_len,
        value=mean_len,
        step=0.01
    )
])

tavg_slider = html.Div(children=[html.P("Average Temp"),
    dcc.Slider(
        id='slider-tavg',
        marks= {
            min_tavg: str(min_tavg),
            max_tavg: str(max_tavg)},
        min=min_tavg,
        max=max_tavg,
        value=mean_tavg,
        step=0.01
    )
])

tmin_slider = html.Div(children=[html.P("Min Temp"),
    dcc.Slider(
        id='slider-tmin',
        marks= {
            min_tmin: str(min_tmin),
            max_tmin: str(max_tmin)},
        min=min_tmin,
        max=max_tmin,
        value=mean_tmin,
        step=0.01
    )
])

tmax_slider = html.Div(children=[html.P("Max Temp"),
    dcc.Slider(
        id='slider-tmax',
        marks= {
            min_tmax: str(min_tmax),
            max_tmax: str(max_tmax)},
        min=min_tmax,
        max=max_tmax,
        value=mean_tmax,
        step=0.01
    )
])

wavg_slider = html.Div(children=[html.P("Average Wind"),
    dcc.Slider(
        id='slider-wavg',
        marks= {
            min_wavg: str(min_wavg),
            max_wavg: str(max_wavg)},
        min=min_wavg,
        max=max_wavg,
        value=mean_wavg,
        step=0.01
    )
])

wmax_slider = html.Div(children=[html.P("Max Wind"),
    dcc.Slider(
        id='slider-wmax',
        marks= {
            min_wmax: str(min_wmax),
            max_wmax: str(max_wmax)},
        min=min_wmax,
        max=max_wmax,
        value=mean_wmax,
        step=0.01
    )
])

sliders = html.Div(children=[lat_slider, lon_slider, len_slider, tavg_slider,
                             tmin_slider, tmax_slider, wavg_slider, wmax_slider], id='sliders-params')


@app.callback(
    Output('pred-graph', 'figure'),
    Input('slider-lat', 'value'),
    Input('slider-lon', 'value'),
    Input('slider-len', 'value'),
    Input('slider-tavg', 'value'),
    Input('slider-tmin', 'value'),
    Input('slider-tmax', 'value'),
    Input('slider-wavg', 'value'),
    Input('slider-wmax', 'value'))



def update_figure(lat, lon, _len, tavg, tmin, tmax, wavg, wmax):

    data = []

    for i in range(1, 365, 7):
        arr = [lat, lon, _len, tavg, tmin, tmax, wavg, wmax, i]
        
        row = model.predict([arr])[0] / 100

        data.append([i, row])

    pred_data = pd.DataFrame(data, columns=['doy', 'pred'])

    r = pred_data['pred'].max()

    pred_data['pred'] = pred_data['pred'].apply(lambda x: x/r)

    
    pred = px.line(pred_data,
               x='doy',
               y='pred',
               title='Risk For Selected Area Over 1 Year')

    pred.update_yaxes(range=[0, 1])

    pred.update_layout(transition_duration=500)

    return pred



##def display_value(lat, lon, _len, tavg, tmin, tmax, wavg, wmax):
##    
##    return html.Div([
##        html.P('lat: {}'.format(lat)), html.Div(),
##        html.P('lon: {}'.format(lon)), html.Div(),
##        html.P('length: {}'.format(_len)), html.Div(),
##        html.P('average temp: {}'.format(tavg)), html.Div(),
##        html.P('low temp: {}'.format(tmin)), html.Div(),
##        html.P('high temp: {}'.format(tmax)), html.Div(),
##        html.P('average winds: {}'.format(wavg)), html.Div(),
##        html.P('max wind speed: {}'.format(wmax))
##        ])



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


severity = dcc.Graph(
        id='severity',
        figure=severity,
    )

temp = dcc.Graph(
        id='temperature',
        figure=temp,
    )

length = dcc.Graph(
    id='length',
    figure=length,
    )

pred = dcc.Graph(
    id='pred-graph'
    )


jumbo = dbc.Jumbotron(
    [
        html.H1("Fire Analysis"),
        html.Hr(className="my-2"),
    ]
)

header = dbc.Row(dbc.Col(html.Div(jumbo), width=12))


rows = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(html.Div(severity), width="auto"),
                dbc.Col(html.Div(temp), width="auto"),
            ]
        ),
        dbc.Row(dbc.Col(html.Div(length), width=6)),
    ], style={"text-align": "center"}
)

# Tabs for data visualization and ML stuff
t_ml = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row(
                [
                dbc.Col(html.Div(children=[
                    sliders,
                    #html.Div(id='info-container', style={'margin-top': 20}),
                    ]), width=6),

                dbc.Col(pred, width=6),
                ]
    )]),
            
    className="mt-3",
)

t_vis = dbc.Card(
    dbc.CardBody(
        [
            rows
        ]
    ),
    className="mt-3",
)


tabs = dbc.Tabs(
    [
        dbc.Tab(t_ml, label="Explore Models"),
        dbc.Tab(t_vis, label="See the Dataset"),
    ]
)


app.layout = html.Div(children=[header, tabs])

if __name__ == "__main__":
    app.run_server(debug=True)
