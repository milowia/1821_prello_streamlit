#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sqlite3
import pandas as pd
import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery

# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)

# Perform query.
# Uses st.cache_data to only rerun when the query changes or after 10 min.
@st.cache_data(ttl=600)

def run_query(table_name):
    query = f"""
    SELECT *
    FROM `learned-raceway-436207-f6.1821_prello.{table_name}`
    """
    df = client.query(query).to_dataframe()
    return df

# List of tables to import
tables = [
    "POI_tourist_establishments",
    "POI_touristic_sites_by_municipality",
    "average_salary_by_municipality",
    "geographical_referential",
    "housing_stock",
    "notary_real_estate_sales",
    "population_by_municipality",
    "poverty_population_by_municipality",
    "real_estate_info_by_municipality"
]

# Dictionary to store DataFrames
dfs = {} 

# Loop through each table and import data
for table in tables:    
    # Run the query and load results into a Pandas DataFrame
    # and store it in the dictionary with the table name as the key
    dfs[table] = run_query(table)



# In[3]:


df_estab = dfs["POI_tourist_establishments"]
df_sites = dfs["POI_touristic_sites_by_municipality"]
df_salary = dfs["average_salary_by_municipality"]
df_geo = dfs["geographical_referential"]
df_housing = dfs["housing_stock"]
df_sales = dfs["notary_real_estate_sales"]
df_pop = dfs["population_by_municipality"]
df_pov = dfs["poverty_population_by_municipality"]
df_realty = dfs["real_estate_info_by_municipality"]


# In[4]:


df_sales.drop(df_sales.loc[df_sales['municipality_code'].isin(['39491', '39530', '39485', '39043', '39577', '39258', '39510', '39339'])].index, inplace=True)


# In[5]:


df_sales_geo = pd.merge(df_sales, df_geo, how='left', on='municipality_code')


# In[6]:


df_sales_geo.drop(df_sales_geo.loc[df_sales_geo['department_code'].isin(['971', '972', '973', '974', '14'])].index, inplace=True)


# In[7]:


# Data treatment

df = df_sales_geo
df = df.dropna(axis=0, subset='department_code')


# In[8]:


df['sales_amount'] = pd.to_numeric(df['sales_amount'])
df['surface'] = pd.to_numeric(df['surface'])
df = df.apply(pd.to_numeric, errors='ignore')


# In[9]:


df['price_per_sqm'] = df['sales_amount'] / df['surface']
department_avg_price = df.groupby('department_code')['price_per_sqm'].mean().reset_index()


# In[10]:


df['price_per_sqm'] = df['sales_amount']/df['surface']
department_avg_price = df.groupby(['department_code', 'department_name'], as_index=False).agg({'price_per_sqm': 'mean',
                                                                                                         'latitude_y': 'mean',
                                                                                                         'longitude_y': 'mean'})
department_avg_price = department_avg_price.rename(columns = {'latitude_y': 'latitude',
                                       'longitude_y': 'longitude'})
dep_avg = department_avg_price


# Cap the price_per_sqm at a reasonable value (e.g., 10,000)
max_cap = 10000
dep_avg['price_per_sqm_capped'] = dep_avg['price_per_sqm'].clip(upper=max_cap)

# Normalize the capped price_per_sqm to a range of 0 to 100
min_price = dep_avg['price_per_sqm_capped'].min()
max_price = dep_avg['price_per_sqm_capped'].max()
dep_avg['normalized_size'] = (
    (dep_avg['price_per_sqm_capped'] - min_price) / (max_price - min_price)
) * 100


# In[11]:


# Group by municipality_code and nom_commune, and calculate the mean for multiple fields
df['price_per_sqm'] = df['sales_amount']/df['surface']

municipality_avg_data = df.groupby(
    ['municipality_code', 'nom_commune', 'department_code'],  # Group by these columns
    as_index=False  # Keep the grouped columns as regular columns
).agg({
    'price_per_sqm': 'mean',  # Average price per square meter
    'latitude_y': 'mean',       # Average latitude
    'longitude_y': 'mean'       # Average longitude
})

municipality_avg_data = municipality_avg_data.rename(columns = {'latitude_y': 'latitude',
                                                                'longitude_y': 'longitude'})


# Cap the price_per_sqm at a reasonable value (e.g., 10,000)
max_cap = 10000
municipality_avg_data['price_per_sqm_capped'] = municipality_avg_data['price_per_sqm'].clip(upper=max_cap)

# Normalize the capped price_per_sqm to a range of 0 to 100
min_price = municipality_avg_data['price_per_sqm_capped'].min()
max_price = municipality_avg_data['price_per_sqm_capped'].max()
municipality_avg_data['normalized_size'] = (
    (municipality_avg_data['price_per_sqm_capped'] - min_price) / (max_price - min_price)
) * 100

mun_avg = municipality_avg_data


# In[12]:


# Create street level

df['price_per_sqm'] = df['sales_amount']/df['surface']

street_avg_data = df.groupby(
    ['municipality_code', 'nom_commune', 'department_code', 'street_code', 'street_name'],  # Group by these columns
    as_index=False  # Keep the grouped columns as regular columns
).agg({
    'price_per_sqm': ['mean', 'max', 'min'],  # Average price per square meter
    'latitude_x': 'mean',       # Average latitude
    'longitude_x': 'mean'       # Average longitude
})

# Flatten the multi-level column index
street_avg_data.columns = [
    'municipality_code', 'nom_commune', 'department_code', 'street_code', 'street_name',
    'price_per_sqm', 'max_per_sqm', 'min_per_sqm',
    'latitude_x', 'longitude_x'
]

street_avg_data = street_avg_data.rename(columns = {'latitude_x': 'latitude',
                                                                'longitude_x': 'longitude'})


# Cap the price_per_sqm at a reasonable value (e.g., 10,000)
max_cap = 10000
street_avg_data['price_per_sqm_capped'] = street_avg_data['price_per_sqm'].clip(upper=max_cap)

# Normalize the capped price_per_sqm to a range of 0 to 100
min_price = street_avg_data['price_per_sqm_capped'].min()
print(min_price)
max_price = street_avg_data['price_per_sqm_capped'].max()
print(max_price)
street_avg_data['normalized_size'] = (
    (street_avg_data['price_per_sqm_capped'] - min_price) / (max_price - min_price)
) * 100

str_avg = street_avg_data


# In[13]:


# Clean Poi data for each municipality 
df_poi=pd.concat([df_estab, df_sites], ignore_index=True)
df_poi.loc[df_poi['poi'] == '1', 'poi'] = df_poi['name'].str.extract(r'\(([^()]+)\)$', expand=False)
df_poi.loc[df_poi['poi'] == '2', 'poi'] = df_poi['name'].str.extract(r'\(([^()]+)\)$', expand=False)
df_poi.dropna(inplace=True)

# Aggregate poi data per municipality
df_poi_agg = df_poi.groupby(['municipality_code'], as_index=False).agg({
    'poi': ['count', 'nunique']
})

df_poi_agg.columns = ['municipality_code', 'poi_count', 'poi_type']



# In[14]:


df_poi_agg.head()


# In[15]:


# Aggregate sales data per municipality per year
df_sales['sales_date'] = pd.to_datetime(df_sales['sales_date'])
df_sales_agg = df_sales.set_index('sales_date').groupby(['municipality_code']).resample('YE').agg({
    'sales_price_m2': 'mean',
    'sales_amount': 'count'
}).reset_index()
df_sales_agg.rename(columns={'sales_amount': 'nb_sales'}, inplace=True)
df_sales_agg['sales_year']=df_sales_agg['sales_date'].dt.year


# In[16]:


# Aggregate sales data per street name per year
# df_street_agg = df_sales.set_index('sales_date').groupby(['municipality_code', 'street_code']).resample('YE').agg({
    #'sales_price_m2': 'mean',
    #'sales_amount': 'count'
#}).reset_index()
#df_street_agg.rename(columns={'sales_amount': 'nb_sales'}, inplace=True)
#df_street_agg['sales_year']=df_street_agg['sales_date'].dt.year


# In[17]:


#df_street_agg.head()


# In[18]:


from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc

# Initialise the app
app = Dash()

# App layout
app.layout = html.Div([
    html.H1("French Real Estate Dashboard"),  # Title
    html.Button('Back to Departments', id='back-button', style={'display': 'none'}),  # Back button (hidden initially)
    dcc.Graph(id='department-map'),  # Department-level map
    dcc.Graph(id='municipality-map', style={'display': 'none'}),  # Municipality-level map (drill-down)
    dcc.Graph(id='street-map', style={'display': 'none'})  # Street-level map (drill-down)
])


# In[19]:


from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc

# Assuming dep_avg, mun_avg, and str_avg are predefined DataFrames

# Create the department map
dep_fig = px.scatter_map(
    dep_avg,
    lat='latitude',
    lon='longitude',
    color='price_per_sqm',  # Use price per sqm for bubble color
    size='normalized_size',  # Use normalized size for bubble size
    color_continuous_scale="bluered",  # Use a different color scale
    zoom=4,
    center={"lat": 46.603354, "lon": 1.888334},  # Center on France
    opacity=0.7,
    labels={"price_per_sqm": "Price per sqm"},
    hover_name='department_name',
    hover_data={
        'price_per_sqm': ':.2f',  # Format the price to 2 decimal places
        'normalized_size': False,   # Hide normalized_size from hover data
        'department_code': True
    },
    custom_data=['department_code']
)

dep_fig.update_layout(
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    coloraxis_colorbar={"title": "Price (€)",
                        #"len": 0.5, # Adjust legend size
                        "thickness": 15, # Reduce the thickness
    },
    map_style="carto-positron"
)

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout of the dashboard
app.layout = dbc.Container([
    #fluid=True,
    dbc.Tabs([
        dbc.Tab(label='Maps', children=[
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Graph(id='map1', figure=dep_fig, style={'height': '45vh'}),
                            dcc.Graph(id='map2', figure=dep_fig, style={'height': '45vh'})
                        ],
                        width=4  # Left column takes 4 out of 12 columns
                    ),
                    dbc.Col(
                        dcc.Graph(id='map3', figure=dep_fig, style={'height': '90vh'}),
                        width=8  # Right column takes 8 out of 12 columns
                    )
                ]
            )
            ]),
            dbc.Tab(label='Detail', children=[
                html.Div(id='detailed-data-tab')
            ])
    ])
], fluid=True)

@app.callback(
    Output('map2', 'figure'),
    Output('map3', 'figure'),
    Output('detailed-data-tab', 'children'),
    Input('map1', 'clickData'),
    Input('map2', 'clickData'),
    Input('map3', 'clickData'),
    prevent_initial_call=True
)

def update_maps(clickData1, clickData2, clickData3):
    if clickData1 is None and clickData2 is None and clickData3 is None:
        return dep_fig, dep_fig, "Select a street to see detailed data."

    # Get the department code from map1, update map2 based on click
    if clickData1 is not None:
        department_code = clickData1['points'][0]['customdata'][0]
        print(department_code)
        filtered_mun_avg = mun_avg[mun_avg['department_code'] == department_code]
        
        mun_fig = px.scatter_map(
            filtered_mun_avg,
            lat='latitude',
            lon='longitude',
            color='price_per_sqm',
            size='normalized_size',
            color_continuous_scale="bluered",
            zoom=7,  # Zoom in for municipalities
            center={"lat": filtered_mun_avg['latitude'].mean(),
                    "lon": filtered_mun_avg['longitude'].mean()},
            opacity=0.7,
            labels={"price_per_sqm": "Price per sqm"},
            hover_name='nom_commune',
            hover_data={
                'price_per_sqm': ':.2f',
                'normalized_size': False,
                'department_code': True,
                'municipality_code': True  # Include municipality_code in hover data
            },
            custom_data=['municipality_code']  # Add municipality_code to customdata
        )
        
        mun_fig.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            coloraxis_colorbar={"title": "Price (€)",
                                #"len": 0.5, # Adjust legend size
                                "thickness": 15, # Reduce the thickness
            },
            map_style="carto-positron"
        )
    else:
        mun_fig = dep_fig # Always return to dep_fig

    # Update map3 with click on map2
    if clickData2 is not None:
        
        # Get the municipality code from map-2
        municipality_code = clickData2['points'][0]['customdata'][0]
        print(municipality_code)
        filtered_str_avg = str_avg[str_avg['municipality_code'] == municipality_code]
    
        # Update str_fig
        str_fig = px.scatter_map(
            filtered_str_avg,
            lat='latitude',
            lon='longitude',
            color='price_per_sqm',
            size='normalized_size',
            color_continuous_scale="bluered",
            zoom=13,  # Zoom in for municipalities
            center={"lat": filtered_str_avg['latitude'].mean(),
                    "lon": filtered_str_avg['longitude'].mean()},
            opacity=0.7,
            labels={"price_per_sqm": "Price per sqm"},
            hover_name='street_name',
            hover_data={
                'price_per_sqm': ':.2f',
                'normalized_size': False,
                'department_code': True,
                'municipality_code': True,  # Include municipality_code in hover data
                'street_code': True
            },
            custom_data=['street_code']  # Add street_code to customdata
        )
        str_fig.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            coloraxis_colorbar={"title": "Price per sqm (€)"},
            map_style="carto-positron"
        )


        
    else:
        str_fig = dep_fig # Always show at least the department


    
    # Update detailed-data-tab based on click on map3
    if clickData3 is not None:
        street_code = clickData3['points'][0]['customdata'][0]
        selected_street = filtered_str_avg[filtered_str_avg['street_code'] == street_code]
        
        # Check if selected_street is not empty
        if not selected_street.empty:
            street_name = selected_street['street_name'].iloc[0]
            avg_price = selected_street['price_per_sqm'].mean()
            max_price = selected_street['max_per_sqm'].max()
            min_price = selected_street['min_per_sqm'].min()
            poi = df_poi_agg[df_poi_agg['municipality_code'] == municipality_code]
            if poi.empty:
                poi_count = 0
                poi_type = 0
            else:
                poi_count = poi['poi_count'].iloc[0]
                poi_type = poi['poi_type'].iloc[0]
        
                #cCreate a small line plot for street sales data
                sales = df_sales_agg[df_sales_agg['municipality_code'] == municipality_code]
                line_fig = px.line(sales, x='sales_year', y='sales_price_m2', text="nb_sales", markers=True, title="Price per square meter evolution")

            
            detailed_content = html.Div([
                html.H4(f"Street: {street_name}"),
                html.P(f"Street code and name: {street_code}, {street_name}"),
                html.P(f"Average Price per sqm: €{avg_price:.2f}"),
                html.P(f"Maximum Price per sqm: €{max_price:.2f}"),
                html.P(f"Minimum Price per sqm: €{min_price:.2f}"),
                html.P(f"Points of interest in the municipality: {poi_count}"),
                html.P(f"Type of points of interest in the municipality: {poi_type}"),
                dcc.Graph(figure=line_fig)  # Add the histogram to the dashboarb
            ])
        else:
            detailed_content = "No data available for the selected street."
    else:
        detailed_content = "Select a street to see detailed data."
    return mun_fig, str_fig, detailed_content

# Run the app
if __name__ == '__main__':
    app.run(port=8053, debug=True)






