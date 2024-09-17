import keras.layers
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from keras.layers import Input, Dense
from keras.models import Model
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# Sample Data - Replace with actual user data
data = {
    'USR_ID': [1, 2, 3, 4, 5],
    'USER_NAME': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'LOGIN_TIME': ['2024-09-12 08:00:00', '2024-09-12 09:15:00', '2024-09-12 08:30:00', '2024-09-12 08:45:00', '2024-09-12 09:00:00'],
    'LOGOUT_TIME': ['2024-09-12 12:00:00', '2024-09-12 12:15:00', '2024-09-12 11:30:00', '2024-09-12 12:45:00', '2024-09-12 12:30:00'],
    'ORG_ID': ['Org1', 'Org2', 'Org1', 'Org2', 'Org3'],
    'ZONE': ['East', 'West', 'East', 'West', 'North']
}

df = pd.DataFrame(data)

# Step 1: Data Preprocessing
df['LOGIN_TIME'] = pd.to_datetime(df['LOGIN_TIME'])
df['LOGOUT_TIME'] = pd.to_datetime(df['LOGOUT_TIME'])
df['SESSION_DURATION'] = (df['LOGOUT_TIME'] - df['LOGIN_TIME']).dt.total_seconds() / 3600

le_org = LabelEncoder()
df['ORG_ID'] = le_org.fit_transform(df['ORG_ID'])

le_zone = LabelEncoder()
df['ZONE'] = le_zone.fit_transform(df['ZONE'])

# Features for clustering
features = ['SESSION_DURATION', 'ORG_ID', 'ZONE']
X = df[features]

# Step 2: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(type(X_scaled))
# Step 3: Build Autoencoder Model with Keras
input_dim = X_scaled.shape[1]
print(input_dim)
encoding_dim = 2  # Compress to 2 dimensions for visualization
input_layer = keras.layers.Input(shape=(input_dim,), name='userId')
#input_layer = Input()
encoder = Dense(encoding_dim, activation='relu')(input_layer)
decoder = Dense(input_dim, activation='sigmoid')(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the Autoencoder
autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=2, shuffle=True, verbose=0)

# Extract the encoder part for dimensionality reduction
encoder_model = Model(inputs=input_layer, outputs=encoder)
X_encoded = encoder_model.predict(X_scaled)

# Step 4: K-Means Clustering on Encoded Data
kmeans = KMeans(n_clusters=3, random_state=42)
df['CLUSTER'] = kmeans.fit_predict(X_encoded)

# Step 5: Ranking users based on session duration and clusters
df['RANK'] = df.groupby('CLUSTER')['SESSION_DURATION'].rank(ascending=False)

# Step 6: Build Dashboard with Dash
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("User Clustering Dashboard with Autoencoder", style={'textAlign': 'center'}),

    dcc.Dropdown(
        id='org-dropdown',
        options=[{'label': f'Organization {org}', 'value': org} for org in df['ORG_ID'].unique()],
        placeholder="Select Organization",
        multi=True
    ),
    dcc.Graph(id='cluster-graph'),

    html.H2("User Ranking"),
    dcc.Dropdown(
        id='cluster-dropdown',
        options=[{'label': f'Cluster {c}', 'value': c} for c in df['CLUSTER'].unique()],
        placeholder="Select Cluster"
    ),
    dcc.Graph(id='rank-graph')
])

# Callback for updating the cluster graph based on selected organization
@app.callback(
    Output('cluster-graph', 'figure'),
    [Input('org-dropdown', 'value')]
)
def update_cluster_graph(selected_orgs):
    filtered_df = df if not selected_orgs else df[df['ORG_ID'].isin(selected_orgs)]
    fig = px.scatter(filtered_df, x='USR_ID', y='SESSION_DURATION', color='CLUSTER',
                     hover_data=['USER_NAME', 'ORG_ID', 'ZONE'],
                     title="User Clustering based on Encoded Data")
    return fig

# Callback for updating the rank graph based on selected cluster
@app.callback(
    Output('rank-graph', 'figure'),
    [Input('cluster-dropdown', 'value')]
)
def update_rank_graph(selected_cluster):
    filtered_df = df if selected_cluster is None else df[df['CLUSTER'] == selected_cluster]
    fig = px.bar(filtered_df, x='USER_NAME', y='RANK', color='SESSION_DURATION',
                 hover_data=['SESSION_DURATION', 'ORG_ID', 'ZONE'],
                 title=f"User Ranking in Cluster {selected_cluster}")
    return fig

# Step 7: Run the Dashboard
if __name__ == '__main__':
    app.run_server(debug=True)
