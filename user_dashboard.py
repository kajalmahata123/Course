from flask import Flask, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging

# Flask Setup
app = Flask(__name__)

# Configure SQLAlchemy with MySQL
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://your_username:your_password@localhost/your_database'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define a model for your table
class YourTable(db.Model):
    __tablename__ = 'your_table'
    id = db.Column(db.Integer, primary_key=True)
    usr_id = db.Column(db.Integer)
    user_name = db.Column(db.String(50))
    login_time = db.Column(db.DateTime)
    logout_time = db.Column(db.DateTime)
    org_id = db.Column(db.String(50))
    zone = db.Column(db.String(50))

# Cluster names mapping
CLUSTER_NAMES = {
    0: 'Cluster A',
    1: 'Cluster B',
    2: 'Cluster C'
}

# Function to fetch data into a DataFrame
def fetch_data_as_dataframe():
    try:
        query = db.session.query(YourTable).statement
        df = pd.read_sql(query, db.session.bind)
        return df
    except Exception as e:
        logging.error(f"An error occurred while fetching data: {e}")
        raise

# Function to preprocess the data and cluster
def preprocess_and_cluster(df):
    df['LOGIN_TIME'] = pd.to_datetime(df['LOGIN_TIME'])
    df['LOGOUT_TIME'] = pd.to_datetime(df['LOGOUT_TIME'])
    df['SESSION_DURATION'] = (df['LOGOUT_TIME'] - df['LOGIN_TIME']).dt.total_seconds() / 3600

    le_org = LabelEncoder()
    df['ORG_ID'] = le_org.fit_transform(df['ORG_ID'])

    le_zone = LabelEncoder()
    df['ZONE'] = le_zone.fit_transform(df['ZONE'])

    features = ['SESSION_DURATION', 'ORG_ID', 'ZONE']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['CLUSTER'] = kmeans.fit_predict(X_scaled)
    df['RANK'] = df.groupby('CLUSTER')['SESSION_DURATION'].rank(ascending=False)

    return df, kmeans, X_scaled

# Initialize and refresh clustering
def initialize_clusters():
    global df, kmeans, X_scaled
    df = fetch_data_as_dataframe()
    df, kmeans, X_scaled = preprocess_and_cluster(df)

# Initialize and start the scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(func=initialize_clusters, trigger="interval", minutes=30)  # Refresh every 30 minutes
scheduler.start()

# API 1: Get Full User Data
@app.route('/api/users', methods=['GET'])
def get_users():
    try:
        if df.empty:
            logging.warning("DataFrame is empty")
            return jsonify({"error": "No data available"}), 404
        result = df.to_dict(orient='records')
        # Add cluster names to the result
        for record in result:
            record['CLUSTER_NAME'] = CLUSTER_NAMES.get(record['CLUSTER'], 'Unknown')
        return jsonify(result)
    except Exception as e:
        logging.error(f"An error occurred while fetching user data: {e}")
        return jsonify({"error": "Internal server error"}), 500

# API 2: Get Cluster Summary Stats
@app.route('/api/cluster_summary', methods=['GET'])
def cluster_summary():
    try:
        cluster_summary = df.groupby('CLUSTER').agg({
            'SESSION_DURATION': ['mean', 'median'],
            'ORG_ID': pd.Series.mode,
            'ZONE': pd.Series.mode
        }).reset_index()

        # Add cluster names
        cluster_summary['CLUSTER_NAME'] = cluster_summary['CLUSTER'].map(CLUSTER_NAMES)

        summary = cluster_summary.to_dict(orient='records')
        return jsonify(summary)
    except Exception as e:
        logging.error(f"An error occurred while fetching cluster summary: {e}")
        return jsonify({"error": "Internal server error"}), 500

# API 3: Get Silhouette Score
@app.route('/api/silhouette', methods=['GET'])
def get_silhouette():
    try:
        score = silhouette_score(X_scaled, df['CLUSTER'])
        return jsonify({'silhouette_score': score})
    except Exception as e:
        logging.error(f"An error occurred while calculating silhouette score: {e}")
        return jsonify({"error": "Internal server error"}), 500

# API 4: Get Cluster Size Distribution
@app.route('/api/cluster_size', methods=['GET'])
def cluster_size():
    try:
        cluster_size = df['CLUSTER'].value_counts().to_dict()
        # Map cluster IDs to names
        cluster_size_named = {CLUSTER_NAMES.get(k, 'Unknown'): v for k, v in cluster_size.items()}
        return jsonify(cluster_size_named)
    except Exception as e:
        logging.error(f"An error occurred while fetching cluster size distribution: {e}")
        return jsonify({"error": "Internal server error"}), 500

# API 5: Get Login/Logout Time Analysis (e.g., average login/logout time per cluster)
@app.route('/api/login_logout_analysis', methods=['GET'])
def login_logout_analysis():
    try:
        login_logout_summary = df.groupby('CLUSTER').agg({
            'LOGIN_TIME': 'mean',
            'LOGOUT_TIME': 'mean'
        }).reset_index()
        # Add cluster names
        login_logout_summary['CLUSTER_NAME'] = login_logout_summary['CLUSTER'].map(CLUSTER_NAMES)
        return jsonify(login_logout_summary.to_dict(orient='records'))
    except Exception as e:
        logging.error(f"An error occurred while fetching login/logout analysis: {e}")
        return jsonify({"error": "Internal server error"}), 500

# API 6: Rank Organizations per Cluster
@app.route('/api/rank_organizations', methods=['GET'])
def rank_organizations_per_cluster():
    try:
        if df.empty:
            logging.warning("DataFrame is empty")
            return jsonify({"error": "No data available"}), 404

        org_duration = df.groupby(['CLUSTER', 'ORG_ID'])['SESSION_DURATION'].mean().reset_index()

        if org_duration.empty:
            logging.warning("No session duration data available after grouping")
            return jsonify({"error": "No valid session duration data"}), 404

        org_duration['RANK'] = org_duration.groupby('CLUSTER')['SESSION_DURATION'].rank(ascending=False)

        # Add cluster names
        org_duration['CLUSTER_NAME'] = org_duration['CLUSTER'].map(CLUSTER_NAMES)

        result = org_duration.to_dict(orient='records')
        return jsonify(result)
    except Exception as e:
        logging.error(f"An error occurred while ranking organizations: {e}")
        return jsonify({"error": "Internal server error"}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
