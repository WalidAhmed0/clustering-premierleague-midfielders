import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Load your data
df = pd.read_csv('/Users/walidahmed/Documents/Documents/DataScience/24_25midfielder.csv')  

# 2. Filter only midfielders
df = df[df['Pos'].str.contains('MF', na=False)]

# 3. Select features for clustering
features = [
    'xA',              # expected assists per 90
    'KP',              # key passes per 90
    'SCA',             # shot-creating actions per 90
    'PPA',             # passes into penalty area per 90
    'PrgP',            # progressive passes completed per 90
    'PrgC',            # progressive carries per 90
    'TcklW',           # tackles won per 90
    'Int',             # interceptions per 90
    'ShortCmp',        # short passes completed per 90
    'MedCmp',          # medium passes completed per 90
    'LongCmp',         # long passes completed per 90
    'xAG',             # expected assisted goals per 90
    '1/3P',            # passes into final third per 90
    'CrsPA',           # crosses into penalty area per 90
    'SuccTkon',        # succesful takeons per 90
    '1/3C',            # carries into final third per 90
    'CPA',             # carries into penatly area per 90
    'PrgR'             # progressive passes received per 90
]
X = df[features].dropna()
player_names = df.loc[X.index, 'Player']

# 4. Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Compute pairwise distance matrix
dist_matrix = squareform(pdist(X_scaled, metric='euclidean'))

# 6. Build the Minimum Spanning Tree
mst_sparse = minimum_spanning_tree(dist_matrix)
G = nx.from_scipy_sparse_array(mst_sparse)

# 7. Compute 2D positions via PCA for visualization
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
pos = {i: coords[i] for i in range(len(coords))}

# 8. Plot the network
plt.figure(figsize=(12, 12))
nx.draw(
    G,
    pos,
    labels={i: player_names.iloc[i] for i in pos},
    node_size=50,
    font_size=6,
    node_color='skyblue',
    edge_color='gray'
)
plt.title('Midfielder Similarity Network (MST)', fontsize=16)
plt.axis('off')
plt.show()