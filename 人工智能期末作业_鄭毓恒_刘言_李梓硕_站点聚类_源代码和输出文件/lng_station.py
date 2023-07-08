import os
import math
import time
import json
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import haversine_distances


def distance(a, b):
    rada = [0, 0]
    radb = [0, 0]
    rada[0] = math.radians(a[0])
    rada[1] = math.radians(a[1])
    radb[0] = math.radians(b[0])
    radb[1] = math.radians(b[1])
    resultm = haversine_distances([rada, radb])
    return 6371 * resultm[0][1]


def get_closest_pt(pt, df):
    dist, index = 1e12, -1
    for i in df.index:
        tmp = distance(pt, df.loc[i])
        if tmp < dist:
            dist, index = tmp, i
    return index if dist < 30 else -1


start = time.time()
cur_dir = os.getcwd()
data_dir = cur_dir + '/data/lng2.csv'
data = pd.read_csv(data_dir, delim_whitespace=True, header=None)
data.columns = ['mmsi', 'time', 'state', 'speed', 'longtitude', 'latitude', 'draft']
data.dropna()
data = data[data['draft'] != 0]
state = data['state'][:]
data = data[(state == 1) | (state == 5) | (state == 15)]

db_start = time.time()
coor_df = data[['longtitude', 'latitude']]
cluster_center = pd.DataFrame(columns=['longtitude', 'latitude'])
i = 0
while i < len(coor_df.index):
    j = 0
    if i + 30000 < len(coor_df.index):
        j = i + 30000
    else:
        j = len(coor_df.index)

    coor_tmp = coor_df.iloc[i:j]
    dbscan_result = DBSCAN(eps=5/6371, min_samples=300, metric='haversine', algorithm='ball_tree',
                           n_jobs=12).fit_predict(np.radians(coor_tmp))
    n_clusters_tmp = len(set(dbscan_result)) - (1 if -1 in dbscan_result else 0)
    for k in range(n_clusters_tmp):
        k_group = coor_tmp[dbscan_result == k]
        mean_lon = k_group['longtitude'].mean()
        mean_lat = k_group['latitude'].mean()
        cluster_center = pd.DataFrame(np.insert(cluster_center.values, len(cluster_center.index), [mean_lon, mean_lat],
                                                axis=0), columns=['longtitude', 'latitude'], dtype=float)
    i = j
dbscan_result = DBSCAN(eps=30/6371, min_samples=1, metric='haversine', algorithm='ball_tree',
                       n_jobs=12).fit_predict(np.radians(cluster_center))
n_clusters_ = len(set(dbscan_result)) - (1 if -1 in dbscan_result else 0)
clusters = pd.DataFrame(columns=['longtitude', 'latitude'])
for k in range(n_clusters_):
    k_group = cluster_center[dbscan_result == k]
    mean_lon = k_group['longtitude'].mean()
    mean_lat = k_group['latitude'].mean()
    clusters = pd.DataFrame(np.insert(clusters.values, len(clusters.index), [mean_lon, mean_lat], axis=0),
                            columns=['longtitude', 'latitude'], dtype=float)
db_end = time.time()

clusters.insert(cluster_center.shape[1], 'isLNG', False)
clusters.insert(cluster_center.shape[1], 'IN', False)
clusters.insert(cluster_center.shape[1], 'OUT', False)

station_data = data[['mmsi', 'longtitude', 'latitude', 'draft']]
pre = station_data.iloc[0]
avg = [pre['longtitude'], pre['latitude']]
count = 0
draft_first, draft_final = pre['draft'], pre['draft']
for index, row in station_data.iterrows():
    cur = row
    if distance([cur['longtitude'], cur['latitude']], [pre['longtitude'], pre['latitude']]) < 30 \
            and cur['mmsi'] == pre['mmsi']:
        avg[0] += cur['longtitude']
        avg[1] += cur['latitude']
        draft_final = cur['draft']
        pre = cur
        count += 1
    else:
        if count > 0:
            avg[0] = avg[0] / (count + 1)
            avg[1] = avg[1] / (count + 1)
            draft_change = draft_final - draft_first

            closest_pt = get_closest_pt(avg, clusters[['longtitude', 'latitude']])
            if closest_pt != -1:
                if draft_change > 5:
                    clusters.at[closest_pt, 'isLNG'] = True
                    clusters.at[closest_pt, 'OUT'] = True
                elif draft_change < -5:
                    clusters.at[closest_pt, 'isLNG'] = True
                    clusters.at[closest_pt, 'IN'] = True
        count = 0
        pre = cur
        avg = [pre['longtitude'], pre['latitude']]
        draft_first = pre['draft']

output_path = cur_dir + "/data/lng_results_list.json"
count_lng, count_out, count_in, count_mooring = 0, 0, 0, 0
with open(output_path, "w") as f:
    count = 1
    f.write("[\n\t")
    for index, row in clusters.iterrows():
        if row['isLNG'] is True:
            count_lng += 1
            if row['IN'] is True:
                if count != 1:
                    f.write(",\n\t")
                json_dict = {
                    "code": count,
                    "latitude": row['latitude'],
                    "longtitude": row['longtitude'],
                    "isLNG": row['isLNG'],
                    "IN": row['IN']
                }
                json.dump(json_dict, f)
                count += 1
                count_in += 1
            if row['OUT'] is True:
                if count != 1:
                    f.write(",\n\t")
                json_dict = {
                    "code": count,
                    "latitude": row['latitude'],
                    "longtitude": row['longtitude'],
                    "isLNG": row['isLNG'],
                    "IN": False
                }
                json.dump(json_dict, f)
                count += 1
                count_out += 1
        else:
            if index != 0:
                f.write(",\n\t")
            json_dict = {
                "code": count,
                "latitude": row['latitude'],
                "longtitude": row['longtitude'],
                "isLNG": row['isLNG'],
                "IN": None
            }
            json.dump(json_dict, f)
            count += 1
            count_mooring += 1
    f.write("\n]")

end = time.time()
print("LNG Station Number: {}".format(count_lng))
print("IN Station Number: {}".format(count_in))
print("OUT Station Number: {}".format(count_out))
print("Mooring Point Number: {}".format(count_mooring))
print("DBSCAN Clustering Time: {} second".format(db_end - db_start))
print("Total Time: {} second".format(end - start))
