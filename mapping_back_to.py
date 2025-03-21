import numpy as np
import pandas as pd

# 讀取 Excel 檔案中的新測量點
file_path = "20.xlsx"  # 你的 Excel 檔案名稱
df = pd.read_excel(file_path)

# 讀取測量點的經緯度
new_longitude = df['經度'].values
new_latitude = df['緯度'].values
new_points = np.vstack((new_latitude, new_longitude)).T

# 轉彎點（原始河道曲線）
turning_points = np.array([
    [24.85673, 121.800161],
    [24.855898, 121.801116],
    [24.853698, 121.804258],
    [24.853605, 121.804203],
    [24.853559, 121.804206],
    [24.8535205, 121.8042687],
    [24.853499, 121.8043151],
    [24.8534829, 121.8043563],
    [24.8534572, 121.804387],
    [24.8533031, 121.8046549],
    [24.853212, 121.80479],
    [24.8530468, 121.8049825],
    [24.8529641, 121.8051042],
    [24.852933, 121.8051702],
    [24.85265, 121.804973],
    [24.852488, 121.804969],
    [24.8523532, 121.8048866],
    [24.8521998, 121.8048336],
    [24.8519978, 121.8047297],
    [24.8518524, 121.804615],
    [24.8515883, 121.8043766],
    [24.85146, 121.8042378],
    [24.8515503, 121.8041366],
    [24.8516121, 121.8040262],
    [24.8512859, 121.8037456],
    [24.8512847, 121.8037047],
    [24.8511633, 121.8036125],
    [24.8510769, 121.8035545],
    [24.8510608, 121.8035398],
    [24.8510495, 121.8035445],
    [24.8509955, 121.8036217],
    [24.8509464, 121.8037056],
    [24.850902, 121.8036953],
    [24.8508618, 121.8036536],
    [24.850858, 121.8036192],
    [24.8507426, 121.8035327],
    [24.8506799, 121.8034925],
    [24.8506349, 121.8034677],
    [24.8506221, 121.803467],
    [24.8505972, 121.8035019],
    [24.8505862, 121.8035093],
    [24.850568, 121.8035079],
    [24.8505083, 121.8034321],
    [24.8502303, 121.8032055],
    [24.8502053, 121.8031907],
    [24.8500988, 121.8033215],
    [24.8500149, 121.8034455],
    [24.8499571, 121.8034911],
    [24.8499205, 121.8034858],
    [24.8498451, 121.8034154],
    [24.8497922, 121.8033497],
    [24.8497678, 121.8033356],
    [24.8497283, 121.8033798],
    [24.8495676, 121.8036179],
    [24.84929, 121.80397],
    [24.848993, 121.804471],
    [24.8489324, 121.8044849],
    [24.8487365, 121.8047924],
    [24.8486689, 121.8048819],
    [24.8486145, 121.8049791],
    [24.8486017, 121.805021],
    [24.8485661, 121.8050639]
])

# 設定河道的總長度
actual_length = 1036.8  # 整條河道的實際長度 (m)

# **🚀 使用 Haversine 公式計算經緯度距離**
from geopy.distance import geodesic

def get_acc_distance(points):
    dist = np.zeros(len(points))
    for i in range(1, len(points)):
        dist[i] = dist[i - 1] + geodesic(points[i - 1], points[i]).meters  # 直接用地理距離
    return dist

accumulated_turning_distances = get_acc_distance(turning_points)

# **🛰️ 新測量點映射到河道曲線**
from scipy.spatial import KDTree

# 使用 KDTree 加速最近轉彎點查找
tree = KDTree(turning_points)

mapped_distances = []
for new_pt in new_points:
    dist, idx = tree.query(new_pt)
    
    if idx == len(turning_points) - 1:
        mapped_distances.append(accumulated_turning_distances[-1])
        continue

    # 找到最近兩個轉彎點，做線性插值
    p1, p2 = turning_points[idx], turning_points[idx + 1]
    segment_length = geodesic(p1, p2).meters
    
    proj_ratio = geodesic(new_pt, p1).meters / segment_length
    mapped_distance = accumulated_turning_distances[idx] + proj_ratio * (accumulated_turning_distances[idx + 1] - accumulated_turning_distances[idx])
    mapped_distances.append(mapped_distance)

# **📏 正規化到 0 ~ 1036.8**
mapped_distances = np.array(mapped_distances)
mapped_distances = (mapped_distances / accumulated_turning_distances[-1]) * actual_length

# 整合到 DataFrame 並顯示結果
result_df = pd.DataFrame({
    'dist': mapped_distances,
    'longitude': new_longitude,
    'latitude': new_latitude
})

# 顯示結果
print(result_df)

# 如果需要存檔：
result_df.to_excel("mapped_coordinates.xlsx", index=False)
