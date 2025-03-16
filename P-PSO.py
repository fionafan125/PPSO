import numpy as np
import math
import random
import pandas as pd
from math import cos, radians
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
import argparse
import os
from src import *
data = {
    'total_dist': {},
    'longitude': {},
    'latitude': {}
}
data['total_dist'] = [1036.8]
data['longitude'] = [
    24.85673,
    24.855898,
    24.853698,
    24.853605,
    24.853559,
    24.8535205,
    24.853499,
    24.8534829,
    24.8534572,
    24.8533031,
    24.853212,
    24.8530468,
    24.8529641,
    24.852933,
    24.85265,
    24.852488,
    24.8523532,
    24.8521998,
    24.8519978,
    24.8518524,
    24.8515883,
    24.85146,
    24.8515503,
    24.8516121,
    24.8512859,
    24.8512847,
    24.8511633,
    24.8510769,
    24.8510608,
    24.8510495,
    24.8509955,
    24.8509464,
    24.850902,
    24.8508618,
    24.850858,
    24.8507426,
    24.8506799,
    24.8506349,
    24.8506221,
    24.8505972,
    24.8505862,
    24.850568,
    24.8505083,
    24.8502303,
    24.8502053,
    24.8500988,
    24.8500149,
    24.8499571,
    24.8499205,
    24.8498451,
    24.8497922,
    24.8497678,
    24.8497283,
    24.8495676,
    24.84929,
    24.848993,
    24.8489324,
    24.8487365,
    24.8486689,
    24.8486145,
    24.8486017,
    24.8485661
]

data['latitude'] = [
    121.800161,
    121.801116,
    121.804258,
    121.804203,
    121.804206,
    121.8042687,
    121.8043151,
    121.8043563,
    121.804387,
    121.8046549,
    121.80479,
    121.8049825,
    121.8051042,
    121.8051702,
    121.804973,
    121.804969,
    121.8048866,
    121.8048336,
    121.8047297,
    121.804615,
    121.8043766,
    121.8042378,
    121.8041366,
    121.8040262,
    121.8037456,
    121.8037047,
    121.8036125,
    121.8035545,
    121.8035398,
    121.8035445,
    121.8036217,
    121.8037056,
    121.8036953,
    121.8036536,
    121.8036192,
    121.8035327,
    121.8034925,
    121.8034677,
    121.803467,
    121.8035019,
    121.8035093,
    121.8035079,
    121.8034321,
    121.8032055,
    121.8031907,
    121.8033215,
    121.8034455,
    121.8034911,
    121.8034858,
    121.8034154,
    121.8033497,
    121.8033356,
    121.8033798,
    121.8036179,
    121.80397,
    121.804471,
    121.8044849,
    121.8047924,
    121.8048819,
    121.8049791,
    121.805021,
    121.8050639
]

actual_length = data['total_dist'][0]  # [0]
#=====storages====


main_file = os.path.join(os.path.dirname(__file__))
store_png_file = "png_file"
penalty_base = 1e5 #改這個
store_result_file = 'result_file'

if not os.path.exists(store_result_file):
    os.makedirs(store_result_file)

storage_name = 'storage_name.pkl'
#==settings =================================
#假設發電15年 設5瓦
parser = argparse.ArgumentParser(description="加參數")
parser.add_argument('--machine_amount', type=int, help="機器數量實驗")
parser.add_argument('--wire_pole', type=int, help="random type")
parser.add_argument('--money_spend', type=str, help="record number")
args = parser.parse_args()

tmp = os.path.join(main_file, store_result_file, 'money_' + str(args.money_spend))
if not os.path.exists(tmp):
    os.mkdir(tmp)

store_file = os.path.join(tmp ,'wire_pole_' + str(args.wire_pole))

if not os.path.exists(store_file):
    os.makedirs(store_file)

store_file = os.path.join(store_file,'machine_amount_'+ str(args.machine_amount) )
print(store_file)
if not os.path.exists(store_file):
    os.makedirs(store_file)

pkl_file = os.path.join(store_file, storage_name)
png_file = os.path.join(store_file, 'converge_curve.png')
year = 10 #假設

iterations = 200  # 迭代次数
record_cost = []
maximum_dist = 50
minimum_dist = 40
KW_volumn = 5.1
#===gain
money_per_year_per_KW = 38551.68 #錢/年1KW
machine_amount = args.machine_amount
maximum_GetBack_money = money_per_year_per_KW * KW_volumn * year * machine_amount
#===cost
cost = 380000
cost_money = cost*KW_volumn*machine_amount
maximum_earned_money = maximum_GetBack_money - cost_money

if maximum_GetBack_money < cost_money:
    print("QAQ")
    exit()
#================================================================

def sampleGeneartor(length):
    X = np.arange(0, length, 0.01)
    return X

class ppso():
    def __init__(self, X_train, LB, UB, dim=4, pop_size=20, max_iter=500, w=0.5, c1=2, c2=2):
        self.X_train = X_train
        self.LB = LB
        self.UB = UB
        self.dim = dim
        self.pop_size = pop_size  # 粒子數
        self.group_size = pop_size // 2  # 每組粒子數
        self.max_iter = max_iter
        self.w = w  # 慣性權重
        self.c1 = c1  # 個體學習因子
        self.c2 = c2  # 群體學習因子

        # 初始化粒子位置和速度
        self.X = np.random.uniform(0, 1, (pop_size, dim)) * (UB - LB) + LB
        self.V = np.zeros((pop_size, dim))  # 初始速度
        self.pBest_X = self.X.copy()  # 個體最佳位置
        self.pBest_score = np.full(pop_size, np.inf)  # 個體最佳適應度
        self.gBest_X_group1 = np.zeros(dim)  # 組1全局最佳位置
        self.gBest_X_group2 = np.zeros(dim)  # 組2全局最佳位置
        self.gBest_score_group1 = np.inf  # 組1全局最佳適應度
        self.gBest_score_group2 = np.inf  # 組2全局最佳適應度
        self.gBest_curve = np.zeros(max_iter)  # 用於記錄每次迭代的最佳適應度

    # 適應度計算（根據需求）
    def fitFunc(self, input):
        # 假設適應度函數邏輯與原程式相同
        pass
    
    # 並行處理適應度和更新
    def parallel_optimize(self, group):
        for i in group:
            # 限制邊界
            self.X[i, :] = np.clip(self.X[i, :], self.LB, self.UB)
            # 計算適應度值
            fitness = self.fitFunc(self.X[i, :])

            # 更新個體最佳位置
            if fitness < self.pBest_score[i]:
                self.pBest_score[i] = fitness
                self.pBest_X[i, :] = self.X[i, :].copy()

            # 更新全局最佳位置（根據粒子所屬的組）
            if group[0] < self.group_size:  # 組1
                if fitness < self.gBest_score_group1:
                    self.gBest_score_group1 = fitness
                    self.gBest_X_group1 = self.X[i, :].copy()
            else:  # 組2
                if fitness < self.gBest_score_group2:
                    self.gBest_score_group2 = fitness
                    self.gBest_X_group2 = self.X[i, :].copy()

        return group

    # 更新速度和位置
    def update_velocity_position(self, group, gBest_X):
        for i in group:
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            self.V[i, :] = (self.w * self.V[i, :] +
                            self.c1 * r1 * (self.pBest_X[i, :] - self.X[i, :]) +
                            self.c2 * r2 * (gBest_X - self.X[i, :]))
            self.X[i, :] = self.X[i, :] + self.V[i, :]

    # 主優化模塊
    def opt(self):
        for t in range(self.max_iter):
            group1 = range(0, self.group_size)
            group2 = range(self.group_size, self.pop_size)

            # 使用多進程並行計算適應度
            with Pool(2) as pool:
                pool.map(self.parallel_optimize, [group1, group2])

            # 交換最優解以避免局部最佳
            if self.gBest_score_group1 < self.gBest_score_group2:
                self.gBest_X_group2 = self.gBest_X_group1
                self.gBest_score_group2 = self.gBest_score_group1
            else:
                self.gBest_X_group1 = self.gBest_X_group2
                self.gBest_score_group1 = self.gBest_score_group2

            # 並行更新速度與位置
            with Pool(2) as pool:
                pool.starmap(self.update_velocity_position, [
                    (group1, self.gBest_X_group1),
                    (group2, self.gBest_X_group2)
                ])

            # 記錄當前全局最佳
            self.gBest_curve[t] = min(self.gBest_score_group1, self.gBest_score_group2)
            if t % 100 == 0:
                print(f'At iteration {t}, best fitness is: {self.gBest_curve[t]}')

        return self.gBest_curve, (self.gBest_X_group1 if self.gBest_score_group1 < self.gBest_score_group2 else self.gBest_X_group2)

    



def plot_converge_curve(record_cost, storage_name):
    
    # 繪製折線圖
    iters = np.arange(1, 2001, 1)
    plt.plot(iters, record_cost, marker='o')  # 使用圓圈標記每個數據點

    # 標題和軸標籤
    plt.title("Simple Line Chart Example")
    plt.xlabel("Iters")
    plt.ylabel("Records")

    # 顯示圖表
    plt.savefig(storage_name)

def compute_dis_xy(init_pos, final_pos, meter_per_degree_latitude):
    

    # 計算南北方向上的距離（Δ緯度 × 每度緯度的公尺數）
    delta_latitude = final_pos[1] - init_pos[1]
    distance_north_south = delta_latitude * meters_per_degree_latitude

    # 計算東西方向上的距離（Δ經度 × cos(平均緯度) × 每度緯度的公尺數）
    average_latitude = radians((init_pos[1] + final_pos[1]) / 2)
    delta_longitude = final_pos[0] - init_pos[0]
    distance_east_west = delta_longitude * cos(average_latitude) * meters_per_degree_latitude
    return [distance_north_south, (-1)*distance_east_west]

def get_compared_dist(points, meters_per_degree_latitude):

    compared_dist = []
    for point in points:
        tmp_dist = compute_dis_xy(points[0,:], point, meters_per_degree_latitude)
        compared_dist.append(tmp_dist)

    compared_dist = np.array(compared_dist)
    return compared_dist

#useless function
def get_acc_distance(points):
    #utilize Danymic programming calculate acc_points
    accumulated_points = points.copy()
    for i in range(1, len(points)):
        accumulated_points[i][0] = accumulated_points[i-1][0] + accumulated_points[i][0]
        accumulated_points[i][1] = accumulated_points[i-1][1] + accumulated_points[i][1]
    return accumulated_points

def get_straight_dist(dist_points_meters, actual_length):
    init = np.array([0, 0])
    distances = np.linalg.norm(dist_points_meters - init, axis=1)

    acc_dist = distances.copy()
    for i in range(1, len(distances)):
        acc_dist[i] = acc_dist[i-1] + acc_dist[i]
    dilation = acc_dist[len(acc_dist)-1] / actual_length
    distances = distances / dilation
    acc_dist = acc_dist / dilation
    return distances, acc_dist

def earned_money_count(input):
    global length, maximum_dist, minimum_dist, maximum_earned_money, money_per_year_per_KW ,year, wire_pole
    input = sorted(input)
    fitFunc = 0
    money_get = 0
    for i in range(1,len(input)):
        distance = abs(input[i] - input[i-1])
        if distance < minimum_dist:
            print(f"index:{i} isn't greater than minimum")
            print("Doesn't work out")
        elif distance <= maximum_dist:
            money_get += money_per_year_per_KW * map_value(distance, minimum_dist, maximum_dist, 1, 5)* year
        money_get += input[0]*money_per_year_per_KW * map_value(abs(input[1]-input[0]), minimum_dist, maximum_dist, 1, 5)* year
    
    for point in wire_pole:
        smallest_dist = 0
        for search_point in input:
            smallest_dist = min(abs(search_point - point), smallest_dist)
        
        fitFunc -= smallest_dist * ((smallest_dist//100)/10 + 1) #<100 =>1 ； > 100 * 階梯式

    return money_get 
   
def map_value(x, old_min, old_max, new_min, new_max):
        # 確保輸入值在舊範圍內
    x = max(min(x, old_max), old_min)
    # 變換公式
    return new_min + (x - old_min) * (new_max - new_min) / (old_max - old_min)

meters_per_degree_latitude = 111000 #經度換算公尺

#with open('point.xlsx', 'rb') as f:
 #   data = pd.read_excel(f)


points = np.vstack((data['經度'], data['緯度'])).T #原始座標
dist_points_meters = get_compared_dist(points, meters_per_degree_latitude) #相對座標
accumulated_points = get_acc_distance(dist_points_meters) #累計相對座標
actual_length = data['總距離'][0]

Straight_distance, accumulated_straight_dist = get_straight_dist(dist_points_meters, actual_length) #直線距離，累積距離

#================================read wire_pole================================
with open(os.path.join(main_file,'random_storage', str(args.wire_pole)+'.pkl'), 'rb') as f:
    rand_pole = pkl.load(f)

wire_pole = map_points_to_range(rand_pole, actual_length)


#取出起始點[0,0] 終點
init_pos = accumulated_straight_dist[0]
final_dist = accumulated_straight_dist[-1]

# #get distance
length = np.linalg.norm(final_dist - init_pos) 
init_pos = 0
final_dist = length

print('================================================================')
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('your cost is:', cost_money)
print('distance :', length)
print('machine amount :', machine_amount)
print('================================================================')
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
# # run

#main function
X = sampleGeneartor(length)
LB = np.asarray([0]*machine_amount)
UB = np.asarray([length]*machine_amount)

fitnessCurve, para = ppso(X, dim=machine_amount, pop_size=60, max_iter=2000, LB = LB, UB = UB,w=0.5, c1=2, c2=2).opt()

final_ans = sorted(para)
print(final_ans)
earned_money = earned_money_count(final_ans) - cost_money
print(f"earned moeny is: {earned_money}")

plot_converge_curve(fitnessCurve, storage_name = png_file) 
store = {
    "earned_money": earned_money,
    "final_ans": final_ans,
    "record_cost":fitnessCurve
}
with open(pkl_file, 'wb') as f:
    pkl.dump(store, f)