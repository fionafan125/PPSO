import numpy as np
import math
import random
import pandas as pd
from math import cos, radians
import matplotlib.pyplot as plt
import pickle as pkl
import argparse
import os
import multiprocessing as mp
from src import *
from multiprocessing import Pool
import matplotlib


# with open('point.xlsx', 'rb') as f:
#     data = pd.read_excel(f, engine="openpyxl")
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
penalty_base = 1e5
minimum_dist = 40  # 設定機器之間的最小距離
maximum_dist = 50  # 設定機器之間的最大距離
KW_volumn = 5.1  # 機器每年產生的 KW 量
year = 10  # 年限
store_result_file = 'result_file'

if not os.path.exists(store_result_file):
    os.makedirs(store_result_file)

storage_name = 'storage_name.pkl'

# == 設定參數解析及最大機器數量計算 ==
parser = argparse.ArgumentParser(description="加參數")
parser.add_argument('--machine_amount', type=int, help="機器數量實驗")
parser.add_argument('--wire_pole', type=int, help="random type")
parser.add_argument('--money_spend', type=str, help="record number")
args = parser.parse_args()
wire_pole = args.wire_pole
tmp = os.path.join(main_file, store_result_file, 'money_' + str(args.money_spend))
if not os.path.exists(tmp):
    os.mkdir(tmp)

store_file = os.path.join(tmp, 'wire_pole_' + str(args.wire_pole))

if not os.path.exists(store_file):
    os.makedirs(store_file)

store_file = os.path.join(store_file, 'machine_amount_' + str(args.machine_amount))
print(store_file)
if not os.path.exists(store_file):
    os.makedirs(store_file)

pkl_file = os.path.join(store_file, storage_name)
png_file = os.path.join(store_file, 'converge_curve.png')
year = 10
iterations = 500  # 迭代次數
record_cost = []
maximum_dist = 50
minimum_dist = 40
KW_volumn = 5.1
#LB = np.asarray([0] * machine_amount)  # 下界
#UB = np.asarray([length] * machine_amount)  # 上界
LB = 0  # 假設機器數量為 2
UB = actual_length  # 設定上限範圍

#===gain
money_per_year_per_KW = 38551.68
machine_amount = args.machine_amount
maximum_GetBack_money = money_per_year_per_KW * KW_volumn * year * machine_amount
cost = 380000
cost_money = cost * KW_volumn * machine_amount
maximum_earned_money = maximum_GetBack_money - cost_money

if maximum_GetBack_money < cost_money:
    print("QAQ")
    exit()
#================================================================

def sampleGeneartor(actual_length):
    X = np.arange(0, actual_length, 0.01)
    return X

class pso():
    def __init__(self, X_train, LB, UB, dim=4, pop_size=20, max_iter=2000, w=0.5, c1=2, c2=2):
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
        self.X = np.random.uniform(LB + minimum_dist, UB - maximum_dist, (pop_size, dim))
        self.X = np.clip(self.X, LB + minimum_dist, UB - maximum_dist)
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
        global minimum_dist, maximum_dist, penalty_base
        
        fitness = 0
        for i in range(1, len(input)):
            distance = abs(input[i] - input[i - 1])
            
            # 若距離過小或過大，加入適當懲罰
            if distance < minimum_dist:
                penalty = penalty_base * (minimum_dist - distance)
                fitness += min(penalty, 1e6)
            elif distance > maximum_dist:
                penalty = penalty_base * (distance - maximum_dist)
                fitness += min(penalty, 1e6)

        # 檢查是否返回了無效值
        if math.isnan(fitness) or fitness > 1e6:
            print(f"Warning: Invalid fitness detected. Input: {input}, Fitness: {fitness}")
            fitness = 1e6  # 設置一個最大值作為保護
        
        return fitness


    
    # 並行處理適應度和更新
    def parallel_optimize(self, group):
        for i in group:
            try:
                # 限制邊界
                self.X[i, :] = np.clip(self.X[i, :], self.LB, self.UB)
                # 計算適應度值
                fitness = self.fitFunc(self.X[i, :])
                if fitness is None:
                    raise ValueError("Fitness function returned None")

                # 更新個體最佳位置
                if fitness < self.pBest_score[i]:  # 確保縮排
                    self.pBest_score[i] = fitness
                    self.pBest_X[i, :] = self.X[i, :].copy()

                # 更新全局最佳位置
                if group[0] < self.group_size:  # 組1
                    if fitness < self.gBest_score_group1:  # 確保縮排
                        self.gBest_score_group1 = fitness
                        self.gBest_X_group1 = self.X[i, :].copy()
                else:  # 組2
                    if fitness < self.gBest_score_group2:  # 確保縮排
                        self.gBest_score_group2 = fitness
                        self.gBest_X_group2 = self.X[i, :].copy()
            except Exception as e:
                print(f"Error during optimization for particle {i}: {e}")

    # 更新速度和位置
    def update_velocity_position(self, group, gBest_X):
        for i in group:
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)

            # 更新速度
            self.V[i, :] = (
                self.w * self.V[i, :] +
                self.c1 * r1 * (self.pBest_X[i, :] - self.X[i, :]) +
                self.c2 * r2 * (gBest_X - self.X[i, :])
            )

            # 限制速度，避免更新過於激進
            self.V[i, :] = np.clip(self.V[i, :], -self.UB, self.UB)

            # 更新位置
            self.X[i, :] += self.V[i, :]
            self.X[i, :] = np.clip(self.X[i, :], self.LB, self.UB)
        
        # 確保 group 傳入並正確使用
        print(f"Updated Positions for Group: {list(group)}\n{self.X[group, :]}")
        print(f"Updated Velocities for Group: {list(group)}\n{self.V[group, :]}")



    # 主優化模塊
    def opt(self):
        for t in range(self.max_iter):
            self.w = 0.9 - (t / self.max_iter) * 0.5  # 動態調整權重
            
            group1 = range(0, self.group_size)
            group2 = range(self.group_size, self.pop_size)

            with Pool(2) as pool:
                pool.starmap(self.parallel_optimize, [(group1,), (group2,)])

            if self.gBest_score_group1 < self.gBest_score_group2:
                self.gBest_X_group2 = self.gBest_X_group1
                self.gBest_score_group2 = self.gBest_score_group1
            else:
                self.gBest_X_group1 = self.gBest_X_group2
                self.gBest_score_group1 = self.gBest_score_group2

            with Pool(2) as pool:
                pool.starmap(self.update_velocity_position, [
                    (group1, self.gBest_X_group1),
                    (group2, self.gBest_X_group2)
                ])

            self.gBest_curve[t] = min(self.gBest_score_group1, self.gBest_score_group2)
            
            # 打印診斷信息
            print(f"Iteration {t}: Best Fitness: {self.gBest_curve[t]}")
            print(f"Group 1 Best Position: {self.gBest_X_group1}")
            print(f"Group 2 Best Position: {self.gBest_X_group2}")

        return self.gBest_curve, (self.gBest_X_group1 if self.gBest_score_group1 < self.gBest_score_group2 else self.gBest_X_group2)

def plot_converge_curve(record_cost, storage_name):
    record_cost = np.array(record_cost)
    record_cost = np.where(record_cost > 1e6, 1e6, record_cost)
    iters = np.arange(1, len(record_cost) + 1)
    plt.plot(iters, record_cost, marker='o', markersize=4)  # 減小標記大小

    plt.title("PSO Convergence Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness Score")
    plt.savefig(storage_name)

def compute_dis_xy(init_pos, final_pos, meter_per_degree_latitude):
    # 計算南北方向上的距離（Δ緯度 × 每度緯度的公尺數）
    delta_latitude = final_pos[1] - init_pos[1]
    distance_north_south = delta_latitude * meters_per_degree_latitude

    # 計算東西方向上的距離（Δ經度 × cos(平均緯度) × 每度緯度的公尺數）
    average_latitude = radians((init_pos[1] + final_pos[1]) / 2)
    delta_longitude = final_pos[0] - init_pos[0]
    distance_east_west = delta_longitude * cos(average_latitude) * meters_per_degree_latitude
    return [distance_north_south, (-1) * distance_east_west]

def get_compared_dist(points, meters_per_degree_latitude):
    compared_dist = []
    for point in points:
        tmp_dist = compute_dis_xy(points[0, :], point, meters_per_degree_latitude)
        compared_dist.append(tmp_dist)

    compared_dist = np.array(compared_dist)
    return compared_dist

# 無用的函數
def get_acc_distance(points):
    # 利用動態規劃計算累計點
    accumulated_points = points.copy()
    for i in range(1, len(points)):
        accumulated_points[i][0] = accumulated_points[i - 1][0] + accumulated_points[i][0]
        accumulated_points[i][1] = accumulated_points[i - 1][1] + accumulated_points[i][1]
    return accumulated_points

def get_straight_dist(dist_points_meters, actual_length):
    init = np.array([0, 0])
    distances = np.linalg.norm(dist_points_meters - init, axis=1)

    acc_dist = distances.copy()
    for i in range(1, len(distances)):
        acc_dist[i] = acc_dist[i - 1] + acc_dist[i]
    dilation = acc_dist[len(acc_dist) - 1] / actual_length
    distances = distances / dilation
    acc_dist = acc_dist / dilation
    return distances, acc_dist

def earned_money_count(input):
    global actual_length, maximum_dist, minimum_dist, maximum_earned_money, money_per_year_per_KW, year, wire_pole
    input = sorted(input)
    fitFunc = 0
    money_get = 0

    for i in range(1, len(input)):
        distance = abs(input[i] - input[i - 1])
        if distance < minimum_dist:
            fitness += penalty_base * (minimum_dist - distance)
        elif distance <= maximum_dist:
            money_get += money_per_year_per_KW * map_value(distance, minimum_dist, maximum_dist, 1, 5) * year

    for point in wire_pole:
        smallest_dist = float('inf')
        for search_point in input:
            smallest_dist = min(abs(search_point - point), smallest_dist)

        fitness += min(smallest_dist, 1e6)

    return max(money_get - cost_money - fitness, -1e10)
    
    def earned_money_count(input):
        global actual_length, maximum_dist, minimum_dist, maximum_earned_money, money_per_year_per_KW, year, wire_pole, penalty_base
        input = sorted(input)
        fitness = 0  # 初始化 fitness
        money_get = 0

        for i in range(1, len(input)):
            distance = abs(input[i] - input[i - 1])
            if distance < minimum_dist:
                fitness += penalty_base * (minimum_dist - distance)
            elif distance <= maximum_dist:
                money_get += money_per_year_per_KW * map_value(distance, minimum_dist, maximum_dist, 1, 5) * year

        for point in wire_pole:
            smallest_dist = float('inf')
            for search_point in input:
                smallest_dist = min(abs(search_point - point), smallest_dist)

            fitness += min(smallest_dist, 1e6)

    return max(money_get - cost_money - fitness, -1e10)


def map_value(x, old_min, old_max, new_min, new_max):
    # 確保輸入值在舊範圍內
    x = max(min(x, old_max), old_min)
    # 變換公式
    return new_min + (x - old_min) * (new_max - new_min) / (old_max - old_min)

meters_per_degree_latitude = 111000  # 經度換算公尺


points = np.vstack((data['longitude'], data['latitude'])).T  # 原始座標
dist_points_meters = get_compared_dist(points, meters_per_degree_latitude)  # 相對座標
accumulated_points = get_acc_distance(dist_points_meters)  # 累計相對座標

Straight_distance, accumulated_straight_dist = get_straight_dist(dist_points_meters, actual_length)  # 直線距離，累積距離

#================================read wire_pole================================
with open(os.path.join(main_file, 'random_storage', str(args.wire_pole) + '.pkl'), 'rb') as f:
    rand_pole = pkl.load(f)
wire_pole = map_points_to_range(rand_pole, actual_length)

# 取出起始點[0,0] 終點
init_pos = accumulated_straight_dist[0]
final_dist = accumulated_straight_dist[-1]

# 計算距離
actual_length = np.linalg.norm(final_dist - init_pos)
init_pos = 0
final_dist = actual_length

print('================================================================')
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('your cost is:', cost_money)
print('distance :', actual_length)
print('machine amount :', machine_amount)
print('================================================================')
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
X = sampleGeneartor(actual_length)
LB = 0  # 假設機器數量為 2
UB = actual_length  # 設定上限範圍

fitnessCurve, para = pso(X, dim=machine_amount, pop_size=60, max_iter=2000, LB = LB, UB = UB,w=0.5, c1=2, c2=2).opt()

final_ans = sorted(para)
print("Optimal Parameters (Final Answer):", final_ans)
earned_money = earned_money_count(final_ans) - cost_money
print(f"Earned Money: {earned_money}")

plot_converge_curve(fitnessCurve, storage_name=png_file)
store = {
    "earned_money": earned_money,
    "final_ans": final_ans,
    "record_cost": fitnessCurve
}
with open(pkl_file, 'wb') as f:
    pkl.dump(store, f)

