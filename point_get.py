import pandas as pd

# 讀取 Excel 檔案
file_path = "20.xlsx"  # 確保這個檔案在你的當前目錄
xls = pd.ExcelFile(file_path)

# 讀取特定工作表 (確認工作表名稱)
sheet_name = xls.sheet_names[0]  # 讀取第一個工作表
df = pd.read_excel(xls, sheet_name=sheet_name)

# 提取經緯度數據
points = df[['經度', '緯度']].values.tolist()

# 顯示所有點
for idx, (lon, lat) in enumerate(points, start=1):
    print(f"點 {idx}: 經度={lon}, 緯度={lat}")

