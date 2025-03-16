import subprocess
import numpy as np

# 定义 Python 解释器和脚本名称
python_executable = "python"
script_name = "new_pso.py"
cost = 380000
spend_money = [1e7, 1.5e7, 2e7]
machine_amount_stack = np.asarray([int(money // cost) for money in spend_money])

# 将 spend_money 转换为字符串格式
spend_money = ['1e7', '1.5e7', '2e7']
print(spend_money)

# 定义 machine_amount 的范围和 wire_pole 的数量
wire_pole = [6, 12, 18]

# 遍历所有 machine_amount 和 wire_pole 的组合
for idx, machine_amount in enumerate(machine_amount_stack):
    for i in range(2, machine_amount):
        for j in wire_pole:
            # 构建命令和参数，添加 -u 参数来实时输出
            command = [python_executable, "-u", script_name, "--machine_amount", str(i), "--wire_pole", str(j), "--money_spend", str(spend_money[idx])]
            try:
                # 实时显示输出
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                # 逐行读取并输出结果
                for line in process.stdout:
                    print(line, end="")

                for line in process.stderr:
                    print(line, end="")

                process.wait()  # 等待子进程结束
                print(f"运行成功: machine_amount={i}, wire_pole={j}")

            except subprocess.CalledProcessError as e:
                print(f"错误发生在: machine_amount={i}, wire_pole={j}")
                print("错误输出:", e.stderr)  # 打印标准错误输出
