import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 读取最终盲测数据
file_name = 'D:\MSc_Project\Evaluate\T6_Evaluation_Data.csv'

if os.path.exists(file_name):
    df = pd.read_csv(file_name)

    # 设置顶级学术期刊的图表风格
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # ==========================================
    # 1. 结局分布对比图 (Outcome Distribution)
    # ==========================================
    outcomes = df.groupby(['Model', 'Outcome']).size().unstack(fill_value=0)
    # 确保列名完整
    for col in ['Collision', 'Timeout', 'Goal Reached']:
        if col not in outcomes.columns:
            outcomes[col] = 0

    outcomes_pct = outcomes[['Collision', 'Timeout', 'Goal Reached']].div(outcomes.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    outcomes_pct.plot(kind='bar', stacked=False, ax=ax, color=['#e74c3c', '#f39c12', '#2ecc71'])
    plt.title('Final Evaluation (60k Steps): Outcomes Comparison', fontsize=15, fontweight='bold', pad=15)
    plt.ylabel('Percentage (%)', fontsize=13)
    plt.xlabel('Model Architecture', fontsize=13)
    plt.xticks(rotation=0)
    plt.legend(title='Outcome', loc='upper right', frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig('T6_Final_Outcomes_60k.png')
    plt.close()

    # ==========================================
    # 2. 存活步数箱型图 (Survival Steps)
    # ==========================================
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    sns.boxplot(x='Model', y='Survival_Steps', data=df, palette='Set2', width=0.5, ax=ax)
    plt.title('Final Evaluation (60k Steps): Survival Steps Distribution', fontsize=15, fontweight='bold', pad=15)
    plt.ylabel('Steps Survived (Max 500)', fontsize=13)
    plt.xlabel('Model Architecture', fontsize=13)
    plt.tight_layout()
    plt.savefig('T6_Final_Survival_60k.png')
    plt.close()

    # ==========================================
    # 3. 路径效率柱状图 (Path Efficiency)
    # ==========================================
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    sns.barplot(x='Model', y='Path_Efficiency', data=df, palette='Set1', capsize=.1, width=0.5, ax=ax)
    plt.title('Final Evaluation (60k Steps): Path Efficiency', fontsize=15, fontweight='bold', pad=15)
    plt.ylabel('Efficiency (Distance Progress per Step)', fontsize=13)
    plt.xlabel('Model Architecture', fontsize=13)
    plt.tight_layout()
    plt.savefig('T6_Final_Efficiency_60k.png')
    plt.close()

    print("🎉 恭喜！针对 60,000 步最终训练数据的三张高清对比图已生成完毕！")
else:
    print(f"❌ 找不到文件：{file_name}，请确保该文件与脚本在同一目录下。")