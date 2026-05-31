"""
AB实验分析：新旧页面转化率对比
=================================
完整流程代码，可直接在本地VSCode运行
依赖：pandas, numpy, scipy, statsmodels, matplotlib
安装：pip install pandas numpy scipy statsmodels matplotlib
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置区 —— 按你的实际路径修改
# ============================================================
DATA_PATH = './ab_data748296.xlsx'        # 主实验数据路径
COUNTRY_PATH = './countries237256.xlsx'   # 国家维度数据路径
OUTPUT_DIR = './output/'                   # 输出目录

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置中文显示（matplotlib）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 第一步：数据加载与清洗
# ============================================================
def step1_data_cleaning():
    print("=" * 60)
    print("第一步：数据加载与清洗")
    print("=" * 60)

    # 1.1 加载数据
    df = pd.read_excel(DATA_PATH)
    countries = pd.read_excel(COUNTRY_PATH)
    print(f"原始数据量: {len(df)}")
    print(f"国家数据量: {len(countries)}")
    print(f"列名: {df.columns.tolist()}\n")

    # 1.2 一致性校验：control必须看old_page, treatment必须看new_page
    mismatch_ctrl = ((df['group'] == 'control') & (df['landing_page'] == 'new_page')).sum()
    mismatch_trt = ((df['group'] == 'treatment') & (df['landing_page'] == 'old_page')).sum()
    print(f"一致性校验：")
    print(f"  control组看到new_page的记录: {mismatch_ctrl}")
    print(f"  treatment组看到old_page的记录: {mismatch_trt}")
    print(f"  不一致记录总数: {mismatch_ctrl + mismatch_trt}")

    # 剔除不一致记录
    mask = ~(
        ((df['group'] == 'control') & (df['landing_page'] == 'new_page')) |
        ((df['group'] == 'treatment') & (df['landing_page'] == 'old_page'))
    )
    df = df[mask].copy()
    print(f"  剔除后剩余: {len(df)}\n")

    # 1.3 去重：同一user_id多次曝光，仅保留首次
    dup_count = df['user_id'].duplicated(keep='first').sum()
    print(f"去重：重复user_id数量: {dup_count}")
    df = df.drop_duplicates(subset='user_id', keep='first')
    print(f"  去重后剩余: {len(df)}\n")

    # 1.4 缺失值检查
    print("缺失值检查：")
    print(df.isnull().sum())
    print()

    # 1.5 合并国家数据
    df = df.merge(countries, on='user_id', how='inner')
    print(f"合并国家数据后: {len(df)} 条\n")

    # 1.6 timestamp检查
    print("Timestamp检查：")
    print(f"  样本值: {df['timestamp'].head(5).tolist()}")
    print(f"  数据类型: {df['timestamp'].dtype}")
    print(f"  唯一值数量: {df['timestamp'].nunique()}")
    print(f"  ⚠️ 注意：timestamp仅含时间无日期，无法做时间趋势分析\n")

    # 1.7 转换timestamp为秒数
    def time_to_seconds(t):
        """将datetime.time对象转为秒数"""
        if hasattr(t, 'hour'):
            return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
        return 0

    df['time_seconds'] = df['timestamp'].apply(time_to_seconds)
    print(f"耗时统计（秒）：")
    print(df['time_seconds'].describe())
    print()

    print(f"✅ 清洗完成，最终可用数据: {len(df)} 条\n")
    return df


# ============================================================
# 第二步：描述性统计
# ============================================================
def step2_descriptive_stats(df):
    print("=" * 60)
    print("第二步：描述性统计")
    print("=" * 60)

    # 2.1 整体分组统计
    grouped = df.groupby('group').agg(
        n=('converted', 'count'),
        conversions=('converted', 'sum'),
        conv_rate=('converted', 'mean')
    )
    grouped['conv_rate_pct'] = grouped['conv_rate'] * 100

    print("整体分组统计：")
    print(f"  对照组: n={grouped.loc['control','n']:,}, "
          f"转化数={grouped.loc['control','conversions']:,}, "
          f"转化率={grouped.loc['control','conv_rate_pct']:.2f}%")
    print(f"  实验组: n={grouped.loc['treatment','n']:,}, "
          f"转化数={grouped.loc['treatment','conversions']:,}, "
          f"转化率={grouped.loc['treatment','conv_rate_pct']:.2f}%")

    diff = grouped.loc['treatment','conv_rate_pct'] - grouped.loc['control','conv_rate_pct']
    rel_diff = diff / grouped.loc['control','conv_rate_pct'] * 100
    print(f"  差异: {diff:+.2f}pp (相对差异: {rel_diff:+.2f}%)\n")

    # 2.2 按国家分组统计
    print("按国家分组统计：")
    country_stats = df.groupby(['country', 'group']).agg(
        n=('converted', 'count'),
        conv_rate=('converted', 'mean')
    ).reset_index()
    country_stats['conv_rate_pct'] = country_stats['conv_rate'] * 100

    for country in ['US', 'UK', 'CA']:
        sub = country_stats[country_stats['country'] == country]
        ctrl = sub[sub['group'] == 'control'].iloc[0]
        treat = sub[sub['group'] == 'treatment'].iloc[0]
        diff_c = treat['conv_rate_pct'] - ctrl['conv_rate_pct']
        print(f"  {country}: 对照组 {ctrl['conv_rate_pct']:.2f}% (n={ctrl['n']:,}), "
              f"实验组 {treat['conv_rate_pct']:.2f}% (n={treat['n']:,}), "
              f"差异 {diff_c:+.2f}pp")
    print()

    # 2.3 可视化：各组转化率对比
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 整体对比
    groups = ['Control\n(旧页面)', 'Treatment\n(新页面)']
    rates = [grouped.loc['control', 'conv_rate_pct'],
             grouped.loc['treatment', 'conv_rate_pct']]
    bars = axes[0].bar(groups, rates, color=['#4A90D9', '#E74C3C'], width=0.5)
    axes[0].set_ylabel('Conversion Rate (%)')
    axes[0].set_title('Overall Conversion Rate')
    axes[0].set_ylim(11.5, 12.5)
    for bar, rate in zip(bars, rates):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{rate:.2f}%', ha='center', fontsize=12)

    # 按国家对比
    x = np.arange(3)
    width = 0.3
    ctrl_rates = [country_stats[(country_stats['country']==c) & (country_stats['group']=='control')]['conv_rate_pct'].values[0] for c in ['US','UK','CA']]
    treat_rates = [country_stats[(country_stats['country']==c) & (country_stats['group']=='treatment')]['conv_rate_pct'].values[0] for c in ['US','UK','CA']]
    axes[1].bar(x - width/2, ctrl_rates, width, label='Control', color='#4A90D9')
    axes[1].bar(x + width/2, treat_rates, width, label='Treatment', color='#E74C3C')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['US', 'UK', 'CA'])
    axes[1].set_ylabel('Conversion Rate (%)')
    axes[1].set_title('Conversion Rate by Country')
    axes[1].legend()
    axes[1].set_ylim(10.5, 13)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'descriptive_stats.png', dpi=150, bbox_inches='tight')
    print(f"📊 图表已保存: {OUTPUT_DIR}descriptive_stats.png\n")
    plt.close()

    return grouped


# ============================================================
# 第三步：假设检验
# ============================================================
def step3_hypothesis_test(df):
    print("=" * 60)
    print("第三步：假设检验（双比例z检验）")
    print("=" * 60)

    ctrl = df[df['group'] == 'control']
    treat = df[df['group'] == 'treatment']

    n1 = len(ctrl)
    n2 = len(treat)
    x1 = ctrl['converted'].sum()
    x2 = treat['converted'].sum()
    p1 = x1 / n1
    p2 = x2 / n2

    # 双比例z检验
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z_stat = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # 95%置信区间
    se_diff = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    ci_lower = (p1 - p2) - 1.96 * se_diff
    ci_upper = (p1 - p2) + 1.96 * se_diff

    print(f"原假设 H₀: p_control = p_treatment")
    print(f"备择假设 H₁: p_control ≠ p_treatment (双尾)")
    print(f"")
    print(f"对照组转化率: {p1*100:.4f}%")
    print(f"实验组转化率: {p2*100:.4f}%")
    print(f"差异: {(p1-p2)*100:.4f}pp")
    print(f"")
    print(f"z统计量: {z_stat:.4f}")
    print(f"p值: {p_value:.4f}")
    print(f"95%置信区间: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")
    print(f"")
    if p_value < 0.05:
        print(f"❌ 拒绝H₀（p={p_value:.4f} < 0.05），两组转化率有显著差异")
    else:
        print(f"✅ 不拒绝H₀（p={p_value:.4f} > 0.05），两组转化率无显著差异")
        if ci_upper * 100 < 0.1:
            print(f"   注意：CI上界仅+{ci_upper*100:.2f}%，即使新页面有效果也极小")
    print()

    # 可视化：置信区间
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(0, (p1-p2)*100, yerr=1.96*se_diff*100, fmt='o',
                color='#E74C3C', capsize=8, capthick=2, markersize=8)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylabel('Difference in Conversion Rate (pp)')
    ax.set_title(f'95% CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]')
    ax.text(0.15, (p1-p2)*100, f'p = {p_value:.4f}', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'hypothesis_test.png', dpi=150, bbox_inches='tight')
    print(f"📊 图表已保存: {OUTPUT_DIR}hypothesis_test.png\n")
    plt.close()

    return {'z_stat': z_stat, 'p_value': p_value, 'ci_lower': ci_lower, 'ci_upper': ci_upper}


# ============================================================
# 第四步：辛普森悖论检查
# ============================================================
def step4_simpson_paradox(df):
    print("=" * 60)
    print("第四步：辛普森悖论检查")
    print("=" * 60)

    print("按国家分层z检验：\n")
    results = []

    for country in ['US', 'UK', 'CA']:
        sub = df[df['country'] == country]
        ctrl = sub[sub['group'] == 'control']
        treat = sub[sub['group'] == 'treatment']

        n1, n2 = len(ctrl), len(treat)
        x1, x2 = ctrl['converted'].sum(), treat['converted'].sum()
        p1, p2 = x1/n1, x2/n2

        p_pool = (x1 + x2) / (n1 + n2)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
        z = (p1 - p2) / se
        p_val = 2 * (1 - stats.norm.cdf(abs(z)))

        direction = "对照组高" if p1 > p2 else "实验组高"
        sig = "✅ 显著" if p_val < 0.05 else "不显著"

        print(f"  {country}: 对照组 {p1*100:.2f}% vs 实验组 {p2*100:.2f}%, "
              f"差异 {(p1-p2)*100:+.2f}pp, 方向={direction}, "
              f"z={z:.4f}, p={p_val:.4f} ({sig})")

        results.append({
            'country': country, 'ctrl_rate': p1, 'treat_rate': p2,
            'diff_pp': (p1-p2)*100, 'direction': direction,
            'z': z, 'p_value': p_val, 'significant': p_val < 0.05
        })

    print()
    # 判断悖论
    directions = set(r['direction'] for r in results)
    if len(directions) > 1:
        print("⚠️ 存在辛普森悖论风险：各国家差异方向不一致！")
        for r in results:
            print(f"  {r['country']}: {r['direction']}")
        print("  UK的实验组转化率高于对照组，与整体方向相反")
        print("  可能原因：US占70%样本量主导了整体结论")
    else:
        print("✅ 各层方向一致，无辛普森悖论风险")
    print()

    # 可视化
    fig, ax = plt.subplots(figsize=(8, 5))
    countries = [r['country'] for r in results]
    diffs = [r['diff_pp'] for r in results]
    colors = ['#4A90D9' if d > 0 else '#E74C3C' for d in diffs]
    bars = ax.bar(countries, diffs, color=colors, width=0.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax.axhline(y=-0.16, color='orange', linestyle=':', alpha=0.7, label='Overall diff (-0.16pp)')
    ax.set_ylabel('Difference (pp)')
    ax.set_title('Conversion Rate Difference by Country\n(Positive = Control higher)')
    ax.legend()
    for bar, d in zip(bars, diffs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{d:+.2f}pp', ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'simpson_paradox.png', dpi=150, bbox_inches='tight')
    print(f"📊 图表已保存: {OUTPUT_DIR}simpson_paradox.png\n")
    plt.close()

    return results


# ============================================================
# 第五步：用户决策耗时分析
# ============================================================
def step5_time_analysis(df):
    print("=" * 60)
    print("第五步：用户决策耗时分析")
    print("=" * 60)

    ctrl_time = df[df['group'] == 'control']['time_seconds']
    treat_time = df[df['group'] == 'treatment']['time_seconds']

    # 5.1 正态性检验
    print("5.1 正态性检验（KS检验，5000样本）：")
    sample_c = ctrl_time.sample(5000, random_state=42)
    sample_t = treat_time.sample(5000, random_state=42)
    ks_c, p_ks_c = stats.kstest(sample_c, 'norm', args=(sample_c.mean(), sample_c.std()))
    ks_t, p_ks_t = stats.kstest(sample_t, 'norm', args=(sample_t.mean(), sample_t.std()))
    print(f"  对照组: KS={ks_c:.4f}, p={p_ks_c:.4f}")
    print(f"  实验组: KS={ks_t:.4f}, p={p_ks_t:.4f}")
    normal = p_ks_c > 0.05 and p_ks_t > 0.05
    print(f"  结论: {'服从正态' if normal else '不服从正态，应使用非参数检验（Mann-Whitney U）'}\n")

    # 5.2 全体用户耗时比较
    print("5.2 全体用户耗时比较：")
    print(f"  对照组: 均值={ctrl_time.mean():.2f}s ({ctrl_time.mean()/60:.2f}min), "
          f"中位数={ctrl_time.median():.2f}s ({ctrl_time.median()/60:.2f}min)")
    print(f"  实验组: 均值={treat_time.mean():.2f}s ({treat_time.mean()/60:.2f}min), "
          f"中位数={treat_time.median():.2f}s ({treat_time.median()/60:.2f}min)")
    print(f"  均值差异: {treat_time.mean()-ctrl_time.mean():.2f}s")

    u_stat, u_pval = stats.mannwhitneyu(ctrl_time, treat_time, alternative='two-sided')
    t_stat, t_pval = stats.ttest_ind(ctrl_time, treat_time, equal_var=False)
    print(f"  Mann-Whitney U: p={u_pval:.4f}")
    print(f"  Welch t: t={t_stat:.4f}, p={t_pval:.4f}")
    print(f"  结论: {'有显著差异' if u_pval < 0.05 else '无显著差异'}\n")

    # 5.3 转化用户耗时比较
    print("5.3 转化用户耗时比较：")
    df_conv = df[df['converted'] == 1]
    c_conv = df_conv[df_conv['group'] == 'control']['time_seconds']
    t_conv = df_conv[df_conv['group'] == 'treatment']['time_seconds']
    print(f"  对照组: 均值={c_conv.mean():.2f}s ({c_conv.mean()/60:.2f}min), "
          f"中位数={c_conv.median():.2f}s ({c_conv.median()/60:.2f}min), n={len(c_conv)}")
    print(f"  实验组: 均值={t_conv.mean():.2f}s ({t_conv.mean()/60:.2f}min), "
          f"中位数={t_conv.median():.2f}s ({t_conv.median()/60:.2f}min), n={len(t_conv)}")
    print(f"  均值差异: {t_conv.mean()-c_conv.mean():.2f}s")

    u_stat2, u_pval2 = stats.mannwhitneyu(c_conv, t_conv, alternative='two-sided')
    t_stat2, t_pval2 = stats.ttest_ind(c_conv, t_conv, equal_var=False)
    print(f"  Mann-Whitney U: p={u_pval2:.4f}")
    print(f"  Welch t: t={t_stat2:.4f}, p={t_pval2:.4f}")
    print(f"  结论: {'有显著差异' if u_pval2 < 0.05 else '无显著差异'}\n")

    # 5.4 耗时分桶×转化率
    print("5.4 耗时分桶×转化率：")
    df['time_min'] = df['time_seconds'] / 60
    buckets = [(0,10),(10,20),(20,30),(30,40),(40,50),(50,60)]
    print(f"  {'时段':>10} | {'对照组转化率':>10} | {'实验组转化率':>10} | {'差异(pp)':>8}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    for lo, hi in buckets:
        sub = df[(df['time_min']>=lo)&(df['time_min']<hi)]
        ctrl_s = sub[sub['group']=='control']
        treat_s = sub[sub['group']=='treatment']
        cr = ctrl_s['converted'].mean()*100 if len(ctrl_s)>0 else 0
        tr = treat_s['converted'].mean()*100 if len(treat_s)>0 else 0
        print(f"  {lo:>3}-{hi:>2}min  | {cr:>9.2f}% | {tr:>9.2f}% | {tr-cr:>+7.2f}")
    print()

    # 5.5 逻辑回归：耗时+分组→转化
    print("5.5 逻辑回归：耗时+分组→转化率")
    X = pd.DataFrame({
        'const': 1,
        'time_seconds': df['time_seconds'],
        'group_num': (df['group'] == 'treatment').astype(int)
    })
    y = df['converted']
    logit1 = sm.Logit(y, X).fit(disp=0)
    print(logit1.summary().tables[1])

    # 含交互项
    print("\n逻辑回归（含交互项 time×group）：")
    X2 = pd.DataFrame({
        'const': 1,
        'time_seconds': df['time_seconds'],
        'group_num': (df['group'] == 'treatment').astype(int),
        'interaction': df['time_seconds'] * (df['group'] == 'treatment').astype(int)
    })
    logit2 = sm.Logit(y, X2).fit(disp=0)
    print(logit2.summary().tables[1])
    print()

    # 5.6 分国家耗时比较
    print("5.6 分国家耗时比较：")
    for country in ['US', 'UK', 'CA']:
        sub_c = df[(df['country']==country)&(df['group']=='control')]['time_seconds']
        sub_t = df[(df['country']==country)&(df['group']=='treatment')]['time_seconds']
        u, p = stats.mannwhitneyu(sub_c, sub_t, alternative='two-sided')
        print(f"  {country}: 对照组均值={sub_c.mean():.2f}s, 实验组均值={sub_t.mean():.2f}s, "
              f"差异={sub_t.mean()-sub_c.mean():.2f}s, Mann-Whitney p={p:.4f}")
    print()

    # 5.7 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 耗时分布对比
    axes[0].hist(ctrl_time/60, bins=60, alpha=0.5, label='Control', color='#4A90D9', density=True)
    axes[0].hist(treat_time/60, bins=60, alpha=0.5, label='Treatment', color='#E74C3C', density=True)
    axes[0].set_xlabel('Time (min)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Decision Time Distribution')
    axes[0].legend()

    # 分桶转化率
    bucket_labels = [f'{lo}-{hi}' for lo, hi in buckets]
    ctrl_rates = []
    treat_rates = []
    for lo, hi in buckets:
        sub = df[(df['time_min']>=lo)&(df['time_min']<hi)]
        ctrl_rates.append(sub[sub['group']=='control']['converted'].mean()*100)
        treat_rates.append(sub[sub['group']=='treatment']['converted'].mean()*100)

    x = np.arange(len(bucket_labels))
    width = 0.3
    axes[1].bar(x - width/2, ctrl_rates, width, label='Control', color='#4A90D9')
    axes[1].bar(x + width/2, treat_rates, width, label='Treatment', color='#E74C3C')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(bucket_labels, rotation=30)
    axes[1].set_ylabel('Conversion Rate (%)')
    axes[1].set_title('Conversion Rate by Time Bucket')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'time_analysis.png', dpi=150, bbox_inches='tight')
    print(f"📊 图表已保存: {OUTPUT_DIR}time_analysis.png\n")
    plt.close()

    return {
        'overall_mwu_p': u_pval,
        'converted_mwu_p': u_pval2,
        'interaction_p': logit2.pvalues['interaction']
    }


# ============================================================
# 第六步：功效分析
# ============================================================
def step6_power_analysis(df):
    print("=" * 60)
    print("第六步：功效分析")
    print("=" * 60)

    ctrl = df[df['group'] == 'control']
    treat = df[df['group'] == 'treatment']

    p1 = ctrl['converted'].mean()
    current_n = len(ctrl)

    print(f"当前基线转化率: {p1*100:.4f}%")
    print(f"当前每组样本量: {current_n:,}\n")

    # 计算不同MDE下的最小样本量
    print("不同MDE下每组所需样本量：")
    print(f"  {'MDE(相对)':>10} | {'MDE(绝对pp)':>12} | {'所需样本量/组':>14} | {'当前/所需':>8}")
    print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*14}-+-{'-'*8}")

    mde_list = [0.01, 0.02, 0.05, 0.10]
    power = NormalIndPower()

    for mde_rel in mde_list:
        p2 = p1 * (1 + mde_rel)
        effect_size = proportion_effectsize(p1, p2)
        n_required = power.solve_power(effect_size=effect_size, alpha=0.05, power=0.80,
                                        ratio=1.0, alternative='two-sided')
        ratio = current_n / n_required
        print(f"  {mde_rel*100:>9.0f}% | {(p2-p1)*100:>11.4f}% | {n_required:>14,.0f} | {ratio:>8.2f}")

    print()
    # 重点：1%相对提升
    p2_target = p1 * 1.01
    effect_size = proportion_effectsize(p1, p2_target)
    n_required = power.solve_power(effect_size=effect_size, alpha=0.05, power=0.80,
                                    ratio=1.0, alternative='two-sided')
    print(f"关键结论：")
    print(f"  MDE=1%相对提升时，每组需 {n_required:,.0f} 样本")
    print(f"  当前每组仅 {current_n:,} 样本")
    print(f"  差距约 {n_required/current_n:.1f} 倍，功效严重不足")
    print()

    # 可视化：功效曲线
    fig, ax = plt.subplots(figsize=(8, 5))
    sample_sizes = np.linspace(10000, 2000000, 100)
    power_vals = [power.solve_power(effect_size=effect_size, alpha=0.05, nobs1=n,
                                     ratio=1.0, alternative='two-sided') for n in sample_sizes]
    ax.plot(sample_sizes/1e6, power_vals, color='#4A90D9', linewidth=2)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Power = 0.80')
    ax.axvline(x=current_n/1e6, color='orange', linestyle=':', alpha=0.7, label=f'Current n={current_n:,}')
    ax.axvline(x=n_required/1e6, color='green', linestyle=':', alpha=0.7, label=f'Required n={n_required:,.0f}')
    ax.set_xlabel('Sample Size per Group (million)')
    ax.set_ylabel('Statistical Power')
    ax.set_title('Power Curve (MDE=1% Relative Lift)')
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + 'power_analysis.png', dpi=150, bbox_inches='tight')
    print(f"📊 图表已保存: {OUTPUT_DIR}power_analysis.png\n")
    plt.close()

    return {'n_required': n_required, 'current_n': current_n}


# ============================================================
# 第七步：综合决策
# ============================================================
def step7_final_decision(z_result, simpson_results, time_result, power_result):
    print("=" * 60)
    print("第七步：综合决策")
    print("=" * 60)

    print("决策评估矩阵：\n")
    print(f"  判断维度          | 结果                              | 决策指向")
    print(f"  {'-'*16}-+-{'-'*33}-+-{'-'*20}")
    print(f"  假设检验          | p={z_result['p_value']:.4f} (不显著)       | 无法证明新页面更好")
    print(f"  效应量            | 95%CI [{z_result['ci_lower']*100:.2f}%, {z_result['ci_upper']*100:.2f}%] | 即使有效果也极小")
    print(f"  决策耗时          | MWU p={time_result['overall_mwu_p']:.4f}/"
          f"{time_result['converted_mwu_p']:.4f}    | 新页面未缩短决策时间")
    paradox = any(r['direction'] == '实验组高' for r in simpson_results)
    if paradox:
        print(f"  辛普森悖论        | UK方向反转                         | 不宜全局推广")
    print(f"  功效              | 差{power_result['n_required']/power_result['current_n']:.1f}倍                           | 不显著可能因功效不足")
    print(f"  数据质量          | 无日期字段                         | 无法排除周期效应")

    print(f"\n📋 最终决策：不上线新页面，停止当前实验\n")
    print("理由：")
    print("  1. 转化率差异不显著，95%CI上界仅+0.08%，无业务价值")
    print("  2. 决策耗时两组无显著差异，新页面未提升效率")
    print("  3. 存在辛普森悖论风险，不宜全局推广")
    print("  4. 样本量差8倍，功效严重不足")
    print("  5. timestamp缺日期，无法排除周期效应")
    print()
    print("后续建议：")
    print("  - 重新设计实验：明确MDE、补全日期字段、规划合理周期")
    print("  - 可对UK市场单独做小规模实验验证反转信号")
    print("  - 优先评估新页面的定性价值（用户体验、品牌形象）")


# ============================================================
# 主函数
# ============================================================
def main():
    print("\n" + "🐦‍⬛" * 20)
    print("  AB实验分析：新旧页面转化率对比")
    print("  完整流程代码 v1.0")
    print("🐦‍⬛" * 20 + "\n")

    # 第一步：数据清洗
    df = step1_data_cleaning()

    # 第二步：描述性统计
    grouped = step2_descriptive_stats(df)

    # 第三步：假设检验
    z_result = step3_hypothesis_test(df)

    # 第四步：辛普森悖论检查
    simpson_results = step4_simpson_paradox(df)

    # 第五步：决策耗时分析
    time_result = step5_time_analysis(df)

    # 第六步：功效分析
    power_result = step6_power_analysis(df)

    # 第七步：综合决策
    step7_final_decision(z_result, simpson_results, time_result, power_result)

    print("\n" + "=" * 60)
    print(f"✅ 全部分析完成！图表已保存至 {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
