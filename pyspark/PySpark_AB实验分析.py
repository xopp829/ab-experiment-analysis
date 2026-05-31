"""
PySpark 实战：电商AB实验数据分析
===================================
场景：用PySpark重做你的AB实验分析项目
      展示Spark核心操作，面试可讲

环境：本地单机模式（pip install pyspark 即可运行）
数据：模拟生成29万条用户行为数据（与原项目结构一致）

运行方式：python PySpark_AB实验分析.py
"""

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
import random
import os

# ============================================================
# 第0步：启动Spark
# ============================================================
# SparkSession是Spark 2.0+的统一入口
# local[*] 表示用本机所有CPU核心，local[2]表示2个核心
# .config("spark.driver.memory", "2g") 可限制内存

spark = SparkSession.builder \
    .appName("AB_Experiment_Analysis") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

# 设置日志级别，减少无关输出
spark.sparkContext.setLogLevel("WARN")

print("=" * 60)
print("Spark版本:", spark.version)
print("Spark UI:", spark.sparkContext.uiWebUrl)  # 可在浏览器查看任务执行情况
print("=" * 60)


# ============================================================
# 第1步：生成模拟数据（实际工作中从HDFS/S3/数据库读取）
# ============================================================
print("\n>>> 第1步：生成模拟数据")

random.seed(42)
n = 290000  # 29万条，与原项目一致

rows = []
for i in range(n):
    user_id = f"user_{i}"
    group = random.choice(["control", "treatment"])  # 分流：对照组 vs 实验组
    page = "new_page" if group == "treatment" else "old_page"
    # 控制组转化率12%，实验组转化率12.3%（差异很小，与原项目结论一致）
    if group == "control":
        converted = 1 if random.random() < 0.120 else 0
        time_spent = float(max(0, random.gauss(180, 120)))  # 耗时（秒），非正态
    else:
        converted = 1 if random.random() < 0.123 else 0
        time_spent = float(max(0, random.gauss(185, 125)))
    
    rows.append((user_id, group, page, converted, round(time_spent, 1)))

schema = StructType([
    StructField("user_id", StringType(), False),
    StructField("group", StringType(), False),
    StructField("landing_page", StringType(), False),
    StructField("converted", IntegerType(), False),
    StructField("time_spent", DoubleType(), False),
])

df = spark.createDataFrame(rows, schema)

print(f"数据量: {df.count()} 行")
df.show(5)
df.printSchema()


# ============================================================
# 第2步：数据校验（面试重点：AB实验数据质量检查）
# ============================================================
print("\n>>> 第2步：数据校验")

# 2.1 检查分组与页面一致性（原项目关键校验点）
inconsistent = df.filter(
    ~((F.col("group") == "control") & (F.col("landing_page") == "old_page") |
      (F.col("group") == "treatment") & (F.col("landing_page") == "new_page"))
)
print(f"分流不一致记录数: {inconsistent.count()}")

# 2.2 检查重复user_id
dup_count = df.groupBy("user_id").count().filter(F.col("count") > 1).count()
print(f"重复user_id数: {dup_count}")

# 2.3 检查缺失值（Spark写法）
df.select([F.count(F.when(F.isnull(c), c)).alias(c) for c in df.columns]).show()


# ============================================================
# 第3步：分组统计（核心操作：groupBy + agg）
# ============================================================
print("\n>>> 第3步：分组统计")

# 这是Spark最常用的操作模式
group_stats = df.groupBy("group").agg(
    F.count("*").alias("total_users"),
    F.sum("converted").alias("conversions"),
    F.round(F.avg("converted") * 100, 2).alias("conversion_rate_%"),
    F.round(F.avg("time_spent"), 1).alias("avg_time_spent"),
    F.round(F.stddev("time_spent"), 1).alias("std_time_spent"),
    F.round(F.percentile_approx("time_spent", 0.5), 1).alias("median_time_spent"),
)
group_stats.show()

# 面试知识点：
# - groupBy会触发shuffle（数据按key重新分发到不同分区）
# - agg中可以用任意聚合函数
# - percentile_approx是近似分位数，比精确分位数快很多（大数据场景必用）


# ============================================================
# 第4步：假设检验（Spark + Python结合）
# ============================================================
print("\n>>> 第4步：假设检验")

# 方式1：Spark算出统计量，Python做检验（生产环境常用模式）
stats = group_stats.collect()  # collect()把结果从集群拉到driver
stats_dict = {row["group"]: row for row in stats}

n_control = stats_dict["control"]["total_users"]
n_treatment = stats_dict["treatment"]["total_users"]
x_control = stats_dict["control"]["conversions"]
x_treatment = stats_dict["treatment"]["conversions"]

p_control = x_control / n_control
p_treatment = x_treatment / n_treatment
p_pool = (x_control + x_treatment) / (n_control + n_treatment)

import math
se = math.sqrt(p_pool * (1 - p_pool) * (1/n_control + 1/n_treatment))
z = (p_treatment - p_control) / se
print(f"对照组转化率: {p_control:.4f}")
print(f"实验组转化率: {p_treatment:.4f}")
print(f"Z统计量: {z:.4f}")
if abs(z) > 1.96:
    print(f"差异显著（|Z| > 1.96, p < 0.05），但效应量极小（diff={p_treatment-p_control:.4f}），业务意义有限")
else:
    print(f"差异不显著（|Z| < 1.96）")

# 方式2：Spark SQL方式（面试加分：你会写Spark SQL）
df.createOrReplaceTempView("ab_test")

sql_result = spark.sql("""
    SELECT 
        'control' as group_name,
        COUNT(*) as total,
        SUM(converted) as conversions,
        ROUND(AVG(converted) * 100, 2) as cvr_pct
    FROM ab_test 
    WHERE `group` = 'control'
    UNION ALL
    SELECT 
        'treatment' as group_name,
        COUNT(*) as total,
        SUM(converted) as conversions,
        ROUND(AVG(converted) * 100, 2) as cvr_pct
    FROM ab_test 
    WHERE `group` = 'treatment'
""")
sql_result.show()


# ============================================================
# 第5步：RDD操作（面试必考：RDD vs DataFrame）
# ============================================================
print("\n>>> 第5步：RDD操作")

# RDD是Spark最底层的抽象，DataFrame是高级API
# 面试常问：RDD和DataFrame的区别？
# 答：RDD无schema，无法被Catalyst优化器优化；DataFrame有schema，可被优化

rdd = df.rdd

# map: 逐行转换
conversion_by_group = rdd \
    .map(lambda row: (row["group"], (row["converted"], 1))) \
    .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])) \
    .mapValues(lambda x: round(x[0] / x[1] * 100, 2)) \
    .collect()

print("RDD方式计算转化率:", conversion_by_group)

# 面试知识点：
# - map: 一对一转换
# - flatMap: 一对多转换
# - reduceByKey: 按key聚合（会先在分区内局部聚合，再跨分区全局聚合，比groupByKey高效）
# - collect: 把RDD数据拉到driver，大数据量别用！


# ============================================================
# 第6步：耗时分析 + 缓存优化
# ============================================================
print("\n>>> 第6步：耗时分析")

# 6.1 按耗时分组分析转化率（看决策时间效应）
df_with_bucket = df.withColumn(
    "time_bucket",
    F.when(F.col("time_spent") < 60, "0-1min")
     .when(F.col("time_spent") < 180, "1-3min")
     .when(F.col("time_spent") < 300, "3-5min")
     .otherwise("5min+")
)

# cache()：面试高频题——什么时候该cache？
# 答：当一个DataFrame被多次使用时（下面用了两次）
df_with_bucket.cache()

bucket_stats = df_with_bucket.groupBy("time_bucket", "group").agg(
    F.count("*").alias("n"),
    F.round(F.avg("converted") * 100, 2).alias("cvr_%"),
)
bucket_stats.orderBy("time_bucket", "group").show()

# 6.2 同一份数据的第二次使用，命中缓存
bucket_cvr = df_with_bucket.groupBy("time_bucket").agg(
    F.round(F.avg("converted") * 100, 2).alias("overall_cvr_%")
).orderBy("time_bucket")
bucket_cvr.show()

# 释放缓存
df_with_bucket.unpersist()

# 面试知识点：
# - cache() = persist(MEMORY_ONLY)，数据存在内存
# - persist(StorageLevel.MEMORY_AND_DISK)：内存放不下会溢写到磁盘
# - 必须在action触发后才会真正缓存
# - 用完unpersist()释放，避免内存泄漏


# ============================================================
# 第7步：写入与读取（生产环境核心操作）
# ============================================================
print("\n>>> 第7步：数据读写")

output_dir = "./PySpark实战教程/output_ab_test"

# 写入Parquet格式（列式存储，Spark生产环境首选）
df.write \
    .mode("overwrite") \
    .parquet(f"{output_dir}/parquet")

# 写入CSV
df.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv(f"{output_dir}/csv")

# 按分组分区写入（分区裁剪优化查询速度）
df.write \
    .mode("overwrite") \
    .partitionBy("group") \
    .parquet(f"{output_dir}/partitioned")

# 读取Parquet（比CSV快很多，保留schema）
df_read = spark.read.parquet(f"{output_dir}/parquet")
print(f"读取Parquet: {df_read.count()} 行")

# 分区裁剪：只读treatment分区，不会扫描control数据
df_treatment = spark.read.parquet(f"{output_dir}/partitioned/group=treatment")
print(f"分区裁剪读取treatment: {df_treatment.count()} 行")

# 面试知识点：
# - Parquet vs CSV：Parquet列式存储、压缩率高、保留schema、支持分区裁剪
# - partitionBy：按列值分目录存储，查询时可以跳过不相关分区
# - overwrite vs append：覆盖 vs 追加


# ============================================================
# 第8步：窗口函数（SQL面试高频）
# ============================================================
print("\n>>> 第8步：窗口函数")

from pyspark.sql.window import Window

# 场景：按耗时排名，看各分组中耗时Top5用户
w = Window.partitionBy("group").orderBy(F.desc("time_spent"))

df_ranked = df.withColumn("rank", F.row_number().over(w)) \
    .filter(F.col("rank") <= 5) \
    .select("user_id", "group", "time_spent", "converted", "rank")

df_ranked.show()

# 面试知识点：
# - row_number()：1,2,3,4,5 不重复
# - rank()：1,2,2,4 并列跳号
# - dense_rank()：1,2,2,3 并列不跳号
# - Window.partitionBy() 相当于 SQL的 PARTITION BY
# - orderBy + rangeBetween/rowsBetween 可以做移动平均、累计求和


# ============================================================
# 第9步：Spark SQL实现完整分析（面试展示）
# ============================================================
print("\n>>> 第9步：Spark SQL完整分析")

df.createOrReplaceTempView("ab_data")

# 用一条SQL完成分组统计 + 差异计算
full_analysis = spark.sql("""
WITH group_stats AS (
    SELECT 
        `group`,
        COUNT(*) as n,
        SUM(converted) as x,
        AVG(converted) as p,
        AVG(time_spent) as avg_time,
        STDDEV(time_spent) as std_time
    FROM ab_data
    GROUP BY `group`
),
control AS (
    SELECT * FROM group_stats WHERE `group` = 'control'
),
treatment AS (
    SELECT * FROM group_stats WHERE `group` = 'treatment'
)
SELECT 
    c.n as control_n,
    c.p as control_cvr,
    t.n as treatment_n,
    t.p as treatment_cvr,
    t.p - c.p as diff,
    CASE WHEN ABS((t.p - c.p) / SQRT(
        (c.x + t.x) / (c.n + t.n) * (1 - (c.x + t.x) / (c.n + t.n)) * (1.0/c.n + 1.0/t.n)
    )) > 1.96 THEN '显著' ELSE '不显著' END as significance
FROM control c CROSS JOIN treatment t
""")
full_analysis.show()


# ============================================================
# 清理
# ============================================================
print("\n>>> 完成！关闭Spark")
spark.stop()
