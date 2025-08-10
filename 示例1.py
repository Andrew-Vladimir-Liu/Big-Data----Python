#演示如何在大数据场景下使用 Apache Spark（PySpark）做端到端的数据处理：从海量时间序列传感器数据中检测并量化“数据缝隙”（缺失区段），
# 并把结果输出为可供下游分析的指标表。代码包含数据读取、时间对齐、缺失段检测与聚合统计，适合在离线批处理或日常数据质量作业中使用。
#代码：
#假设原始传感器数据以 Parquet/CSV 存储，包含字段：device_id, sensor_type, ts (ISO 字符串或 epoch 毫秒), value
#输出为每个 device_id + sensor_type 在指定时间窗口内的缺失指标（最长连续缺失时长、缺失段数量、观测率等）
# 运行环境：PySpark（SparkSession 已安装并配置）
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("iot_data_gaps_detection") \
    .getOrCreate()

# 1. 读取数据（示例以 Parquet 为例）
# 数据 schema 假定： device_id STRING, sensor_type STRING, ts LONG (epoch ms), value DOUBLE
df = spark.read.parquet("/data/iot/raw/sensor_events/")

# 2. 预处理：转换时间，按固定采样粒度重采样（例如 1 分钟）
# 配置参数
sampling_ms = 60 * 1000  # 1 minute
start_ts = 1690000000000  # 可按需要设定或从数据中计算
end_ts = 1690086400000

# 2.1 过滤时间窗口并取必要字段
df = df.select("device_id", "sensor_type", "ts", "value") \
       .filter((F.col("ts") >= F.lit(start_ts)) & (F.col("ts") <= F.lit(end_ts)))

# 2.2 将时间对齐到采样窗口的起点（向下取整到 sampling_ms）
df = df.withColumn("aligned_ts", (F.col("ts") / F.lit(sampling_ms)).cast("long") * F.lit(sampling_ms))

# 3. 聚合到采样粒度：每个采样点是否有观测（存在 value）
agg = df.groupBy("device_id", "sensor_type", "aligned_ts") \
        .agg(F.count(F.lit(1)).alias("obs_cnt"),
             F.avg("value").alias("value_avg"))

# 4. 生成完整时间格栅（time grid）用于检测缺失（左连接）
# 首先构建所有 device+sensor 的组合
devices_sensors = agg.select("device_id", "sensor_type").distinct()

# 生成时间格栅 DataFrame（单列 aligned_ts），基于 start_ts..end_ts
ts_seq = spark.range(start_ts, end_ts + 1, sampling_ms).withColumnRenamed("id", "aligned_ts")

# 交叉 join 生成完整网格（注意：大规模设备时需改成分区化或按批处理以避免爆炸性膨胀）
full_grid = devices_sensors.crossJoin(ts_seq)

# 5. 左连接观测数据，标记是否缺失
grid_with_obs = full_grid.join(agg, on=["device_id", "sensor_type", "aligned_ts"], how="left") \
                         .withColumn("observed", F.when(F.col("obs_cnt").isNotNull() & (F.col("obs_cnt") > 0), F.lit(1)).otherwise(F.lit(0)))

# 6. 计算缺失段：用窗口函数查找连续块（run-length encoding 风格）
# 为每一行生成一个序号，按 device+sensor+aligned_ts 排序
w = Window.partitionBy("device_id", "sensor_type").orderBy("aligned_ts")
grid_with_obs = grid_with_obs.withColumn("row_num", F.row_number().over(w))

# 通过与累积观察量相减的方法分组连续 observed 值
# 当 observed==1 时 group_id = row_num - cumulative_sum(observed) ; observed==0 时类似
grid_with_obs = grid_with_obs.withColumn("cum_obs", F.sum("observed").over(w))
grid_with_obs = grid_with_obs.withColumn("group_id", (F.col("row_num") - F.col("cum_obs")))

# 7. 聚合每组计算连续段的长度和是否为缺失段
grp = grid_with_obs.groupBy("device_id", "sensor_type", "group_id", "observed") \
                   .agg(F.min("aligned_ts").alias("seg_start"),
                        F.max("aligned_ts").alias("seg_end"),
                        ( (F.max("aligned_ts") - F.min("aligned_ts")) / F.lit(sampling_ms) + 1 ).cast("long").alias("seg_len"))

# 只保留缺失段（observed == 0），统计每个设备传感器的缺失指标
gaps = grp.filter(F.col("observed") == 0).groupBy("device_id", "sensor_type") \
          .agg(F.max("seg_len").alias("max_gap_len"),
               F.sum("seg_len").alias("total_missing_points"),
               F.count("*").alias("num_gap_segments"))

# 8. 计算观测率等总体指标（以采样点总数为基准）
# 先计算每个 device+sensor 的总采样点数
total_points = ts_seq.count()  # 注意这是假定全体时间点数量
# 更稳健的做法：计算每个 device 的时间范围并求点数，这里示例简化
gaps = gaps.withColumn("total_expected_points", F.lit(total_points)) \
           .withColumn("observed_points", F.col("total_expected_points") - F.col("total_missing_points")) \
           .withColumn("observed_rate", F.col("observed_points") / F.col("total_expected_points"))

# 9. 保存结果到下游表（Parquet）
gaps.write.mode("overwrite").parquet("/data/iot/quality/gaps_by_device_sensor/")

spark.stop()