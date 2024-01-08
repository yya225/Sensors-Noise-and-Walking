import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from scipy import signal
from pyspark.sql import SparkSession, functions, types
from pyspark.sql.window import Window
from pyspark.sql.functions import rank
import matplotlib.pyplot as plt

# run with spark-submit data_analysis.py data output

spark = SparkSession.builder.appName('first Spark app').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 8) # make sure we have Python 3.8+
assert spark.version >= '3.2' # make sure we have Spark 3.2+

# read file as following schema
pages_schema = types.StructType([
    types.StructField('ax', types.FloatType()),
    types.StructField('ay', types.FloatType()),
    types.StructField('az', types.FloatType())])

# extract the filename from path
def extract_file (filename):
    output = filename[filename.rfind('/'): -1]
    output = output[output.find('/')+1 : output.rfind('.')]
    return output

file_name = functions.udf ( lambda x: extract_file(x), returnType = types.StringType())

# split the total dataframe into x y z dataframe
# create the x y z dataframe in the format of row as index, column as filename
def splitData(dataframe):
    #for x-axis data
    ax_data = dataframe.pivot(index = "num_lines",columns = "filename", values = "ax")
    ax_data = ax_data.rename_axis("num_lines", axis=1).reset_index(drop=True)
    ax_data = ax_data.reset_index()

    #for y-axis data
    ay_data = dataframe.pivot(index = "num_lines",columns = "filename", values = "ay")
    ay_data = ay_data.rename_axis("num_lines", axis=1).reset_index(drop=True)
    ay_data = ay_data.reset_index()

    #for z-axis data
    az_data = dataframe.pivot(index = "num_lines",columns = "filename", values = "az")
    az_data = az_data.rename_axis("num_lines", axis=1).reset_index(drop=True)
    az_data = az_data.reset_index()

    return ax_data, ay_data, az_data

# filter the data on butterworth filter
def filterData(df, freq):
    for column in df:
        temp = df[column].values
        b, a = signal.butter(3, freq, btype='lowpass', analog=False)
        result = signal.filtfilt(b, a, temp)
        df[column] = pd.Series(result)
    return df

# plot all the x y z data
def plotData(ax_L,ay_L,az_L,ax_R,ay_R,az_R,num):

    fig, axes = plt.subplots(nrows=3, ncols=2)

    #plot left data
    ax_L.plot(ax=axes[0,0],figsize=(50,10))
    axes[0,0].set_title('filtered left x_axis data')
    ay_L.plot(ax=axes[1,0],figsize=(50,10))
    axes[1,0].set_title('filtered left y_axis data')
    az_L.plot(ax=axes[2,0],figsize=(50,10))
    axes[2,0].set_title('filtered left z_axis data')

    #plot right data
    ax_R.plot(ax=axes[0,1],figsize=(50,10))
    axes[0,1].set_title('filtered right x_axis data')
    ay_R.plot(ax=axes[1,1],figsize=(50,10))
    axes[1,1].set_title('filtered right y_axis data')
    az_R.plot(ax=axes[2,1],figsize=(50,10))
    axes[2,1].set_title('filtered right z_axis data')

    # use type variables to save plot
    if (num==0):
        plt.savefig('normal_analysis.png')
    if (num==1):
        plt.savefig('injury_analysis.png')
    if (num==2):
        plt.savefig('hand_analysis.png')

# slice the group for 1.5 second
def getSliceGroups(df):
    arr = np.arange(0,(len(df)//150)+1,1)
    repeat = np.repeat(arr,150)
    df["split"] = pd.Series(repeat)
    return df

# compare the variance for each 1.5 group data and get the p-value
def pValue(left,right):
    varLeft = left.groupby("split").var()
    varRight = right.groupby("split").var()
    arr = np.array([])
    for i in range(0,len(varLeft.columns)):
        stat,p=stats.ttest_ind(columnLeft,columnRight)
        arr= np.insert(arr,len(arr),p)
    return arr


def main(in_directory, out_directory):
    # read the file, except the path [only with file name]
    data_total = spark.read.csv(in_directory, schema=pages_schema,sep=",").withColumn('filename',functions.input_file_name())
    data_total = data_total.withColumn('filename', file_name(data_total.filename))

    # drop any NaN value
    data_total = data_total.dropna()

    # create id for the dataframe
    data_total = data_total.select("*").withColumn("id", functions.monotonically_increasing_id())
    data_total = data_total.select("*").withColumn("id", data_total.id.cast("int"))

    # Refrence: https://sparkbyexamples.com/spark/spark-sql-window-functions/
    # to start the num_lines variables at 0 for each of different filenames
    window = Window.partitionBy(data_total['filename']).orderBy(data_total['id'])
    data_total = data_total.select('*', rank().over(window).alias('num_lines'))

    # remove data for first 4 second and last 6 seconds
    data_total = data_total.filter(data_total["num_lines"] > 400)
    data_total = data_total.filter(data_total["num_lines"] < 6400)

    # change to pandas dataframe in order to use matplotlab
    data_panda = data_total.toPandas()
    data_panda = data_panda.dropna()

    #split the data
    ax_data, ay_data, az_data = splitData(data_panda)

    ax_filtered = filterData(ax_data, 0.12)
    ay_filtered = filterData(ay_data, 0.07)
    az_filtered = filterData(az_data, 0.17)

    # #seperate regular,injury and hand data
    ax_df_regular = ax_filtered.loc[:,ax_filtered.columns.str.contains("regular")]
    ay_df_regular = ay_filtered.loc[:,ay_filtered.columns.str.contains("regular")]
    az_df_regular = az_filtered.loc[:,az_filtered.columns.str.contains("regular")]


    ax_df_injury = ax_filtered.loc[:,ax_filtered.columns.str.contains("injury")]
    ay_df_injury = ay_filtered.loc[:,ay_filtered.columns.str.contains("injury")]
    az_df_injury = az_filtered.loc[:,az_filtered.columns.str.contains("injury")]

    ax_df_hand = ax_filtered.loc[:,ax_filtered.columns.str.contains("hand")]
    ay_df_hand = ay_filtered.loc[:,ay_filtered.columns.str.contains("hand")]
    az_df_hand = az_filtered.loc[:,az_filtered.columns.str.contains("hand")]

    # seperate left and right data
    ax_L_regular_filtered = ax_df_regular.loc[:,ax_df_regular.columns.str.contains("L")]
    ay_L_regular_filtered = ay_df_regular.loc[:,ay_df_regular.columns.str.contains("L")]
    az_L_regular_filtered = az_df_regular.loc[:,az_df_regular.columns.str.contains("L")]
    ax_R_regular_filtered = ax_df_regular.loc[:,ax_df_regular.columns.str.contains("R")]
    ay_R_regular_filtered = ay_df_regular.loc[:,ay_df_regular.columns.str.contains("R")]
    az_R_regular_filtered = az_df_regular.loc[:,az_df_regular.columns.str.contains("R")]

    ax_L_injury_filtered = ax_df_injury.loc[:,ax_df_injury.columns.str.contains("L")]
    ay_L_injury_filtered = ay_df_injury.loc[:,ay_df_injury.columns.str.contains("L")]
    az_L_injury_filtered = az_df_injury.loc[:,az_df_injury.columns.str.contains("L")]
    ax_R_injury_filtered = ax_df_injury.loc[:,ax_df_injury.columns.str.contains("R")]
    ay_R_injury_filtered = ay_df_injury.loc[:,ay_df_injury.columns.str.contains("R")]
    az_R_injury_filtered = az_df_injury.loc[:,az_df_injury.columns.str.contains("R")]

    ax_L_hand_filtered = ax_df_hand.loc[:,ax_df_hand.columns.str.contains("L")]
    ay_L_hand_filtered = ay_df_hand.loc[:,ay_df_hand.columns.str.contains("L")]
    az_L_hand_filtered = az_df_hand.loc[:,az_df_hand.columns.str.contains("L")]
    ax_R_hand_filtered = ax_df_hand.loc[:,ax_df_hand.columns.str.contains("R")]
    ay_R_hand_filtered = ay_df_hand.loc[:,ay_df_hand.columns.str.contains("R")]
    az_R_hand_filtered = az_df_hand.loc[:,az_df_hand.columns.str.contains("R")]

    #get the plot for all dataset
    plotData(ax_L_regular_filtered,ay_L_regular_filtered,az_L_regular_filtered,ax_R_regular_filtered,ay_R_regular_filtered,az_R_regular_filtered,0)
    plotData(ax_L_injury_filtered,ay_L_injury_filtered,az_L_injury_filtered,ax_R_injury_filtered,ay_R_injury_filtered,az_R_injury_filtered,1)
    plotData(ax_L_hand_filtered,ay_L_hand_filtered,az_L_hand_filtered,ax_R_hand_filtered,ay_R_hand_filtered,az_R_hand_filtered,2)

    # #get data for each 1.5 seconds
    ax_L_regular_filtered = getSliceGroups(ax_L_regular_filtered)
    ay_L_regular_filtered = getSliceGroups(ay_L_regular_filtered)
    az_L_regular_filtered = getSliceGroups(az_L_regular_filtered)
    ax_R_regular_filtered = getSliceGroups(ax_R_regular_filtered)
    ay_R_regular_filtered = getSliceGroups(ay_R_regular_filtered)
    az_R_regular_filtered = getSliceGroups(az_R_regular_filtered)

    ax_L_injury_filtered = getSliceGroups(ax_L_injury_filtered)
    ay_L_injury_filtered = getSliceGroups(ay_L_injury_filtered)
    az_L_injury_filtered = getSliceGroups(az_L_injury_filtered)
    ax_R_injury_filtered = getSliceGroups(ax_R_injury_filtered)
    ay_R_injury_filtered = getSliceGroups(ay_R_injury_filtered)
    az_R_injury_filtered = getSliceGroups(az_R_injury_filtered)

    ax_L_hand_filtered = getSliceGroups(ax_L_hand_filtered)
    ay_L_hand_filtered = getSliceGroups(ay_L_hand_filtered)
    az_L_hand_filtered = getSliceGroups(az_L_hand_filtered)
    ax_R_hand_filtered = getSliceGroups(ax_R_hand_filtered)
    ay_R_hand_filtered = getSliceGroups(ay_R_hand_filtered)
    az_R_hand_filtered = getSliceGroups(az_R_hand_filtered)

    # #get p_value for comparing the variance
    ax_p_regular = pValue(ax_L_regular_filtered, ax_R_regular_filtered)
    ay_p_regular = pValue(ay_L_regular_filtered, ay_R_regular_filtered)
    az_p_regular = pValue(az_L_regular_filtered, az_R_regular_filtered)

    ax_p_injury = pValue(ax_L_injury_filtered, ax_R_injury_filtered)
    ay_p_injury = pValue(ay_L_injury_filtered, ay_R_injury_filtered)
    az_p_injury = pValue(az_L_injury_filtered, az_R_injury_filtered)

    axL_p_hand = pValue(ax_L_hand_filtered, ax_L_regular_filtered)
    ayL_p_hand = pValue(ay_L_hand_filtered, ay_L_regular_filtered)
    azL_p_hand = pValue(az_L_hand_filtered, az_L_regular_filtered)

    axR_p_hand = pValue(ax_R_hand_filtered, ax_R_regular_filtered)
    ayR_p_hand = pValue(ay_R_hand_filtered, ay_R_regular_filtered)
    azR_p_hand = pValue(az_R_hand_filtered, az_R_regular_filtered)

    # output the p-value into csv for further explanation
    out_directory = pd.DataFrame(
        columns=['ax_regular', 'ay_regular','az_regular','ax_injury', 'ay_injury','az_injury', 'axL_p_hand', 'ayL_p_hand', 'azL_p_hand','axR_p_hand', 'ayR_p_hand', 'azR_p_hand']
    )
    out_directory["ax_regular"] = pd.Series(ax_p_regular)
    out_directory["ay_regular"] = pd.Series(ay_p_regular)
    out_directory["az_regular"] = pd.Series(az_p_regular)
    out_directory["ax_injury"] = pd.Series(ax_p_injury)
    out_directory["ay_injury"] = pd.Series(ay_p_injury)
    out_directory["az_injury"] = pd.Series(az_p_injury)
    out_directory["axL_hand"] = pd.Series(axL_p_hand)
    out_directory["ayL_hand"] = pd.Series(ayL_p_hand)
    out_directory["azL_hand"] = pd.Series(azL_p_hand)
    out_directory["axR_hand"] = pd.Series(axR_p_hand)
    out_directory["ayR_hand"] = pd.Series(ayR_p_hand)
    out_directory["azR_hand"] = pd.Series(azR_p_hand)

    out_directory.to_csv("p-values.csv",index=False)

if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
