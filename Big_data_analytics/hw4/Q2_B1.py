"""
Retrieves the visited URLs during the last minute, and update the results every
10 seconds, such that the top k of the number of visits within the last minute with descending order.

 Usage: aggregation.py <directory>
   <directory> is the directory that Spark Streaming will use to find and read new text files (this directory must be in HDFS)

 To run this on Spark cluster, and to monitor directory `streamed_data` on HDFS, run this example
    $ python Q2_B1.py streamed_data <value of K>

 Then run the python script 'start_stream.py' that creates different text files in `streamed_data` that corresponds to user clicks.
 Note: Replace 'hn0-sparkl.koydbiauu5yuthevx5bhblrehb.ax.internal.cloudapp.net' with your primary namenode that you can get from
 calling this command: hdfs getconf -namenodes
 """

import sys
from pyspark import SparkContext
from pyspark.streaming import StreamingContext

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: Q2_B1.py <directory> <Value of K>")
        exit(-1)

    sc = SparkContext(appName="myAppAggregation")
    sc.setCheckpointDir("hdfs://hn0-sbdacl.rwcq00v1gfyetmn4ktdcz0g53b.ax.internal.cloudapp.net/checkpointing/")
    ssc = StreamingContext(sc, 5)
    lines = ssc.textFileStream(sys.argv[1])
    counts = lines.map(lambda line: line.split(" ")).map(lambda x: (x[2],1)).\
    reduceByKeyAndWindow(lambda a,b:a+b, lambda x,y: x-y, 60, 10).\
    transform(lambda rdd: rdd.sortBy(lambda x: x[1], ascending=False))
    
    counts.pprint(int(sys.argv[2]))

    ssc.start()
    ssc.awaitTermination()
