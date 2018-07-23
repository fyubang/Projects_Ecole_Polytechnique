"""
Retrieves the visited URLs during the last minute, and update the results every
10 seconds, such that the top k of the number of visits within the last minute with descending order.

 Usage: aggregation.py <directory>
   <directory> is the directory that Spark Streaming will use to find and read new text files (this directory must be in HDFS)

 To run this on Spark cluster, and to monitor directory `streamed_data` on HDFS, run this example
    $ python Q2_B2.py streamed_data

 Then run the python script 'start_stream.py' that creates different text files in `streamed_data` that corresponds to user clicks.
 Note: Replace 'hn0-sparkl.koydbiauu5yuthevx5bhblrehb.ax.internal.cloudapp.net' with your primary namenode that you can get from
 calling this command: hdfs getconf -namenodes
 """
import sys
from pyspark import SparkContext
from pyspark.streaming import StreamingContext

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: Q2_B2.py <directory>")
        exit(-1)

    sc = SparkContext(appName="myAppSessionization")
    sc.setCheckpointDir("hdfs://hn0-sbdacl.rwcq00v1gfyetmn4ktdcz0g53b.ax.internal.cloudapp.net/checkpointing/")
    ssc = StreamingContext(sc, 5)
    lines = ssc.textFileStream(sys.argv[1])
    counts = lines.map(lambda line: line.split(" ")).map(lambda x: (x[1], int(x[0])))

    def updateFunction(newVs, state):
        '''
        newVs<list>: new values for each key in this batch
        state<tuple>: (last timestamp, 
                        # click which is ready to output, 
                        # click to pass to next iteration for continuing to calculate, 
                        # session, 
                        output or not)

        '''
        if state is None:
            return (max(newVs), 0, len(newVs), 0, 0)
        else:
            if (len(newVs) == 0):
                return state[:4] + (0,) # avoid printing twice
            if min(newVs) - state[0] <= 30:
                return (max(newVs), 0, (len(newVs) + state[2]), state[3], 0)
            else:
                return (max(newVs), state[2], len(newVs), state[3] + 1, 1)
            
    countuup = counts.updateStateByKey(updateFunction).\
                filter(lambda x: x[1][4] == 1).\
                map(lambda x: (x[0], x[1][1], x[1][3]))
    countuup.pprint()
    ssc.start()
    ssc.awaitTermination()

