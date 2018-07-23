//Q2.A1
val k = 5
val paperauths = sc.textFile("lab_spark_fzhao/dblp/dblp_tsv/paperauths.tsv")
val paperAuthsTokens = paperauths.map(_.split("\t"))
val authCounts = sc.parallelize(paperAuthsTokens.map(tokens => (tokens(1), 1)).reduceByKey((a, b) => a + b).sortBy(_._2, false).take(k))
val auths = sc.textFile("lab_spark_fzhao/dblp/dblp_tsv/authors.tsv")
val authsTokens = auths.map(_.split("\t")).map(tokens => (tokens(0),tokens(1)))
authsTokens.join(authCounts).map(_._2).map(_._1).collect.foreach(println)
// res5: Wei Liu
// Wen Gao
// Yan Zhang
// H. Vincent Poor
// Wei Wang


//Q2.A2
import org.apache.spark.mllib.fpm.FPGrowth
import org.apache.spark.rdd.RDD
val paperauths = sc.textFile("lab_spark_fzhao/dblp/dblp_tsv/paperauths.tsv")
val paperAuthsTokens = paperauths.map(_.split("\t"))
val dataset: RDD[Array[String]] = paperAuthsTokens.map(tokens => (tokens(0), tokens(1))).groupByKey().mapValues(_.toSet.toArray).map(_._2)
dataset.first()

val fpg = new FPGrowth().setMinSupport(0.0001)
val model = fpg.run(dataset)

model.freqItemsets.map(itemset => itemset.items.mkString("[", ",", "]") + ", " + itemset.freq).saveAsTextFile("lab_spark_fzhao/result")
