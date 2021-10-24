import sparknlp

spark = sparknlp.start()

print("Spark NLP version: ", sparknlp.version())
print("Apache Spark version", spark.version)


from pyspark.ml import Pipeline

from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *

trainDataset = spark.read \
      .option("header", True) \
      .csv("data/aclimdb_test.n100.csv")

# actual content is inside description column
document = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

use = UniversalSentenceEncoder.pretrained() \
 .setInputCols(["document"])\
 .setOutputCol("sentence_embeddings")

# the classes/labels/categories are in category column
sentimentdl = SentimentDLApproach()\
  .setInputCols(["sentence_embeddings"])\
  .setOutputCol("class")\
  .setLabelColumn("label")\
  .setMaxEpochs(5)\
  .setEnableOutputLogs(True)

pipeline = Pipeline(
    stages = [
        document,
        use,
        sentimentdl
    ])

pipelineModel = pipeline.fit(trainDataset)
pipelineModel.stages[-1].write().overwrite().save('./tmp_sentimentdl_model')

# In a new pipeline you can load it for prediction
document = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

use = UniversalSentenceEncoder.pretrained() \
 .setInputCols(["document"])\
 .setOutputCol("sentence_embeddings")

sentimentdl = SentimentDLModel.load("./tmp_sentimentdl_model") \
  .setInputCols(["sentence_embeddings"])\
  .setOutputCol("class")

pipeline = Pipeline(
    stages = [
        document,
        use,
        sentimentdl
    ])


from pyspark.sql.types import StringType

dfTest = spark.createDataFrame([
    "This movie is a delight for those of all ages. I have seen it several times and each time I am enchanted by the characters and magic. The cast is outstanding, the special effects delightful, everything most believable.",
    "This film was to put it simply rubbish. The child actors couldn't act, as can be seen by Harry's supposed surprise on learning he's a wizard. I'm a wizard! is said with such indifference you'd think he's not surprised at all."
], StringType()).toDF("text")

prediction = pipeline.fit(dfTest).transform(dfTest)

prediction.select("class.result").show()

prediction.select("class.metadata").show(truncate=False)
