from pyspark.sql import SparkSession
from sparknlp.pretrained import PretrainedPipeline

def generate_summary(input_text):
    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName("TextSummarization") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:3.3.0") \
        .getOrCreate()

    # Load pre-trained pipeline for summarization
    pipeline = PretrainedPipeline("explain_document_dl_minimal", lang="en")

    # Process input text to generate summary
    result = pipeline.annotate(input_text)

    # Extract summary from the processed result
    summary = result["summary"]

    # Stop SparkSession
    spark.stop()

    return summary

# Example usage
input_text = "Your input text goes here. This is a sample text for summarization."
summary = generate_summary(input_text)
print(summary)
