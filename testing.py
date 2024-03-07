import sparknlp_jsl
import os
from dotenv import load_dotenv
from pyspark.sql import SparkSession

# Access the secret code
load_dotenv()
secret_code = os.getenv('SECRET_CODE')
version_public = os.getenv('VERSION_PUBLIC')
version = os.getenv('VERSION')
license_key = os.getenv('LICENSE_KEY')
print(f'secret_code: {secret_code}')
print(f'sparknlp_jsl version: {sparknlp_jsl.version()}')
print(f'version_public: {version_public}')
print(f'version: {version}')
print(f'license_key: {license_key}')
spark = sparknlp_jsl.start(license_key)


#spark = SparkSession.builder \
#    .appName("Spark NLP Enterprise") \
#    .master("local[*]") \
#    .config("spark.driver.memory","16g") \
#    .config("spark.driver.maxResultSize", "2G") \
#    .config("spark.jars.packages", f"com.johnsnowlabs.nlp:spark-nlp_2.11:{version_public}") \
#    .config("spark.jars", f"https://pypi.johnsnowlabs.com/{secret_code}/spark-nlp-jsl-{version}.jar") \
#    .getOrCreate()

print('Hello World!')