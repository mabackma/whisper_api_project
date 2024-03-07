import json, os
import sparknlp
import sparknlp_jsl

from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp_jsl.annotator import *

import pandas as pd
import warnings

with open('spark_jsl.json') as f:
    license_keys = json.load(f)

# Defining license key-value pairs as local variables
locals().update(license_keys)
os.environ.update(license_keys)

pd.set_option('display.max_colwidth', 200)

warnings.filterwarnings('ignore')

params = {"spark.driver.memory":"16G",
          "spark.kryoserializer.buffer.max":"2000M",
          "spark.driver.maxResultSize":"2000M"}

spark = sparknlp_jsl.start(license_keys['SECRET'],params=params)

print("Spark NLP Version :", sparknlp.version())
print("Spark NLP_JSL Version :", sparknlp_jsl.version())

spark

paper_abstract = [
    "We have previously reported the feasibility of diagnostic and therapeutic peritoneoscopy including liver biopsy, gastrojejunostomy, and tubal ligation by an oral transgastric approach. We present results of per-oral transgastric splenectomy in a porcine model. The goal of this study was to determine the technical feasibility of per-oral transgastric splenectomy using a flexible endoscope. We performed acute experiments on 50-kg pigs. All animals were fed liquids for 3 days prior to procedure. The procedures were performed under general anesthesia with endotracheal intubation. The flexible endoscope was passed per orally into the stomach and puncture of the gastric wall was performed with a needle knife. The puncture was extended to create a 1.5-cm incision using a pull-type sphincterotome, and a double-channel endoscope was advanced into the peritoneal cavity. The peritoneal cavity was insufflated with air through the endoscope. The spleen was visualized. The splenic vessels were ligated with endoscopic loops and clips, and then mesentery was dissected using electrocautery. Endoscopic splenectomy was performed on six pigs. There were no complications during gastric incision and entrance into the peritoneal cavity. Visualization of the spleen and other intraperitoneal organs was very good. Ligation of the splenic vessels and mobilization of the spleen were achieved using commercially available devices and endoscopic accessories.",
    "To evaluate the degree to which histologic chorioamnionitis, a frequent finding in placentas submitted for histopathologic evaluation, correlates with clinical indicators of infection in the mother. A retrospective review was performed on 52 cases with a histologic diagnosis of acute chorioamnionitis from 2,051 deliveries at University Hospital, Newark, from January 2003 to July 2003. Third-trimester placentas without histologic chorioamnionitis (n = 52) served as controls. Cases and controls were selected sequentially. Maternal medical records were reviewed for indicators of maternal infection. Histologic chorioamnionitis was significantly associated with the usage of antibiotics (p = 0.0095) and a higher mean white blood cell count (p = 0.018). The presence of 1 or more clinical indicators was significantly associated with the presence of histologic chorioamnionitis (p = 0.019).",
    "In patients with Los Angeles (LA) grade C or D oesophagitis, a positive relationship has been established between the duration of intragastric acid suppression and healing.AIM: To determine whether there is an apparent optimal time of intragastric acid suppression for maximal healing of reflux oesophagitis. Post hoc analysis of data from a proof-of-concept, double-blind, randomized study of 134 adult patients treated with esomeprazole (10 or 40 mg od for 4 weeks) for LA grade C or D oesophagitis. A curve was fitted to pooled 24-h intragastric pH (day 5) and endoscopically assessed healing (4 weeks) data using piecewise quadratic logistic regression. Maximal reflux oesophagitis healing rates were achieved when intragastric pH>4 was achieved for approximately 50-70% (12-17 h) of the 24-h period. Acid suppression above this threshold did not yield further increases in healing rates."
                  ]

question = [
    "Transgastric endoscopic splenectomy: is it possible?",
    "Does histologic chorioamnionitis correspond to clinical chorioamnionitis?",
    "Is there an optimal time of acid suppression for maximal healing?"
          ]

data = spark.createDataFrame(
    [
        [paper_abstract[0],  question[0]],
        [paper_abstract[1],  question[1]],
        [paper_abstract[2],  question[2]],
    ]
).toDF("context","question")

data.show(truncate=False)
#data.show(truncate = 60)