import openai
import os
import dotenv

import pandas as pd
import numpy as np

dotenv.load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# res = client.embeddings.create(
#     input="candy canes",
#     model="text-embedding-ada-002",
# )
# print(res.data[0].embedding)

dataset_path = "./movie_plots.csv"
df = pd.read_csv(dataset_path)
#gets the dataframe where axes (rowsand columns) are aligned or joined and data of both the dataframes overlaps
    #only gets the data that matches "American" and sorts the value by Release Year in descending order
reduce = df["Origin/Ethnicity"] == "American"
print(reduce,"\n","="*50)

movies = df[reduce].sort_values("Release Year", ascending=False).head(5000)
print(movies)
