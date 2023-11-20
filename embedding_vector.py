import openai
import os
import dotenv

import pandas as pd
import numpy as np

from tenacity import retry, wait_random_exponential, stop_after_attempt
#find way to store embedding in a embedding specific vector store
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

embedding_cache_path = "movie_embeddings.pkl"
#load the cache if it exists and savea copy to disk
try:
    embedding_cache = pd.read_pickle(embedding_cache_path)
except FileNotFoundError:
    embedding_cache = {}
with open(embedding_cache_path, "wb") as f:
    pd.dump(embedding_cache, f)
#define a function to retrieve embeddings form the cache if present, and otherwise request via the API
def embedding_from_string(string, model="text-embedding-ada-002", embedding_cache=embedding_cache):
    if (string, model) not in embedding_cache.keys():
        embedding_cache[(string, model)] = get_embedding(string, model)
        print(f"GOT EMBEDDING FOR {string:[:20]}")
        with open(embedding_cache_path, "wb") as f:
            pd.dump(embedding_cache, f)
        return embedding_cache[(string, model)]