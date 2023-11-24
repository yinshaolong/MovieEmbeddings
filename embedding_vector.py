import openai
import os
import dotenv

import pandas as pd
import numpy as np

from tenacity import retry, wait_random_exponential, stop_after_attempt
import pickle
#find way to store embedding in a embedding specific vector store
dotenv.load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   response = client.embeddings.create(input = [text], model=model)
   print(response)
   return response.data[0].embedding


# dataset_path = "./movie_plots.csv"
# df = pd.read_csv(dataset_path)
# #gets the dataframe where axes (rowsand columns) are aligned or joined and data of both the dataframes overlaps
#     #only gets the data that matches "American" and sorts the value by Release Year in descending order
# reduce = df["Origin/Ethnicity"] == "American"
# print(reduce,"\n","="*50)

# movies = df[reduce].sort_values("Release Year", ascending=False).head(5000)
# print(movies)

embedding_cache_path = "movie_embeddings.pkl"
#define a function to retrieve embeddings form the cache if present, and otherwise request via the API
try:
    embedding_cache = pd.read_pickle(embedding_cache_path)
except:
    embedding_cache = {}
with open(embedding_cache_path, "wb") as embedding_cache_file:
    pickle.dump(embedding_cache, embedding_cache_file)

# define a function to retrieve embeddings from the cache if present, and otherwise request via the API

def embedding_from_string(
    string,
    model="text-embedding-ada-002",
    embedding_cache=embedding_cache
):
    """Return embedding of given string, using a cache to avoid recomputing."""
    if (string, model) not in embedding_cache.keys():
        embedding_cache[(string, model)] = get_embedding(string, model)
        print(f"GOT EMBEDDING FROM OPENAI FOR {string[:20]}")
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, model)]

    
embedding_from_string("American")
#load the cache if it exists and savea copy to disk
with open(embedding_cache_path, "wb") as embedding_cache_file:
    pickle.dump(embedding_cache, embedding_cache_file)