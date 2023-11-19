import openai
import os
import dotenv

import pandas as pd
import numpy as np

dotenv.load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

res = client.embeddings.create(
    input="candy canes",
    model="text-embedding-ada-002",
)
print(res.data[0].embedding)
