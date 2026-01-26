
models = ['llama3.1:8b', 'qwen3:8b', 'gemma3:4b', 'deepseek-r1:8b']

import kagglehub
from kagglehub import KaggleDatasetAdapter
from pydantic import BaseModel
from typing import Literal, Optional
from tqdm import tqdm
import ollama

def download_and_clean(n=5000):
    # Set the path to the file you'd like to load
    file_path = "twitter_sentiment_data.csv"

    # Load the latest version
    df = kagglehub.load_dataset(
      KaggleDatasetAdapter.PANDAS,
      "edqian/twitter-climate-change-sentiment-dataset",
      file_path,
    )

    df = df.sample(n=n, random_state = 42)


    sentiment_map = {
        2: "news",
        1: "pro",
        0: "neutral",
        -1: "anti"
    }

    df["sentiment_text"] = df["sentiment"].map(sentiment_map)

    # remove urls
    df['clean_message'] = df['message'].str.replace(r'http\S+|www\S+', '', regex=True)

    # remove handle after 'RT @'
    df['clean_message'] = df['clean_message'].str.replace(r'^RT\s+@\w+:\s*', '', regex=True)

    # drop ascii characters
    df['clean_message'] = df['clean_message'].str.encode('ascii', 'ignore').str.decode('ascii')

    return df


def transform_data(df, model, response_format, prompt, think = False):
    cols = response_format.__fields__.keys()
    for col in cols:
        df[col] = None
    if think:
        df['reasoning'] = None

    for idx, row in tqdm(df.iterrows(), total = len(df)):
        response = ollama.chat(
            model=model,
            messages=[{'role': 'system', 'content': prompt}, {'role': 'user',
            'content': row['clean_message']}],
            format=response_format.model_json_schema(),
            think = think
        )
        parsed_response = response_format.model_validate_json(response.message.content)
        for col in cols:
            df.at[idx, col] = getattr(parsed_response, col)
        if think:
            df.at[idx, 'reasoning'] = response.message.thinking

    return df

prompt = '''

You are a helpful research assistant, interested in the framing of narratives in tweets about climate change. You have been tasked with identifying the heroes, villains and victims in a selection of tweets.
Task:
Read each tweet and decide if there is a hero, a villain or a victim, as per the following criteria:

Hero: an entity contributing to/responsible for issue resolution
Villain: an entity contributing to/responsible for issue cause
Victim: an entity suffering the consequences of an issue

Chain of thought:
1. Identify the central issue: Determine what climate-related problem or event the tweet is discussing.
2. Look for conflict or tension: Check if the tweet highlights a problem, blame, praise, or action.
3. Detect heroes: Identify entities praised for mitigating or solving the issue.
4. Detect victims: Identify entities suffering negative consequences of the issue.
5. Detect villains: Identify entities blamed for causing or worsening the issue.
6. Identify Roles: For any identified actors, assign them one of the available types to the best of your ability. Only assing roles if you have identified an actor. Any identified actor will have an allocated type

Actor Types:
    "ENVIRONMENT": The natural world including ecosystems, wildlife, and natural resources.
    "CLIMATE_CHANGE": Long-term changes in temperature, precipitation, and weather patterns caused by human activities.
    "ENVIRONMENTAL_ACTIVISTS": Individuals or groups advocating for environmental protection and sustainability.
    "GENERAL_PUBLIC": The broad populations, communities, or individuals affected by or involved in environmental issues.
    "GOVERNMENTS_AND_POLITICIANS": Authorities and elected officials responsible for creating and enforcing laws and policies.
    "GREEN_TECHNOLOGY": Innovations and technologies aimed at reducing environmental impact and promoting sustainability.
    "INDUSTRY": Businesses and sectors involved in production, manufacturing, and economic activities impacting the environment.
    "EMISSIONS": Release of pollutants or greenhouse gases into the atmosphere from various sources.
    "LEGISLATION_AND_POLICY": Laws, regulations, and guidelines designed to manage environmental and climate-related issues.
    "MEDIA": Channels and platforms that disseminate information and shape public opinion on environmental topics.
    "SCIENCE_AND_EXPERTS": Researchers and professionals providing knowledge, data, and analysis on environmental and climate matters.

Examples:
1. Theresa May's new chief of staff, Gavin Barwell, is known for his knowledgable concern about climate change
    hero: Gavin Barwell
    hero_type: "GOVERNMENTS_AND_POLITICIANS",

1. The reality of climate change impacts everyone but the truth is that poor communities suffer most...perpetuating the injustice
    victim: poor communities
    hero_type: "GENERAL_PUBLIC"
    villain: Climate Change
    villain_type: "CLIMATE_CHANGE"

1. Anti-Trump actor fights global warming, but wont give up 14 homes and private jet
    hero: Anti-Trump actor
    hero_type: "ENVIROMENTAL_ACTIVISTS"
    villain: Anti-Trump actor
    villain_type: "ENVIROMENTAL_ACTIVISTS"

Extract from the text the names of entities (people, groups, organisations) that are explicitly framed as either heroes, victims or villains. Do not make your own interpretations. If there is no hero, villain, or victim, respond with 'none'.
'''



ActorType = Literal[
    "ENVIRONMENT",
    "CLIMATE_CHANGE",
    "ENVIROMENTAL_ACTIVISTS",
    "GENERAL_PUBLIC",
    "GOVERNMENTS_AND_POLITICIANS",
    "GREEN_TECHNOLOGY",
    "INDUSTRY",
    "EMISSIONS",
    "LEGISLATION_AND_POLICY",
    "MEDIA",
    "SCIENCE_AND_EXPERTS",
]
class ClassificationResponseFormat(BaseModel):
    hero: Optional[str] = None
    hero_type: Optional[ActorType] = None
    victim: Optional[str] = None
    victim_type: Optional[ActorType] = None
    villain: Optional[str] = None
    villain_type: Optional[ActorType] = None

def run_model(model, df):
    response_df = transform_data(df.copy(), model, ClassificationResponseFormat(), prompt=prompt)
    response_df.to_pickle(f'{model}.pkl')

def main():
    df = download_and_clean()
    for model in models:
        run_model(model, df.copy())

