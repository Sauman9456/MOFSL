import logging
from .config import themes as themes_list

from typing import List, Literal
from pydantic import BaseModel, Field


from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

themes_list = themes_list + ["None"]
themes_literal = Literal[tuple(themes_list)]

formatted_string = "\n".join(f"- {theme}" for theme in themes_list)


class MultiClassPrediction(BaseModel):
    themes: List[
        themes_literal  # type: ignore
    ] = Field(
        ...,
        description="Select only the themes that the company belongs to.",
    )


def identify_themes(instructor_client, company_name: str, context: str) -> List[str]:
    """
    Use OpenAI to identify themes from company news context.
    Optionally applies advanced techniques like embeddings, KMeans clustering, sentiment.

    Parameters
    ----------
    context : str
        Combined text from news articles.
    use_advanced : bool
        Whether to use advanced embeddings + clustering approach.

    Returns
    -------
    List[str]
        A List containing identified theme information.
    """
    if not context.strip():
        return {}

    # Simple GPT Prompt-based approach (base)
    system_thematic_prompt = f"""
**Context Information:**  
------------------------------------------  
{context}  
------------------------------------------  

**Background:**  
Thematic investing has become an appealing strategy for investors aiming to leverage major trends that are shaping the future.

**Available Themes:**  
{formatted_string}

**Task:**  
You are a financial analyst specializing in thematic investing. Your task is to classify the company into one or more of the **Available Themes** based on only the provided context. The context Information includes the latest news titles retrieved from a Google News search for the company named **{company_name}**.

**Important Notes:**  
1. The context Information provided is derived from Google News search results. Not all news items may be directly related to the company named **{company_name}**.  
2. Carefully analyze the news articles, verify the relevance of each item, and only classify the company based on valid and substantiated information.
3. As these themes are directly related to investing,**accuracy is paramount, and mistakes must be avoided at all costs**.
4. If, based on the context Information, the company does not belong to any of the available themes, classify the theme as **'None'**.
4. Take your time to ensure the classification is precise and aligns with the most relevant themes.
    """
    try:
        response = instructor_client.chat.completions.create(
            model="gpt-4o",
            response_model=MultiClassPrediction,
            messages=[
                {"role": "system", "content": system_thematic_prompt},
            ],
        )
        base_themes = response.themes
    except Exception as e:
        logger.error(f"Base theme extraction failed: {e}")
        base_themes = []

    # If advanced is not requested, return base themes
    # if not use_advanced:
    return base_themes

    # Advanced: Embeddings + Clustering + Sentiment Analysis
    # --------------------------------------------------------------------
    # 1) Get embedding for each sentence or the entire context
    # embedding = self._get_embedding(context)

    # 2) Cluster - here we do a trivial example by stacking a single embedding
    #    and running KMeans. In reality, you'd chunk your text or have multiple embeddings.
    #    We'll show a simplified example.
    #    For production, consider splitting context into sentences or paragraphs.
    # embeddings = np.array([embedding] * 5)
    # clusters = self._cluster_themes(embeddings, n_clusters=5)

    # 3) Sentiment Analysis (placeholder example)
    # sentiment = self._analyze_sentiment(context)

    # # Combine everything
    # advanced_themes = {
    #     "base_themes": base_themes,
    #     # "clusters": clusters.tolist(),
    #     "sentiment": sentiment
    # }
    # return advanced_themes


# def _get_embedding(self, text: str) -> np.ndarray:
#     """
#     Get text embedding using OpenAI's Ada embedding model.
#     Returns a NumPy array.
#     """
#     text = text.replace("\n", " ")
#     try:
#         response = self.openai_client.embeddings.create(
#             input=[text],
#             model="text-embedding-3-small"
#         )
#         return np.array(response['data'][0]['embedding'])
#     except Exception as e:
#         logger.error(f"Embedding request failed: {e}")
#         return np.zeros(1536)  # Typical embedding size fallback

# def _cluster_themes(self, embeddings: np.ndarray, n_clusters: int = 5) -> np.ndarray:
#     """
#     Cluster embeddings using K-Means.
#     Returns cluster labels.
#     """
#     try:
#         km = KMeans(n_clusters=n_clusters, random_state=42)
#         labels = km.fit_predict(embeddings)
#         return labels
#     except Exception as e:
#         logger.error(f"KMeans clustering failed: {e}")
#         return np.array([])

# def _analyze_sentiment(self, text: str) -> str:  ##Todo need to improve with instructor_client
#     """
#     Simple sentiment analysis using GPT.
#     Returns 'Positive', 'Negative', or 'Neutral' (placeholder).
#     """
#     system_sentiment_prompt = f"Determine the sentiment (Positive/Negative/Neutral) of the following text:\n{text}"
#     try:
#         response = self.openai_client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                     {"role": "system", "content": system_sentiment_prompt}],
#             temperature=0.0
#         )
#         sentiment = response.choices[0].message.strip()
#         return sentiment
#     except Exception as e:
#         logger.error(f"Sentiment analysis failed: {e}")
#         return "Unknown"
