import os
import logging
import redis


from openai import OpenAI
from gnews import GNews
from typing import Dict, Any
import instructor

from .news import get_company_news
from .themes_extractor import identify_themes


from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _process_single_company(row: Dict[str, Any]) -> tuple:
    """
    Helper function for parallel processing of a single S&P 500 company.

    This function:
      1) Extracts the company's symbol and name.
      2) Fetches relevant news articles for that company.
      3) Identifies the key investment themes for the company using AI techniques.

    Parameters
    ----------
    row : Dict[str, Any]
        A dictionary containing row data from the S&P 500 DataFrame,
        including the keys 'Symbol' (ticker) and 'Security' (company name).

    Returns
    -------
    tuple
        A tuple of the form (symbol, context, themes), where
         - symbol is the company's ticker symbol (str).
         - context is the concatenated news articles text (str).
         - themes is a list of identified themes (List[str]).
    """

    symbol = row["Symbol"]
    company_name = row["Security"]

    print(
        f"[Child PID={os.getpid()}] Processing company: {company_name} ({symbol}) ..."
    )

    # Get company news
    cache = redis.Redis(
        host=os.getenv("REDIS_HOST"),
        port=os.getenv("REDIS_PORT"),
        decode_responses=True,
        username="default",
        password=os.getenv("REDIS_PASSWORD"),
    )

    google_news = GNews(language="en", max_results=10)
    instructor_client = instructor.from_openai(OpenAI())

    context = get_company_news(google_news, cache, f"{company_name}({symbol})")

    # Identify themes
    themes = identify_themes(instructor_client, company_name, context)
    return symbol, context, themes
