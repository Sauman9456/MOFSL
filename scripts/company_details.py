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
    Helper function for parallel processing.
    Takes (row, use_advanced) as input, returns (symbol, context, themes).
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
