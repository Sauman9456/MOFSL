import logging
import warnings
from .config import themes

warnings.filterwarnings("ignore")


# os.getenv("NEWS_API_KEY")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_company_news(google_news, cache, company_name: str) -> str:
    """
    Fetch recent news articles for a company using NewsAPI.
    Returns combined string of titles for analysis.
    Caches the result if Redis is available to avoid repeated calls.
    """

    if cache:
        cached_data = cache.get(f"news:{company_name}")
        if cached_data:
            logger.info(f"Using cached news for {company_name}")
            return cached_data

    try:
        context = ""

        counter = 0

        for theme in themes:
            try:
                articles = google_news.get_news(f"{company_name} {theme}")

                for article in articles:
                    counter = counter + 1
                    context = context + f"{counter}. {article['title']}\n\n"

            except Exception as e:
                logger.warning(
                    f"Error fetching news for {company_name} with {theme}: {e}"
                )
                continue

        # Store in cache
        if cache:
            cache.set(f"news:{company_name}", context)

        return context
    except Exception as e:
        logger.warning(f"Error fetching news for {company_name}: {e}")
        return ""
