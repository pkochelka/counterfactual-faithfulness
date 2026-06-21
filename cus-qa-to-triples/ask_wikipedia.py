import sys
import argparse
import urllib.request
import urllib.error
import urllib.parse
import json
import textwrap
import time

# ---------------------------------------------------------------------------
# Compliance with Wikimedia API best practices:
# https://www.mediawiki.org/wiki/API:Etiquette
#
# 1. Meaningful User-Agent with contact info — moves you from the
#    "Unidentified" tier (10 req/min) to "User-Agent only" (200 req/min).
# 2. Respect the Retry-After header on 429 responses.
# 3. Max 3 concurrent requests (we use 1, always satisfied).
# ---------------------------------------------------------------------------

USER_AGENT = "wiki_tagging/1.0 (https://github.com/pkochelka/counterfactual-faithfulness; dvorak@ufal.mff.cuni.cz)"
MAX_RETRIES = 3
BATCH_SIZE = 50


def wiki_api_url(lang: str) -> str:
    return f"https://{lang}.wikipedia.org/w/api.php"


def fetch(url: str, delay: float) -> dict:
    """GET a URL and return parsed JSON, with Retry-After and exponential back-off."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, MAX_RETRIES + 1):
        time.sleep(delay)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                # Respect Retry-After if present, otherwise exponential back-off
                wait = int(e.headers.get("Retry-After", 2 ** attempt))
                print(f"Rate limited. Waiting {wait}s (attempt {attempt}/{MAX_RETRIES})...",
                      file=sys.stderr)
                time.sleep(wait)
                if attempt == MAX_RETRIES:
                    raise
            else:
                raise
        except urllib.error.URLError:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {url}")


def search_wikipedia(keyword: str, lang: str, delay: float) -> str | None:
    """Return the best-matching article title for a single keyword."""
    params = urllib.parse.urlencode({
        "action": "query",
        "list": "search",
        "srsearch": keyword,
        "srlimit": 1,
        "format": "json",
    })
    data = fetch(f"{wiki_api_url(lang)}?{params}", delay)
    results = data.get("query", {}).get("search", [])
    return results[0]["title"] if results else None


def fetch_extracts_batch(titles: list[str], lang: str, delay: float) -> dict[str, str | None]:
    """
    Fetch the first paragraph for up to BATCH_SIZE titles in a single API call.
    Returns a dict mapping each title to its first paragraph (or None if missing).
    """
    params = urllib.parse.urlencode({
        "action": "query",
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "titles": "|".join(titles),
        "format": "json",
    })
    data = fetch(f"{wiki_api_url(lang)}?{params}", delay)
    pages = data.get("query", {}).get("pages", {})

    results = {}
    for page in pages.values():
        title = page.get("title")
        extract = page.get("extract", "").strip()
        if "missing" in page or not extract:
            results[title] = None
            continue
        for paragraph in extract.split("\n"):
            paragraph = paragraph.strip()
            if paragraph:
                results[title] = paragraph
                break
        else:
            results[title] = None
    return results


def fetch_batch(keywords: list[str], lang: str = "en", delay: float = 0.5) -> dict[str, str | None]:
    """
    Fetch the first paragraph for a batch of keywords.

    Steps:
      1. Search API to resolve each keyword to an article title (one request each).
      2. Extracts API to fetch all resolved titles in one batched request.

    Args:
        keywords: List of search keywords (max BATCH_SIZE = 50).
        lang:     Wikipedia language code, e.g. "en", "cs", "de".
        delay:    Seconds to wait between requests (default: 0.5s — well within
                  the 200 req/min limit for identified User-Agent clients).

    Returns:
        {"keyword": "First paragraph...", ...}
        Value is None if no article was found.
    """
    if len(keywords) > BATCH_SIZE:
        raise ValueError(f"Batch size exceeds maximum of {BATCH_SIZE}. Split your input.")

    # Step 1: resolve each keyword to a title
    keyword_to_title: dict[str, str | None] = {}
    for keyword in keywords:
        keyword_to_title[keyword] = search_wikipedia(keyword, lang, delay)

    # Step 2: fetch all resolved titles in one batched extracts call
    titles = [t for t in keyword_to_title.values() if t is not None]
    title_to_paragraph: dict[str, str | None] = {}
    if titles:
        title_to_paragraph = fetch_extracts_batch(titles, lang, delay)

    # Step 3: map results back to original keywords
    return {
        keyword: title_to_paragraph.get(title)
        for keyword, title in keyword_to_title.items()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fetch the first paragraph of the most relevant Wikipedia article."
    )
    parser.add_argument("keyword", nargs="+", help="Search keyword(s)")
    parser.add_argument(
        "--lang", "-l",
        default="en",
        metavar="LANG",
        help="Wikipedia language code (default: en). Examples: de, fr, es, ja, cs",
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=0.5,
        metavar="SECONDS",
        help="Delay between requests in seconds (default: 0.5)",
    )
    args = parser.parse_args()

    keyword = " ".join(args.keyword)
    lang = args.lang.lower().strip()

    print(f'Searching {lang}.wikipedia.org for: "{keyword}"\n')

    try:
        results = fetch_batch([keyword], lang=lang, delay=args.delay)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    paragraph = results.get(keyword)
    if not paragraph:
        print("No article or extract found.")
        sys.exit(1)

    print(textwrap.fill(paragraph, width=80))


if __name__ == "__main__":
    main()