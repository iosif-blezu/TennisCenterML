# utils/tennisexplorer_news.py
import re, requests, bs4 as bs
from urllib.parse import urljoin

BASE = "https://www.tennisexplorer.com/tennis-news/"

def _parse_page(html: str):
    """
    Returns list[ dict(bucket,title,url) ]  for ONE page.
    """
    soup   = bs.BeautifulSoup(html, "lxml")
    raw    = soup.select_one("div#center")   # banner-free middle column
    if raw is None:
        return []

    text = raw.decode(formatter="html")
    # keep <b> bucket headings, strip everything else to plain text
    text = re.sub(r"<(?!/?b).*?>", "", text, flags=re.S)      # drop non-<b> tags
    # replace HTML entities
    text = bs.BeautifulSoup(text, "lxml").get_text("\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    current_bucket = None
    out = []
    for ln in lines:
        # bucket heading?
        if ln.startswith("In the last") or ln in ("Today", "Yesterday"):
            current_bucket = ln
            continue
        # headline lines look like:  "19:21 French Open … (BBC)"
        m = re.match(r"^\d\d:\d\d\s+(.*?)\s+\((.+?)\)$", ln)
        if m:
            title = m.group(1).strip()
            # need link – fish it from original soup:
            a_tag = raw.find("a", string=re.compile(re.escape(title)))
            if a_tag and a_tag.get("href"):
                url = a_tag["href"]
                if url.startswith("/"):
                    url = urljoin(BASE, url.lstrip("/"))
                out.append({"bucket": current_bucket, "title": title, "url": url})
    return out

def scrape_latest_news(pages: int = 1):
    """
    Scrape *pages* paginated pages (1–5) and return a flat list
    [ {bucket,title,url}, … ] newest → oldest.
    """
    pages = max(1, min(pages, 5))
    articles = []
    for p in range(1, pages + 1):
        url  = BASE if p == 1 else f"{BASE}?page={p}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        articles.extend(_parse_page(resp.text))
    return articles
