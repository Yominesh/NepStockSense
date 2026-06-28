"""Stage 1 — Scrape Nepali financial news from ShareSansar (2020-01-01 to present)."""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
from datetime import datetime

BASE_URL = "https://www.sharesansar.com/category/latest"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}
OUTPUT_FILE = "raw_news.csv"
START_DATE = datetime(2020, 1, 1)
BATCH_EVERY = 50   # save after every N pages
REQUEST_DELAY = 1  # seconds between requests


def parse_date(date_str):
    """Parse 'Sunday, June 21, 2026' -> datetime."""
    try:
        return datetime.strptime(date_str.strip(), '%A, %B %d, %Y')
    except Exception:
        return None


def scrape_page(url):
    """Return (list_of_articles, next_page_url_or_None)."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')

    articles = []
    links = soup.find_all('a', href=lambda h: h and '/newsdetail/' in h)

    for link in links:
        h4 = link.find('h4')
        if not h4:
            continue
        headline = h4.get_text(strip=True)
        href = link['href']
        if not href.startswith('http'):
            href = 'https://www.sharesansar.com' + href

        # Date is in the <p> immediately after the <a>
        p = link.find_next_sibling('p')
        date_str = p.get_text(strip=True) if p else ''
        date = parse_date(date_str)

        if headline and date:
            articles.append({
                'date': date.strftime('%Y-%m-%d'),
                'headline': headline,
                'source': 'sharesansar',
                'url': href,
            })

    # Next page cursor link
    next_url = None
    for a in soup.find_all('a', href=True):
        href = a['href']
        if 'cursor=' in href and 'Next' in a.get_text(strip=True):
            if href.startswith('?'):
                next_url = BASE_URL + href
            elif href.startswith('http'):
                next_url = href
            else:
                next_url = 'https://www.sharesansar.com' + href
            break

    return articles, next_url


def main():
    all_articles = []
    url = BASE_URL
    page_num = 0
    pages_since_save = 0
    stop = False

    # Load existing if resuming
    if os.path.exists(OUTPUT_FILE):
        existing_df = pd.read_csv(OUTPUT_FILE)
        print(f"Resuming — {len(existing_df)} articles already in {OUTPUT_FILE}")
        seen_urls = set(existing_df['url'].tolist())
    else:
        existing_df = None
        seen_urls = set()

    while url and not stop:
        page_num += 1
        print(f"  Page {page_num}: {url[:80]}...")

        try:
            articles, next_url = scrape_page(url)
        except Exception as exc:
            print(f"    Error: {exc} — sleeping 5s and retrying once")
            time.sleep(5)
            try:
                articles, next_url = scrape_page(url)
            except Exception as exc2:
                print(f"    Retry failed: {exc2} — stopping")
                break

        added = 0
        for art in articles:
            if art['url'] in seen_urls:
                continue
            seen_urls.add(art['url'])
            art_date = datetime.strptime(art['date'], '%Y-%m-%d')
            if art_date < START_DATE:
                stop = True
                print(f"    Reached articles before 2020 — stopping.")
                break
            all_articles.append(art)
            added += 1

        print(f"    Added {added} articles (total so far: {len(all_articles)})")

        pages_since_save += 1
        if pages_since_save >= BATCH_EVERY:
            _save(all_articles, existing_df)
            pages_since_save = 0

        if stop or not next_url:
            break

        url = next_url
        time.sleep(REQUEST_DELAY)

    # Final save
    df = _save(all_articles, existing_df)
    print(f"\nDone. {len(df)} total articles in {OUTPUT_FILE}")
    print(f"Date range: {df['date'].min()}  to  {df['date'].max()}")
    print(df[['date', 'headline', 'source']].head(10).to_string())
    return df


def _save(new_articles, existing_df):
    new_df = pd.DataFrame(new_articles) if new_articles else pd.DataFrame(
        columns=['date', 'headline', 'source', 'url'])
    if existing_df is not None:
        df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        df = new_df
    df.drop_duplicates(subset=['url'], inplace=True)
    df.to_csv(OUTPUT_FILE, index=False)
    return df


if __name__ == '__main__':
    main()
