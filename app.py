from __future__ import annotations

from datetime import datetime
from html import escape
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup


st.set_page_config(page_title="평범한 날에, 신승호", page_icon="🗓️", layout="centered")

DATA_PATH = Path(__file__).parent / "data" / "named.xlsx"
ALL_DATA_PATH = Path(__file__).parent / "data" / "all.xlsx"
REQUIRED_COLUMNS = ["날짜", "시간", "매체명", "제목", "url"]


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame | None:
    """Accept both header-row sheets and headerless article sheets."""
    if frame.empty:
        return None

    frame = frame.iloc[:, :5].copy()
    if frame.shape[1] < 5:
        return None
    frame.columns = REQUIRED_COLUMNS
    frame = frame.dropna(subset=["날짜", "제목", "url"])
    frame["날짜"] = pd.to_datetime(frame["날짜"], errors="coerce")
    frame["url"] = frame["url"].astype(str).str.strip()
    frame = frame[frame["날짜"].notna() & frame["url"].str.startswith(("http://", "https://"))]
    return frame


@st.cache_data(show_spinner=False)
def load_articles(file_bytes: bytes, preferred_sheet: str | None = None) -> pd.DataFrame:
    """Read a preferred article sheet, or combine all usable sheets."""
    workbook = pd.ExcelFile(BytesIO(file_bytes))
    frames: list[pd.DataFrame] = []

    sheet_names = (
        [preferred_sheet]
        if preferred_sheet and preferred_sheet in workbook.sheet_names
        else workbook.sheet_names
    )
    for sheet in sheet_names:
        raw = pd.read_excel(workbook, sheet_name=sheet, header=None)
        if raw.empty or raw.shape[1] < 5:
            continue
        first_row = raw.iloc[0, :5].astype(str).str.replace(" ", "", regex=False).tolist()
        has_header = first_row[0] in {"날짜", "date"} and first_row[3] in {"제목", "title"}
        cleaned = _normalise_columns(raw.iloc[1:] if has_header else raw)
        if cleaned is not None:
            frames.append(cleaned)

    if not frames:
        raise ValueError("기사 데이터가 담긴 시트를 찾지 못했습니다.")

    articles = pd.concat(frames, ignore_index=True)
    articles = articles.drop_duplicates(subset=["날짜", "시간", "매체명", "제목", "url"])
    articles["시간정렬"] = pd.to_datetime(articles["시간"].astype(str), errors="coerce")
    articles = articles.sort_values(
        ["날짜", "시간정렬", "제목"],
        ascending=[False, False, False],
        na_position="last",
    )
    articles["연도"] = articles["날짜"].dt.year
    return articles


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def first_article_image(article_url: str) -> str | None:
    """Find the publisher-designated article photo before inspecting body images."""
    try:
        response = requests.get(
            article_url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ShinSeunghoArchive/1.0)"},
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException:
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Publishers normally set this to the actual lead photo used when an article
    # is shared on KakaoTalk, Naver and social networks. It is much more reliable
    # than the first <img>, which can be a logo, an advert, or a related-story image.
    for selector, attribute in (
        ('meta[property="og:image"]', "content"),
        ('meta[name="twitter:image"]', "content"),
        ('meta[name="twitter:image:src"]', "content"),
        ('link[rel="image_src"]', "href"),
    ):
        tag = soup.select_one(selector)
        if tag and tag.get(attribute):
            return urljoin(article_url, tag[attribute])

    # Some news sites provide the lead image only as schema.org Article data.
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            payload = json.loads(script.string or script.get_text())
        except (json.JSONDecodeError, TypeError):
            continue
        records = payload if isinstance(payload, list) else [payload]
        for record in records:
            if not isinstance(record, dict):
                continue
            if "@graph" in record and isinstance(record["@graph"], list):
                records.extend(record["@graph"])
                continue
            image = record.get("image")
            if isinstance(image, list) and image:
                image = image[0]
            if isinstance(image, dict):
                image = image.get("url") or image.get("contentUrl")
            if isinstance(image, str) and image:
                return urljoin(article_url, image)

    candidates = []
    # Last-resort fallback for publishers that do not expose a representative image.
    for selector in ("article img", ".article-body img", ".article_view img", ".view_content img", ".news_view img", ".content img"):
        candidates.extend(soup.select(selector))
    candidates.extend(soup.find_all("img"))

    for image in candidates:
        src = image.get("data-src") or image.get("data-original") or image.get("data-lazy-src") or image.get("src")
        if not src or src.startswith("data:"):
            continue
        text = " ".join([str(src), str(image.get("class", "")), str(image.get("alt", ""))]).lower()
        if any(word in text for word in ("logo", "icon", "banner", "ad_", "advert")):
            continue
        return urljoin(article_url, src)

    return None


def display_cards(articles: pd.DataFrame) -> None:
    cards = []
    for _, article in articles.iterrows():
        image_url = first_article_image(article["url"])
        safe_url = escape(str(article["url"]), quote=True)
        safe_title = escape(str(article["제목"]), quote=True)
        if image_url:
            image = (
                f'<a class="article-image" href="{safe_url}" target="_blank">'
                f'<img src="{escape(image_url, quote=True)}" alt="{safe_title}" loading="lazy"></a>'
            )
        else:
            image = f'<a class="no-image" href="{safe_url}" target="_blank">사진을 불러올 수 없습니다</a>'
        cards.append(
            f'<article class="article-card">{image}'
            f'<a class="article-title" href="{safe_url}" target="_blank">{safe_title}</a></article>'
        )
    st.markdown(f'<section class="article-grid">{"".join(cards)}</section>', unsafe_allow_html=True)


def display_title_links(articles: pd.DataFrame) -> None:
    links = []
    for _, article in articles.iterrows():
        safe_url = escape(str(article["url"]), quote=True)
        safe_title = escape(str(article["제목"]), quote=True)
        links.append(
            f'<li><a class="more-title" href="{safe_url}" target="_blank">{safe_title}</a></li>'
        )
    st.markdown(f'<ul class="more-list">{"".join(links)}</ul>', unsafe_allow_html=True)


st.markdown(
    """<style>
    .block-container {max-width: 760px; padding-top: 2.2rem; padding-bottom: 4rem;}
    h1 {letter-spacing: -0.06em; margin-bottom: 0.2rem;}
    .site-subtitle {font-size: 1rem; color: #777; margin: -.35rem 0 1.4rem; letter-spacing: .01em;}
    .article-grid {display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .7rem; align-items: start;}
    .article-card {min-width: 0;}
    .article-image img, .no-image {width: 100%; aspect-ratio: 1 / 1; object-fit: cover; display: block; border-radius: 5px; background: #efefeb;}
    .no-image {color: #777 !important; padding: 45% 1rem 0; text-align: center; font-size: .8rem;}
    .article-title {display: block; color: #171717 !important; font-size: .93rem; line-height: 1.45; text-decoration: none !important; margin: .45rem 0 1.6rem;}
    .article-title:hover, .article-title:focus, .article-title:visited {text-decoration: none !important;}
    .more-list {list-style: none; padding: 0; margin: .35rem 0 1.5rem;}
    .more-list li {padding: .7rem 0; border-bottom: 1px solid #ececea;}
    .more-title {color: #171717 !important; font-size: .93rem; line-height: 1.5; text-decoration: none !important;}
    .more-title:hover, .more-title:focus, .more-title:visited {text-decoration: none !important;}
    </style>""",
    unsafe_allow_html=True,
)
st.title("평범한 날에, 신승호")
st.markdown('<p class="site-subtitle">Ordinary days with SHIN SEUNGHO</p>', unsafe_allow_html=True)
st.caption("날짜를 고르면 같은 월·일의 모든 연도 기사가 표시됩니다. 사진 또는 제목을 누르면 원문으로 이동합니다.")

if not DATA_PATH.exists():
    st.error("`data/named.xlsx` 파일을 찾을 수 없습니다. 앱 폴더 안의 `data` 폴더에 파일을 넣어 주세요.")
    st.stop()

try:
    articles = load_articles(DATA_PATH.read_bytes(), preferred_sheet="NAMED")
except Exception as exc:
    st.error(f"엑셀 파일을 읽지 못했습니다: {exc}")
    st.stop()

available_dates = set(articles["날짜"].dt.date)
today_kst = datetime.now(ZoneInfo("Asia/Seoul")).date()
calendar_min = min(min(available_dates), today_kst)
calendar_max = max(max(available_dates), today_kst)
selected_date = st.date_input(
    "날짜 선택",
    value=today_kst,
    min_value=calendar_min,
    max_value=calendar_max,
    help="캘린더에서 날짜를 선택하면 그날의 기사로 이동합니다.",
)

selected_articles = articles[
    (articles["날짜"].dt.month == selected_date.month)
    & (articles["날짜"].dt.day == selected_date.day)
]

if not selected_articles.empty:
    for year, year_articles in selected_articles.groupby("연도", sort=False):
        st.subheader(f"{year}년 {selected_date.month}월 {selected_date.day}일")
        display_cards(year_articles)
else:
    st.info(f"{selected_date.month}월 {selected_date.day}일에 등록된 기사가 없습니다.")

if st.toggle("더보기"):
    if not ALL_DATA_PATH.exists():
        st.info("`data/all.xlsx` 파일을 찾을 수 없습니다.")
    else:
        try:
            all_articles = load_articles(ALL_DATA_PATH.read_bytes(), preferred_sheet="ALL")
            more_articles = all_articles[
                (all_articles["날짜"].dt.month == selected_date.month)
                & (all_articles["날짜"].dt.day == selected_date.day)
            ]
            if more_articles.empty:
                st.info(f"{selected_date.month}월 {selected_date.day}의 추가 기사가 없습니다.")
            else:
                display_title_links(more_articles)
        except Exception as exc:
            st.error(f"전체 기사 엑셀 파일을 읽지 못했습니다: {exc}")
