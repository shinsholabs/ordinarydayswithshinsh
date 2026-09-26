from __future__ import annotations

from datetime import date
from html import escape
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup


st.set_page_config(page_title="평범한 날들, 신승호", page_icon="🗓️", layout="centered")

DATA_PATH = Path(__file__).parent / "data" / "named.xlsx"
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
def load_articles(file_bytes: bytes) -> pd.DataFrame:
    """Read every article sheet, including workbooks whose NAMED sheet has no header."""
    workbook = pd.ExcelFile(BytesIO(file_bytes))
    frames: list[pd.DataFrame] = []

    # NAMED is the curated article list in the supplied workbook. If it is absent,
    # the app falls back to collecting the remaining data sheets.
    sheet_names = ["NAMED"] if "NAMED" in workbook.sheet_names else workbook.sheet_names
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
    articles = articles.sort_values(["날짜", "시간정렬", "제목"], na_position="last")
    articles["연도"] = articles["날짜"].dt.year
    return articles


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def first_article_image(article_url: str) -> str | None:
    """Find the leading article photo, with Open Graph as a reliable fallback."""
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
    candidates = []
    # Article-content selectors are checked first so logos and navigation images are skipped.
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

    og_image = soup.find("meta", property="og:image") or soup.find("meta", attrs={"name": "twitter:image"})
    if og_image and og_image.get("content"):
        return urljoin(article_url, og_image["content"])
    return None


def display_cards(articles: pd.DataFrame) -> None:
    for index in range(0, len(articles), 2):
        columns = st.columns(2, gap="small")
        for column, (_, article) in zip(columns, articles.iloc[index : index + 2].iterrows()):
            with column:
                image_url = first_article_image(article["url"])
                safe_url = escape(str(article["url"]), quote=True)
                safe_title = escape(str(article["제목"]), quote=True)
                if image_url:
                    st.markdown(
                        f'<a class="article-image" href="{safe_url}" target="_blank">'
                        f'<img src="{escape(image_url, quote=True)}" alt="{safe_title}" loading="lazy"></a>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f'<a class="no-image" href="{safe_url}" target="_blank">사진을 불러올 수 없습니다</a>', unsafe_allow_html=True)
                time_text = "" if pd.isna(article["시간"]) else f" · {str(article["시간"])}"
                st.caption(f'{article["매체명"]}{time_text}')
                st.markdown(f'<a class="article-title" href="{safe_url}" target="_blank">{safe_title}</a>', unsafe_allow_html=True)


st.markdown(
    """<style>
    .block-container {max-width: 760px; padding-top: 2.2rem; padding-bottom: 4rem;}
    h1 {letter-spacing: -0.06em; margin-bottom: 0.2rem;}
    .article-image img, .no-image {width: 100%; aspect-ratio: 1 / 1; object-fit: cover; display: block; border-radius: 5px; background: #efefeb;}
    .no-image {color: #777 !important; padding: 45% 1rem 0; text-align: center; font-size: .8rem;}
    .article-title {display: block; color: #171717 !important; font-size: .93rem; line-height: 1.45; text-decoration: none; margin: -0.25rem 0 1.6rem;}
    .article-title:hover {text-decoration: underline;}
    [data-testid="stCaptionContainer"] {font-size: .72rem; color: #777; margin-top: .45rem;}
    </style>""",
    unsafe_allow_html=True,
)
st.title("평범한 날들, 신승호")
st.caption("날짜를 골라 그날의 기사를 찾아보세요. 사진 또는 제목을 누르면 원문으로 이동합니다.")

if not DATA_PATH.exists():
    st.error("`data/named.xlsx` 파일을 찾을 수 없습니다. 앱 폴더 안의 `data` 폴더에 파일을 넣어 주세요.")
    st.stop()

try:
    articles = load_articles(DATA_PATH.read_bytes())
except Exception as exc:
    st.error(f"엑셀 파일을 읽지 못했습니다: {exc}")
    st.stop()

available_dates = set(articles["날짜"].dt.date)
default_date = max(available_dates)
selected_date = st.date_input(
    "날짜 선택",
    value=default_date,
    min_value=min(available_dates),
    max_value=max(available_dates),
    help="캘린더에서 날짜를 선택하면 그날의 기사로 이동합니다.",
)

if selected_date in available_dates:
    selected_articles = articles[articles["날짜"].dt.date == selected_date]
    st.subheader(selected_date.strftime("%Y년 %m월 %d일"))
    display_cards(selected_articles)
else:
    st.info(f"{selected_date.strftime('%Y년 %m월 %d일')}에 등록된 기사가 없습니다. 아래에서 전체 기록을 볼 수 있어요.")

st.divider()
st.subheader("전체 기록")
years = sorted(articles["연도"].unique())
selected_year = st.selectbox("연도별 보기", years, index=years.index(selected_date.year) if selected_date.year in years else 0)
year_articles = articles[articles["연도"] == selected_year]
st.header(f"{selected_year}년")
for day, day_articles in year_articles.groupby("날짜", sort=True):
    st.markdown(f"#### {day.strftime('%m월 %d일')}")
    display_cards(day_articles)
