from __future__ import annotations

import base64
import calendar
from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError


st.set_page_config(page_title="평범한 날에, 신승호", page_icon="🗓️", layout="centered")

DATA_PATH = Path(__file__).parent / "data" / "named.xlsx"
ALL_DATA_PATH = Path(__file__).parent / "data" / "all.xlsx"
LOGO_PATH = Path(__file__).parent / "data" / "logo.png"
ARTICLE_COLUMNS = ["날짜", "시간", "매체명", "제목", "url"]
IMAGE_COLUMN = "대표이미지"


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame | None:
    """Accept both header-row sheets and headerless article sheets."""
    if frame.empty:
        return None

    frame = frame.iloc[:, :6].copy()
    if frame.shape[1] < 5:
        return None
    frame.columns = ARTICLE_COLUMNS + ([IMAGE_COLUMN] if frame.shape[1] >= 6 else [])
    if IMAGE_COLUMN not in frame.columns:
        frame[IMAGE_COLUMN] = ""
    frame = frame.dropna(subset=["날짜", "제목", "url"])
    frame["날짜"] = pd.to_datetime(frame["날짜"], errors="coerce")
    frame["url"] = frame["url"].astype(str).str.strip()
    frame[IMAGE_COLUMN] = frame[IMAGE_COLUMN].fillna("").astype(str).str.strip()
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


@st.cache_data(persist="disk", show_spinner=False)
def thumbnail_data_uri(image_url: str, article_url: str) -> str | None:
    """Download once and return a compact 360px square WebP thumbnail."""
    try:
        response = requests.get(
            image_url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/153.0.0.0 Safari/537.36",
                "Referer": article_url,
            },
            timeout=(5, 12),
            stream=True,
        )
        response.raise_for_status()
        chunks = []
        size = 0
        for chunk in response.iter_content(64 * 1024):
            size += len(chunk)
            if size > 15 * 1024 * 1024:
                return None
            chunks.append(chunk)

        with Image.open(BytesIO(b"".join(chunks))) as original:
            # Preserve the whole photograph instead of cropping around its centre.
            contained = ImageOps.contain(
                original.convert("RGB"),
                (480, 360),
                method=Image.Resampling.LANCZOS,
            )
            image = Image.new("RGB", (480, 360), color=(244, 244, 240))
            offset = (
                (image.width - contained.width) // 2,
                (image.height - contained.height) // 2,
            )
            image.paste(contained, offset)
            output = BytesIO()
            image.save(output, format="WEBP", quality=72, method=6)
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return f"data:image/webp;base64,{encoded}"
    except (requests.RequestException, OSError, UnidentifiedImageError):
        return None


@st.cache_data(show_spinner=False)
def local_logo_data_uri(logo_bytes: bytes) -> str:
    """Embed the local fallback logo so Streamlit can show it inside HTML cards."""
    encoded = base64.b64encode(logo_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def card_image_source(article_url: str, preferred_image_url: str = "") -> str | None:
    if preferred_image_url.startswith(("http://", "https://")):
        return thumbnail_data_uri(preferred_image_url, article_url) or preferred_image_url
    if LOGO_PATH.exists():
        return local_logo_data_uri(LOGO_PATH.read_bytes())
    return None


def display_article_group(articles: pd.DataFrame) -> None:
    title_links = []
    photos = []
    for _, article in articles.iterrows():
        image_url = card_image_source(article["url"], article.get(IMAGE_COLUMN, ""))
        safe_url = escape(str(article["url"]), quote=True)
        safe_title = escape(str(article["제목"]), quote=True)
        title_links.append(
            f'<li><a class="article-title" href="{safe_url}" target="_blank">{safe_title}</a></li>'
        )
        if image_url:
            image = (
                f'<a class="article-image" href="{safe_url}" target="_blank">'
                f'<img src="{escape(image_url, quote=True)}" alt="{safe_title}" loading="lazy"></a>'
            )
        else:
            image = f'<a class="no-image" href="{safe_url}" target="_blank">사진을 불러올 수 없습니다</a>'
        photos.append(f'<div class="photo-card">{image}</div>')
    st.markdown(
        f'<ul class="article-list">{"".join(title_links)}</ul>'
        f'<section class="photo-grid">{"".join(photos)}</section>',
        unsafe_allow_html=True,
    )


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
    .site-title {font-size: 1.72rem; font-weight: 700; letter-spacing: -.055em; line-height: 1.15; margin: 0;}
    .site-subtitle {font-size: 1.08rem; color: #707070; margin: .08rem 0 1.3rem; line-height: 1.18; letter-spacing: .01em;}
    div[data-testid="stHorizontalBlock"] {flex-wrap: nowrap !important; gap: .55rem;}
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {min-width: 0 !important; width: calc(50% - .275rem) !important; flex: 1 1 0 !important;}
    div[data-testid="stNumberInput"] {max-width: 9rem; margin-bottom: .35rem;}
    div[data-testid="stNumberInput"] input {border-color: #eadfda; background: #fffaf8;}
    button[data-testid="stNumberInputStepDown"] svg,
    button[data-testid="stNumberInputStepUp"] svg {display: none;}
    button[data-testid="stNumberInputStepDown"]::before {content: "◀"; font-size: .65rem;}
    button[data-testid="stNumberInputStepUp"]::before {content: "▶"; font-size: .65rem;}
    div[data-testid="stNumberInput"] button:first-of-type svg,
    div[data-testid="stNumberInput"] button:last-of-type svg {display: none;}
    div[data-testid="stNumberInput"] button:first-of-type::before {content: "◀"; font-size: .65rem;}
    div[data-testid="stNumberInput"] button:last-of-type::before {content: "▶"; font-size: .65rem;}
    .article-list {list-style: none; padding: 0; margin: 0 0 1.1rem;}
    .article-list li {padding: .62rem 0; border-bottom: 1px solid #ececea;}
    .photo-grid {display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .7rem; align-items: start; margin-bottom: 1.45rem;}
    .photo-card {min-width: 0;}
    .article-image img, .no-image {width: 100%; aspect-ratio: 4 / 3; object-fit: contain; display: block; border-radius: 8px; background: #f4f4f0;}
    .no-image {display: grid; place-items: center; color: #777 !important; padding: 1rem; text-align: center; font-size: .8rem;}
    .article-title {display: block; color: #171717 !important; font-size: 1.02rem; font-weight: 400; line-height: 1.48; text-decoration: none !important;}
    .article-title:hover, .article-title:focus, .article-title:visited {text-decoration: none !important;}
    .year-heading {font-size: 1.18rem; font-weight: 700; letter-spacing: -.025em; margin: 1.55rem 0 .7rem;}
    .more-list {list-style: none; padding: 0; margin: .35rem 0 1.5rem;}
    .more-list li {padding: .7rem 0; border-bottom: 1px solid #ececea;}
    .more-year {font-size: 1.08rem; margin: 1.25rem 0 .15rem;}
    .more-title {color: #171717 !important; font-size: .93rem; line-height: 1.5; text-decoration: none !important;}
    .more-title:hover, .more-title:focus, .more-title:visited {text-decoration: none !important;}
    div[data-testid="stButton"] > button {border: 0 !important; background: transparent !important; box-shadow: none !important; padding: .25rem 0 !important; min-height: auto !important; color: #555 !important;}
    div[data-testid="stButton"] > button:hover {color: #111 !important; background: transparent !important;}
    div[data-testid="stButton"] > button:focus {box-shadow: none !important;}
    </style>""",
    unsafe_allow_html=True,
)
st.markdown('<h1 class="site-title">평범한 날에, 신승호</h1>', unsafe_allow_html=True)
st.markdown('<p class="site-subtitle">Ordinary days with SHIN SEUNGHO</p>', unsafe_allow_html=True)

if not DATA_PATH.exists():
    st.error("`data/named.xlsx` 파일을 찾을 수 없습니다. 앱 폴더 안의 `data` 폴더에 파일을 넣어 주세요.")
    st.stop()

try:
    articles = load_articles(DATA_PATH.read_bytes(), preferred_sheet="NAMED")
except Exception as exc:
    st.error(f"엑셀 파일을 읽지 못했습니다: {exc}")
    st.stop()

today_kst = datetime.now(ZoneInfo("Asia/Seoul")).date()
month_state_key = "selected_month_v16"
day_state_key = "selected_day_v16"
if month_state_key not in st.session_state:
    st.session_state[month_state_key] = today_kst.month
if day_state_key not in st.session_state:
    st.session_state[day_state_key] = today_kst.day

month_column, day_column = st.columns(2)
with month_column:
    selected_month = int(st.number_input(
        "월",
        min_value=1,
        max_value=12,
        step=1,
        key=month_state_key,
    ))

maximum_day = calendar.monthrange(2024, selected_month)[1]
if st.session_state[day_state_key] > maximum_day:
    st.session_state[day_state_key] = maximum_day
with day_column:
    selected_day = int(st.number_input(
        "일",
        min_value=1,
        max_value=maximum_day,
        step=1,
        key=day_state_key,
    ))

selected_articles = articles[
    (articles["날짜"].dt.month == selected_month)
    & (articles["날짜"].dt.day == selected_day)
]

if not selected_articles.empty:
    for year, year_articles in selected_articles.groupby("연도", sort=False):
        st.markdown(
            f'<h2 class="year-heading">{year}</h2>',
            unsafe_allow_html=True,
        )
        display_article_group(year_articles)
else:
    st.info(f"{selected_month}월 {selected_day}일에 등록된 기사가 없습니다.")

selected_day_key = f"{selected_month:02d}-{selected_day:02d}"
if st.button("> 더보기", key=f"more_{selected_day_key}"):
    st.session_state["more_open_day"] = selected_day_key

if st.session_state.get("more_open_day") == selected_day_key:
    if not ALL_DATA_PATH.exists():
        st.info("`data/all.xlsx` 파일을 찾을 수 없습니다.")
    else:
        try:
            all_articles = load_articles(ALL_DATA_PATH.read_bytes(), preferred_sheet="ALL")
            more_articles = all_articles[
                (all_articles["날짜"].dt.month == selected_month)
                & (all_articles["날짜"].dt.day == selected_day)
            ]
            if more_articles.empty:
                st.info(f"{selected_month}월 {selected_day}일의 추가 기사가 없습니다.")
            else:
                for year, year_articles in more_articles.groupby("연도", sort=False):
                    st.markdown(
                        f'<h3 class="more-year">{year}</h3>',
                        unsafe_allow_html=True,
                    )
                    display_title_links(year_articles)
        except Exception as exc:
            st.error(f"전체 기사 엑셀 파일을 읽지 못했습니다: {exc}")
