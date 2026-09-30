from __future__ import annotations

import base64
import calendar
from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError


st.set_page_config(page_title="평범한 날에, 신승호", page_icon="🗓️", layout="centered")

DATA_PATH = Path(__file__).parent / "data" / "named.xlsx"
ALL_DATA_PATH = Path(__file__).parent / "data" / "all.xlsx"
LOGO_PATH = Path(__file__).parent / "data" / "logo.png"
YUNET_MODEL_PATH = Path(__file__).parent / "data" / "face_detection_yunet_2023mar.onnx"
YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
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


@st.cache_resource(show_spinner=False)
def load_face_detector():
    """Load YuNet once, downloading the small official model when necessary."""
    try:
        import cv2
    except ImportError:
        return None

    try:
        if YUNET_MODEL_PATH.exists() and YUNET_MODEL_PATH.stat().st_size > 100_000:
            model_path = YUNET_MODEL_PATH
        else:
            model_path = Path(tempfile.gettempdir()) / "face_detection_yunet_2023mar.onnx"

        if not model_path.exists() or model_path.stat().st_size <= 100_000:
            response = requests.get(
                YUNET_MODEL_URL,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=(5, 30),
            )
            response.raise_for_status()
            if len(response.content) <= 100_000:
                return None
            temporary_path = model_path.with_suffix(".download")
            temporary_path.write_bytes(response.content)
            temporary_path.replace(model_path)

        return cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (320, 320),
            score_threshold=0.65,
            nms_threshold=0.3,
            top_k=1000,
        )
    except (OSError, requests.RequestException, cv2.error):
        return None


def face_focused_thumbnail(original: Image.Image) -> Image.Image:
    """Create a fixed 4:3 thumbnail focused on the largest face and upper body."""
    image = ImageOps.exif_transpose(original).convert("RGB")
    detector = load_face_detector()

    if detector is not None:
        try:
            import cv2
            import numpy as np

            detection_scale = min(1.0, 640 / max(image.width, image.height))
            detection_image = image
            if detection_scale < 1.0:
                detection_image = image.resize(
                    (
                        max(1, round(image.width * detection_scale)),
                        max(1, round(image.height * detection_scale)),
                    ),
                    Image.Resampling.LANCZOS,
                )

            rgb = np.asarray(detection_image)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            detector.setInputSize((detection_image.width, detection_image.height))
            _, faces = detector.detect(bgr)

            if faces is not None and len(faces):
                detected_face = max(
                    faces,
                    key=lambda face: float(face[2]) * float(face[3]) * float(face[14]),
                )
                face_x, face_y, face_width, face_height = (
                    float(value) / detection_scale for value in detected_face[:4]
                )

                maximum_crop_height = min(image.height, image.width * 3 / 4)
                crop_height = min(
                    maximum_crop_height,
                    max(face_height * 4.5, maximum_crop_height * 0.6),
                )
                crop_width = crop_height * 4 / 3
                face_center_x = face_x + face_width / 2
                face_center_y = face_y + face_height / 2

                left = max(0.0, min(image.width - crop_width, face_center_x - crop_width / 2))
                top = max(0.0, min(image.height - crop_height, face_center_y - crop_height * 0.30))
                crop = image.crop(
                    (
                        round(left),
                        round(top),
                        round(left + crop_width),
                        round(top + crop_height),
                    )
                )
                return ImageOps.fit(
                    crop,
                    (480, 360),
                    method=Image.Resampling.LANCZOS,
                )
        except (ImportError, ValueError, TypeError, cv2.error):
            pass

    # If no face is found, prefer the upper centre where a person's head and
    # torso usually appear in press photographs.
    target_ratio = 4 / 3
    source_ratio = image.width / image.height
    if source_ratio < target_ratio:
        crop_width = image.width
        crop_height = crop_width / target_ratio
        left = 0
        top = max(0, (image.height - crop_height) * 0.12)
    else:
        crop_height = image.height
        crop_width = crop_height * target_ratio
        left = max(0, (image.width - crop_width) / 2)
        top = 0

    upper_crop = image.crop(
        (
            round(left),
            round(top),
            round(left + crop_width),
            round(top + crop_height),
        )
    )
    return ImageOps.fit(
        upper_crop,
        (480, 360),
        method=Image.Resampling.LANCZOS,
    )


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
            image = face_focused_thumbnail(original)
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


def display_cards(articles: pd.DataFrame) -> None:
    cards = []
    for _, article in articles.iterrows():
        image_url = card_image_source(article["url"], article.get(IMAGE_COLUMN, ""))
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
            f'<article class="article-card">'
            f'<a class="article-title" href="{safe_url}" target="_blank">{safe_title}</a>'
            f'{image}</article>'
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
    .article-grid {display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .7rem; align-items: start;}
    .article-card {min-width: 0; margin-bottom: 1.45rem;}
    .article-image img, .no-image {width: 100%; aspect-ratio: 4 / 3; object-fit: cover; display: block; border-radius: 8px; background: #f4f4f0;}
    .no-image {display: grid; place-items: center; color: #777 !important; padding: 1rem; text-align: center; font-size: .8rem;}
    .article-title {display: block; min-height: 3.1em; color: #171717 !important; font-size: 1.02rem; font-weight: 400; line-height: 1.48; text-decoration: none !important; margin: 0 0 .55rem;}
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
        display_cards(year_articles)
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
