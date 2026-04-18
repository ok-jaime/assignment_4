from __future__ import annotations

import base64
import html
from io import BytesIO
import os
from pathlib import Path
import re
from typing import Dict
from uuid import uuid4

from docx import Document
from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pypdf import PdfReader
from striprtf.striprtf import rtf_to_text
import streamlit as st


SYSTEM_PROMPT = """You are a careful writing assistant for a class assignment.
Follow the user's instructions exactly.
Return concise, polished, beginner-friendly writing.
"""

TITLE_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)

SOCIAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)

HISTORY_STORE: Dict[str, InMemoryChatMessageHistory] = {}


def setup_page() -> None:
    st.set_page_config(page_title="Article Studio", page_icon="AI", layout="wide")


def init_state() -> None:
    defaults = {
        "session_id": f"assignment-{uuid4().hex[:8]}",
        "article_text": "",
        "article_name": "",
        "parse_method": "",
        "title_history": [],
        "title_index": -1,
        "title_draft": "",
        "accepted_title": "",
        "summary_history": [],
        "summary_index": -1,
        "summary_draft": "",
        "accepted_summary": "",
        "selected_platform": "LinkedIn",
        "keywords": "",
        "social_history": [],
        "social_index": -1,
        "social_post": "",
        "accepted_social_post": "",
        "generated_image_b64": "",
        "generated_image_prompt": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_current_version(history_key: str, index_key: str) -> str:
    history = st.session_state[history_key]
    idx = st.session_state[index_key]
    if not history or idx < 0:
        return ""
    return history[idx]


def append_version(history_key: str, index_key: str, new_text: str) -> None:
    st.session_state[history_key].append(new_text)
    st.session_state[index_key] = len(st.session_state[history_key]) - 1


def move_version(history_key: str, index_key: str, step: int) -> None:
    history = st.session_state[history_key]
    if not history:
        return
    max_idx = len(history) - 1
    st.session_state[index_key] = max(0, min(max_idx, st.session_state[index_key] + step))


def create_chat_model() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is missing. Add it to .env.")

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model_name, temperature=0.7)


def create_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is missing. Add it to .env.")
    return OpenAI(api_key=api_key)


def build_image_prompt(article: str, title: str, summary: str, user_prompt: str) -> str:
    base_prompt = f"""
Create a clean editorial illustration for an article.

Article title:
{title}

Article summary:
{summary}

Article context:
{article}
""".strip()
    if user_prompt.strip():
        return f"{base_prompt}\n\nUser direction:\n{user_prompt.strip()}"
    return base_prompt


def generate_article_image(client: OpenAI, prompt: str) -> bytes:
    response = client.images.generate(
        model="gpt-image-1-mini",
        prompt=prompt,
        size="1024x1024",
        quality="medium",
    )
    image_data = response.data[0]
    b64_json = getattr(image_data, "b64_json", None)
    if not b64_json:
        raise ValueError("No image data was returned by the model.")
    return base64.b64decode(b64_json)


def build_chain(prompt: ChatPromptTemplate, model: ChatOpenAI) -> RunnableWithMessageHistory:
    base_chain = prompt | model | StrOutputParser()

    # Keep an in-memory history per session_id.
    def get_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in HISTORY_STORE:
            HISTORY_STORE[session_id] = InMemoryChatMessageHistory()
        return HISTORY_STORE[session_id]

    return RunnableWithMessageHistory(
        base_chain,
        get_history,
        input_messages_key="input",
        history_messages_key="history",
    )


def build_title_input(article: str, previous_title: str, revision_guidance: str) -> str:
    return f"""
Create a strong title for the article below.
Use the revision guidance if present.

Article:
{article}

Previous title:
{previous_title or 'None'}

Revision guidance:
{revision_guidance or 'None'}

Return only the title text.
""".strip()


def build_summary_input(article: str, accepted_title: str, previous_summary: str, revision_guidance: str) -> str:
    return f"""
Write a concise summary for the article below using the accepted title.
Use the revision guidance if present.

Article:
{article}

Accepted title:
{accepted_title}

Previous summary:
{previous_summary or 'None'}

Revision guidance:
{revision_guidance or 'None'}

Return only the summary text.
""".strip()


def build_social_input(
    article: str,
    accepted_title: str,
    accepted_summary: str,
    selected_platform: str,
    keywords: str,
) -> str:
    return f"""
Write a short social media blurb about the article.

Article:
{article}

Accepted title:
{accepted_title}

Accepted summary:
{accepted_summary}

Platform:
{selected_platform}

Keywords:
{keywords}

Requirements:
- Keep it short and suitable for the platform.
- Include exactly three suggested hashtags.
- Return only the post text.
""".strip()


def invoke_chain(chain: RunnableWithMessageHistory, session_id: str, input_text: str) -> str:
    return chain.invoke(
        {"input": input_text},
        config={"configurable": {"session_id": session_id}},
    ).strip()


def extract_with_openai_file(client: OpenAI, file_bytes: bytes, filename: str, mime_type: str) -> str:
    b64_data = base64.b64encode(file_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64_data}"
    model_name = os.getenv("OPENAI_VISION_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    response = client.responses.create(
        model=model_name,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Extract all readable text from this file. Return plain text only.",
                    },
                    {
                        "type": "input_file",
                        "filename": filename,
                        "file_data": data_url,
                    },
                ],
            }
        ],
        temperature=0,
    )
    return (response.output_text or "").strip()


def extract_with_openai_image(client: OpenAI, file_bytes: bytes, mime_type: str) -> str:
    b64_data = base64.b64encode(file_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64_data}"
    model_name = os.getenv("OPENAI_VISION_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    response = client.responses.create(
        model=model_name,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Read this image and extract all visible text. Return plain text only.",
                    },
                    {
                        "type": "input_image",
                        "image_url": data_url,
                    },
                ],
            }
        ],
        temperature=0,
    )
    return (response.output_text or "").strip()


def local_extract_text(filename: str, file_bytes: bytes) -> str:
    suffix = Path(filename).suffix.lower()

    if suffix in {".md", ".txt"}:
        return file_bytes.decode("utf-8", errors="ignore").strip()

    if suffix == ".pdf":
        reader = PdfReader(BytesIO(file_bytes))
        return "\n".join((page.extract_text() or "") for page in reader.pages).strip()

    if suffix == ".docx":
        document = Document(BytesIO(file_bytes))
        return "\n".join(paragraph.text for paragraph in document.paragraphs).strip()

    if suffix == ".rtf":
        raw = file_bytes.decode("utf-8", errors="ignore")
        return rtf_to_text(raw).strip()

    return ""


def extract_text_from_upload(uploaded_file, client: OpenAI) -> tuple[str, str]:
    if uploaded_file is None:
        raise ValueError("Please upload a file first.")

    file_bytes = uploaded_file.getvalue()
    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    suffix = Path(uploaded_file.name).suffix.lower()
    mime_type = uploaded_file.type or "application/octet-stream"

    local_text = local_extract_text(uploaded_file.name, file_bytes)
    if local_text:
        return local_text, "local-parser"

    image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
    if suffix in image_suffixes or mime_type.startswith("image/"):
        text = extract_with_openai_image(client, file_bytes, mime_type)
        if text:
            return text, "openai-vision"
        raise ValueError("Could not extract text from the image.")

    text = extract_with_openai_file(client, file_bytes, uploaded_file.name, mime_type)
    if text:
        return text, "openai-file-parser"
    raise ValueError("Could not extract text from the uploaded file.")


def enforce_exactly_three_hashtags(post_text: str, keywords: str, accepted_title: str) -> str:
    tags = re.findall(r"#\w+", post_text)
    if len(tags) == 3:
        return post_text.strip()

    words = re.findall(r"[A-Za-z0-9]+", f"{keywords} {accepted_title}")
    selected = []
    seen = set()
    for word in words:
        token = "#" + word.lower()
        if len(token) < 4:
            continue
        if token not in seen:
            selected.append(token)
            seen.add(token)
        if len(selected) == 3:
            break

    fallback = ["#studentproject", "#contentstrategy", "#genai"]
    while len(selected) < 3:
        selected.append(fallback[len(selected)])

    body = re.sub(r"#\w+", "", post_text)
    body = re.sub(r"\s+", " ", body).strip()
    return f"{body}\n\nSuggested hashtags: {' '.join(selected[:3])}".strip()


def reset_downstream() -> None:
    st.session_state.summary_history = []
    st.session_state.summary_index = -1
    st.session_state.summary_draft = ""
    st.session_state.accepted_summary = ""
    st.session_state.social_history = []
    st.session_state.social_index = -1
    st.session_state.social_post = ""
    st.session_state.accepted_social_post = ""
    reset_image()


def reset_image() -> None:
    st.session_state.generated_image_b64 = ""
    st.session_state.generated_image_prompt = ""


def generate_title_version(title_chain: RunnableWithMessageHistory, revision_guidance: str = "") -> str:
    new_title = invoke_chain(
        title_chain,
        st.session_state.session_id,
        build_title_input(
            st.session_state.article_text,
            get_current_version("title_history", "title_index"),
            revision_guidance,
        ),
    )
    append_version("title_history", "title_index", new_title)
    st.session_state.title_draft = new_title
    return new_title


def generate_summary_version(summary_chain: RunnableWithMessageHistory, revision_guidance: str = "") -> str:
    new_summary = invoke_chain(
        summary_chain,
        st.session_state.session_id,
        build_summary_input(
            st.session_state.article_text,
            st.session_state.accepted_title,
            get_current_version("summary_history", "summary_index"),
            revision_guidance,
        ),
    )
    append_version("summary_history", "summary_index", new_summary)
    st.session_state.summary_draft = new_summary
    return new_summary


def render_featured_text(label: str, text: str, font_size: str = "1.3rem") -> None:
    if not text:
        return
    escaped_text = html.escape(text).replace("\n", "<br>")
    st.markdown(
        f"<div><strong>{label}</strong></div><div style='font-size:{font_size}; font-weight:700; line-height:1.5; margin-top:0.25rem;'>{escaped_text}</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    load_dotenv()
    setup_page()
    init_state()

    st.title("Article Studio")
    st.write("Upload a file, iterate on title and summary, then generate a social post with exactly three hashtags.")

    try:
        model = create_chat_model()
        openai_client = create_openai_client()
    except EnvironmentError as exc:
        st.error(str(exc))
        return

    title_chain = build_chain(TITLE_PROMPT, model)
    summary_chain = build_chain(SUMMARY_PROMPT, model)
    social_chain = build_chain(SOCIAL_PROMPT, model)

    uploaded_file = st.file_uploader(
        "Upload article or media",
        type=["md", "txt", "pdf", "rtf", "docx", "doc", "png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff"],
        help="Uses local extraction first. Falls back to OpenAI extraction for images and hard-to-parse files.",
    )
    parse_clicked = st.button("Parse Uploaded File", use_container_width=True)

    if parse_clicked:
        try:
            article_text, parse_method = extract_text_from_upload(uploaded_file, openai_client)
            st.session_state.article_text = article_text
            st.session_state.article_name = uploaded_file.name if uploaded_file else ""
            st.session_state.parse_method = parse_method
            st.session_state.title_history = []
            st.session_state.title_index = -1
            st.session_state.title_draft = ""
            st.session_state.accepted_title = ""
            reset_downstream()
            reset_image()
            generate_title_version(title_chain)
            st.success(f"Extraction complete using: {parse_method}")
        except Exception as exc:
            st.error(f"Parsing failed: {exc}")

    if st.session_state.article_text:
        st.subheader("Parsed Article")
        st.caption(
            f"Source: {st.session_state.article_name} | Parser: {st.session_state.parse_method} | Session: {st.session_state.session_id}"
        )
        show_parsed_media = st.toggle("Show Parsed Media", value=False)
        if show_parsed_media:
            st.text_area("Parsed Media", st.session_state.article_text, height=220)

        st.subheader("1) Title")
        render_featured_text("Suggested Title", st.session_state.title_draft, font_size="1.55rem")
        title_revision = st.text_input("Revision space for title (optional)")
        col_t1, col_t2, col_t3 = st.columns(3)

        if col_t1.button("Revise Title", use_container_width=True):
            generate_title_version(title_chain, title_revision)
            st.session_state.accepted_title = ""
            reset_downstream()

        if col_t2.button("Previous Title", use_container_width=True):
            move_version("title_history", "title_index", -1)
            st.session_state.title_draft = get_current_version("title_history", "title_index")

        if col_t3.button("Next Title", use_container_width=True):
            move_version("title_history", "title_index", 1)
            st.session_state.title_draft = get_current_version("title_history", "title_index")

        if st.button("Accept Title", use_container_width=True):
            selected_title = get_current_version("title_history", "title_index")
            if selected_title:
                st.session_state.accepted_title = selected_title
                st.session_state.summary_history = []
                st.session_state.summary_index = -1
                st.session_state.summary_draft = ""
                st.session_state.accepted_summary = ""
                st.session_state.social_history = []
                st.session_state.social_index = -1
                st.session_state.social_post = ""
                st.session_state.accepted_social_post = ""
                reset_image()
                generate_summary_version(summary_chain)
            else:
                st.warning("Generate a title first.")

        if st.session_state.title_history:
            st.caption(f"Title version {st.session_state.title_index + 1} of {len(st.session_state.title_history)}")
        if st.session_state.accepted_title:
            st.success(f"Accepted title: {st.session_state.accepted_title}")

    if st.session_state.accepted_title:
        st.subheader("2) Summary")
        render_featured_text("Suggested Article Summary", st.session_state.summary_draft, font_size="1.35rem")
        summary_revision = st.text_input("Revision space for summary (optional)")
        col_s1, col_s2, col_s3 = st.columns(3)

        if col_s1.button("Revise Summary", use_container_width=True):
            generate_summary_version(summary_chain, summary_revision)
            st.session_state.accepted_summary = ""
            st.session_state.social_history = []
            st.session_state.social_index = -1
            st.session_state.social_post = ""
            st.session_state.accepted_social_post = ""
            reset_image()

        if col_s2.button("Previous Summary", use_container_width=True):
            move_version("summary_history", "summary_index", -1)
            st.session_state.summary_draft = get_current_version("summary_history", "summary_index")

        if col_s3.button("Next Summary", use_container_width=True):
            move_version("summary_history", "summary_index", 1)
            st.session_state.summary_draft = get_current_version("summary_history", "summary_index")

        if st.button("Accept Summary", use_container_width=True):
            selected_summary = get_current_version("summary_history", "summary_index")
            if selected_summary:
                st.session_state.accepted_summary = selected_summary
                reset_image()
            else:
                st.warning("Generate a summary first.")

        if st.session_state.summary_history:
            st.caption(f"Summary version {st.session_state.summary_index + 1} of {len(st.session_state.summary_history)}")
        if st.session_state.accepted_summary:
            st.success("Accepted summary saved.")

    if st.session_state.accepted_summary:
        st.subheader("3) Social Blurb")
        platforms = ["LinkedIn", "Instagram", "X (Twitter)", "Facebook", "Threads", "TikTok"]
        selected_platform = st.selectbox("Social platform", platforms)
        keywords = st.text_input("Optional keywords (comma-separated)")

        col_p1, col_p2, col_p3, col_p4 = st.columns(4)

        if col_p1.button("Generate Social Post", use_container_width=True):
            st.session_state.selected_platform = selected_platform
            st.session_state.keywords = keywords
            raw_post = invoke_chain(
                social_chain,
                st.session_state.session_id,
                build_social_input(
                    st.session_state.article_text,
                    st.session_state.accepted_title,
                    st.session_state.accepted_summary,
                    st.session_state.selected_platform,
                    st.session_state.keywords or "None provided",
                ),
            )
            new_post = enforce_exactly_three_hashtags(
                raw_post,
                st.session_state.keywords,
                st.session_state.accepted_title,
            )
            append_version("social_history", "social_index", new_post)
            st.session_state.social_post = new_post
            st.session_state.accepted_social_post = ""

        if col_p2.button("Previous Post", use_container_width=True):
            move_version("social_history", "social_index", -1)
            st.session_state.social_post = get_current_version("social_history", "social_index")

        if col_p3.button("Next Post", use_container_width=True):
            move_version("social_history", "social_index", 1)
            st.session_state.social_post = get_current_version("social_history", "social_index")

        if col_p4.button("Accept Post", use_container_width=True):
            selected_post = get_current_version("social_history", "social_index")
            if selected_post:
                st.session_state.accepted_social_post = selected_post
            else:
                st.warning("Generate a social post first.")

        if st.session_state.social_history:
            st.caption(f"Social version {st.session_state.social_index + 1} of {len(st.session_state.social_history)}")
        st.text_area("Suggested social post", st.session_state.social_post, height=180)
        if st.session_state.accepted_social_post:
            st.success("Accepted social post saved.")

    if st.session_state.accepted_social_post:
        st.subheader("4) Optional Image")
        image_prompt = st.text_area(
            "Image direction (optional)",
            placeholder="Describe the visual style, subject, or mood you want. Leave blank to auto-generate from the article.",
            height=120,
        )

        if st.button("Generate Article Image", use_container_width=True):
            try:
                prompt = build_image_prompt(
                    st.session_state.article_text,
                    st.session_state.accepted_title,
                    st.session_state.accepted_summary,
                    image_prompt,
                )
                image_bytes = generate_article_image(openai_client, prompt)
                st.session_state.generated_image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                st.session_state.generated_image_prompt = prompt
                st.success("Image generated successfully.")
            except Exception as exc:
                st.error(f"Image generation failed: {exc}")

        if st.session_state.generated_image_b64:
            st.image(base64.b64decode(st.session_state.generated_image_b64), caption="Generated article image", use_container_width=True)
            st.download_button(
                "Download image",
                data=base64.b64decode(st.session_state.generated_image_b64),
                file_name="article_image.png",
                mime="image/png",
                use_container_width=True,
            )

    if st.session_state.accepted_social_post:
        st.subheader("Final Output")
        st.markdown(f"### Title\n{st.session_state.accepted_title}")
        st.markdown(f"### Summary\n{st.session_state.accepted_summary}")
        st.markdown(f"### Social Post ({st.session_state.selected_platform})\n{st.session_state.accepted_social_post}")


if __name__ == "__main__":
    main()
