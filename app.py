import os
import re
from functools import lru_cache
from typing import Dict, List, Tuple

import gradio as gr
import requests
from transformers import pipeline

TYPE_LABELS = [
    "Logic/Bug",
    "Suggestion",
    "Style/Nitpick",
    "Question",
    "Praise",
]

SENTIMENT_LABELS = [
    "Positive",
    "Neutral",
    "Negative",
]

GITHUB_REVIEW_URL = re.compile(
    r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>\d+)(?:/[^#]*)?(?:#(?P<fragment>.*))?",
    re.IGNORECASE,
)

MAX_COMMENT_LENGTH = 4000
REQUEST_TIMEOUT_SECONDS = 10
APP_USER_AGENT = "CodeReviewQualityAnalyzer/0.1"

def _extract_comment_id(fragment: str) -> Tuple[str, str]:
    if not fragment:
        raise ValueError("URL must include a fragment pointing to a specific comment.")

    discussion_match = re.search(r"discussion_r(\d+)", fragment)
    if discussion_match:
        return "pull_review_comment", discussion_match.group(1)

    issue_match = re.search(r"issuecomment-(\d+)", fragment)
    if issue_match:
        return "issue_comment", issue_match.group(1)

    review_match = re.search(r"pullrequestreview-(\d+)", fragment)
    if review_match:
        return "pull_review", review_match.group(1)

    raise ValueError(
        "Unsupported GitHub fragment. Supported fragments include '#discussion_r<ID>' and '#issuecomment-<ID>'."
    )

def _github_headers() -> Dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": APP_USER_AGENT,
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token.strip()}"
    return headers


def fetch_comment_from_github(url: str) -> str:
    match = GITHUB_REVIEW_URL.match(url.strip())
    if not match:
        raise ValueError("Only GitHub pull request comment URLs are supported at the moment.")

    owner = match.group("owner")
    repo = match.group("repo")
    fragment = match.group("fragment")

    comment_type, comment_id = _extract_comment_id(fragment)

    if comment_type == "pull_review_comment":
        api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/comments/{comment_id}"
    elif comment_type == "issue_comment":
        api_url = f"https://api.github.com/repos/{owner}/{repo}/issues/comments/{comment_id}"
    elif comment_type == "pull_review":
        api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/reviews/{comment_id}"
    else:
        raise ValueError("Unsupported comment type.")

    try:
        response = requests.get(
            api_url,
            headers=_github_headers(),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as err:
        raise ValueError("Unable to reach GitHub. Check your network connection or try again later.") from err

    if response.status_code == 404:
        raise ValueError("Comment not found. Double-check that the link points to a public review comment.")
    if response.status_code == 403:
        raise ValueError(
            "GitHub API rate limit exceeded or access forbidden. Try again later or paste the comment text manually."
        )
    response.raise_for_status()

    payload = response.json()

    if "body" not in payload:
        raise ValueError("Unable to extract comment body from GitHub response.")

    return payload["body"].strip()

@lru_cache(maxsize=1)
def get_zero_shot_pipeline():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)

def build_table(labels: List[str], scores: List[float]) -> List[List[str]]:
    rows: List[List[str]] = []
    for label, score in zip(labels, scores):
        rows.append([label, f"{score:.2%}"])
    return rows

def classify_comment(comment: str) -> Dict[str, object]:
    classifier = get_zero_shot_pipeline()

    type_result = classifier(comment, TYPE_LABELS, multi_label=False)
    sentiment_result = classifier(comment, SENTIMENT_LABELS, multi_label=False)

    best_type = type_result["labels"][0]
    best_type_score = type_result["scores"][0]

    best_sentiment = sentiment_result["labels"][0]
    best_sentiment_score = sentiment_result["scores"][0]

    type_table = build_table(type_result["labels"], type_result["scores"])
    sentiment_table = build_table(sentiment_result["labels"], sentiment_result["scores"])

    summary = (
        f"**Feedback Type:** {best_type} ({best_type_score:.1%} confidence)\n"
        f"**Sentiment:** {best_sentiment} ({best_sentiment_score:.1%} confidence)\n"
    )

    return {
        "summary": summary,
        "type_rows": type_table,
        "sentiment_rows": sentiment_table,
    }

def analyze_comment(comment_text: str, review_url: str):
    comment_text = (comment_text or "").strip()
    review_url = (review_url or "").strip()

    if comment_text and review_url:
        try:
            fetched_comment = fetch_comment_from_github(review_url)
            # Prioritize pasted text but expose fetched variant for comparison.
            combined_comment = comment_text
            comment_note = (
                "Using the pasted comment text. Fetched GitHub comment is shown in the preview for reference."
            )
        except Exception:
            fetched_comment = ""
            combined_comment = comment_text
            comment_note = "Using the pasted comment text."
    elif comment_text:
        combined_comment = comment_text
        fetched_comment = ""
        comment_note = "Using the pasted comment text."
    elif review_url:
        try:
            combined_comment = fetch_comment_from_github(review_url)
            fetched_comment = combined_comment
            comment_note = "Using the comment fetched from GitHub."
        except Exception as err:
            raise gr.Error(str(err))
    else:
        raise gr.Error("Provide either comment text or a GitHub review URL to analyze.")

    if not combined_comment:
        raise gr.Error("Could not determine any comment text to analyze.")

    if len(combined_comment) > MAX_COMMENT_LENGTH:
        raise gr.Error(f"Comment is too long. Please provide text under {MAX_COMMENT_LENGTH:,} characters.")

    analysis = classify_comment(combined_comment)

    preview_parts = [comment_note]
    preview_parts.append("")
    preview_parts.append(combined_comment)
    preview = "\n".join(preview_parts).strip()

    fetched_preview = fetched_comment if fetched_comment else ""

    return (
        analysis["summary"],
        analysis["type_rows"],
        analysis["sentiment_rows"],
        preview,
        fetched_preview,
    )

with gr.Blocks(title="Code Review Quality Analyzer") as demo:
    gr.Markdown(
        "# Code Review Quality Analyzer\n"
        "Paste a code review comment or provide a GitHub review URL to classify the feedback type and sentiment.\n"
        "This demo uses the open-source zero-shot classifier `facebook/bart-large-mnli` so it runs on CPU-only Spaces."
    )

    with gr.Row():
        comment_input = gr.Textbox(
            label="Review Comment Text",
            placeholder="Paste a single review comment...",
            lines=6,
        )
        url_input = gr.Textbox(
            label="GitHub Review URL",
            placeholder="https://github.com/org/repo/pull/123#discussion_r456",
            lines=2,
        )

    analyze_button = gr.Button("Analyze Review")

    summary_output = gr.Markdown(label="Classification Summary")
    type_output = gr.Dataframe(
        headers=["Label", "Confidence"],
        label="Feedback Type Confidence",
        datatype=["str", "str"],
        interactive=False,
    )
    sentiment_output = gr.Dataframe(
        headers=["Label", "Confidence"],
        label="Sentiment Confidence",
        datatype=["str", "str"],
        interactive=False,
    )
    preview_output = gr.Textbox(label="Analyzed Comment", lines=6)
    fetched_preview_output = gr.Textbox(label="Fetched GitHub Comment", lines=6)

    analyze_button.click(
        analyze_comment,
        inputs=[comment_input, url_input],
        outputs=[summary_output, type_output, sentiment_output, preview_output, fetched_preview_output],
    )

if __name__ == "__main__":
    demo.launch()
