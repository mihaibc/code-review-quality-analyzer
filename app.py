"""Code Review Quality Analyzer (Gradio / HF Spaces)

This app classifies a single code review comment by:
  - Feedback Type: Logic/Bug, Suggestion, Style/Nitpick, Question, Praise
  - Sentiment: Positive, Neutral, Negative

It uses a zero-shot classifier (`facebook/bart-large-mnli`) so it runs on CPU.
You can paste comment text directly, or fetch from a GitHub PR comment URL.
"""

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
PIPELINE_MODEL_ID = "facebook/bart-large-mnli"

# Simple emojis to make results easier to scan at a glance.
TYPE_EMOJI = {
    "Logic/Bug": "🐞",
    "Suggestion": "💡",
    "Style/Nitpick": "✏️",
    "Question": "❓",
    "Praise": "🙌",
}
SENTIMENT_EMOJI = {
    "Positive": "🙂",
    "Neutral": "😐",
    "Negative": "🙁",
}

def _extract_comment_id(fragment: str) -> Tuple[str, str]:
    """Parse the fragment from a PR URL and extract the comment type and id."""
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
    """Build GitHub headers, optionally adding a bearer token to increase limits."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": APP_USER_AGENT,
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token.strip()}"
    return headers


def fetch_comment_from_github(url: str) -> str:
    """Fetch a PR review comment body from a public GitHub URL.

    Supported fragments:
      - #discussion_r<ID>
      - #issuecomment-<ID>
      - #pullrequestreview-<ID>
    """
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
    """Lazily load the zero-shot pipeline on CPU."""
    return pipeline("zero-shot-classification", model=PIPELINE_MODEL_ID, device=-1)

def build_table(labels: List[str], scores: List[float]) -> List[List[str]]:
    """Convert labels + scores into a 2D table for display."""
    rows: List[List[str]] = []
    for label, score in zip(labels, scores):
        rows.append([label, f"{score:.2%}"])
    return rows

def _format_summary(best_type: str, best_type_score: float, best_sentiment: str, best_sentiment_score: float) -> str:
    """Build a professional, emoji-enhanced Markdown summary."""
    type_emoji = TYPE_EMOJI.get(best_type, "")
    sent_emoji = SENTIMENT_EMOJI.get(best_sentiment, "")
    return (
        f"### Result\n"
        f"- Feedback Type: {type_emoji} {best_type} ({best_type_score:.1%})\n"
        f"- Sentiment: {sent_emoji} {best_sentiment} ({best_sentiment_score:.1%})\n"
        f"\n"
        f"Model: `{PIPELINE_MODEL_ID}` · Device: CPU · Method: zero-shot\n"
    )


def classify_comment(comment: str) -> Dict[str, object]:
    """Run zero-shot classification for feedback type and sentiment."""
    classifier = get_zero_shot_pipeline()

    type_result = classifier(comment, TYPE_LABELS, multi_label=False)
    sentiment_result = classifier(comment, SENTIMENT_LABELS, multi_label=False)

    best_type = type_result["labels"][0]
    best_type_score = type_result["scores"][0]

    best_sentiment = sentiment_result["labels"][0]
    best_sentiment_score = sentiment_result["scores"][0]

    type_table = build_table(type_result["labels"], type_result["scores"])
    sentiment_table = build_table(sentiment_result["labels"], sentiment_result["scores"])

    summary = _format_summary(best_type, best_type_score, best_sentiment, best_sentiment_score)

    return {
        "summary": summary,
        "type_rows": type_table,
        "sentiment_rows": sentiment_table,
    }

def analyze_comment(comment_text: str, review_url: str):
    """Main handler called from the UI.

    Rules:
      - If both fields are provided, prefer the pasted text (URL is fetched for preview only).
      - If only URL is provided, attempt to fetch the comment body.
      - Validate size and emit structured outputs.
    """
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

def _clear():
    """Reset inputs and outputs to a clean state."""
    return "", "", "", [], [], "", ""


theme = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")

with gr.Blocks(title="Code Review Quality Analyzer", theme=theme) as demo:
    gr.Markdown(
        "# Code Review Quality Analyzer\n"
        "Classify a code review comment by feedback type and sentiment.\n\n"
        "- Runs on CPU (no GPU needed) using zero-shot classification.\n"
        f"- Model: `{PIPELINE_MODEL_ID}` · Categories are configurable."
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            with gr.Tabs():
                with gr.TabItem("Paste Comment"):
                    comment_input = gr.Textbox(
                        label="Review Comment Text",
                        placeholder="Paste a single review comment...",
                        lines=8,
                        autofocus=True,
                    )
                with gr.TabItem("GitHub URL"):
                    url_input = gr.Textbox(
                        label="Public GitHub PR Comment URL",
                        placeholder="https://github.com/org/repo/pull/123#discussion_r456",
                        lines=2,
                        info="Works for #discussion_r<ID> and #issuecomment-<ID> on public repos.",
                    )

            gr.Markdown("### Examples")
            gr.Examples(
                examples=[
                    [
                        "This will break when `user` is None. Consider checking for None before calling `get_id()`.",
                        "",
                    ],
                    [
                        "Nice cleanup here — this reads much better now. Thanks!",
                        "",
                    ],
                    [
                        "Nit: rename `x` to something more descriptive like `retry_interval`.",
                        "",
                    ],
                    [
                        "Why do we need this extra flag? Doesn't the existing `bar` already handle that case?",
                        "",
                    ],
                    [
                        "Consider extracting this logic into a helper function to avoid duplication across handlers.",
                        "",
                    ],
                    [
                        "This is a risky approach; I recommend reverting and discussing alternatives.",
                        "",
                    ],
                ],
                inputs=[comment_input, url_input],
                run_on_click=False,
            )

            with gr.Row():
                analyze_button = gr.Button("Analyze Review", variant="primary")
                clear_button = gr.Button("Clear")

        with gr.Column(scale=1):
            summary_output = gr.Markdown(label="Classification Summary")
            with gr.Row():
                type_output = gr.Dataframe(
                    column_names=["Label", "Confidence"],
                    label="Feedback Type Confidence",
                    datatype=["str", "str"],
                    interactive=False,
                    row_count=(0, "dynamic"),
                    col_count=(2, "fixed"),
                    value=[],
                )
                sentiment_output = gr.Dataframe(
                    column_names=["Label", "Confidence"],
                    label="Sentiment Confidence",
                    datatype=["str", "str"],
                    interactive=False,
                    row_count=(0, "dynamic"),
                    col_count=(2, "fixed"),
                    value=[],
                )
            with gr.Accordion("Preview", open=False):
                preview_output = gr.Textbox(label="Analyzed Comment", lines=6)
                fetched_preview_output = gr.Textbox(label="Fetched GitHub Comment", lines=6)

            with gr.Accordion("Tips", open=False):
                gr.Markdown(
                    "- Use concise, single-comment inputs for best results.\n"
                    "- For organization-wide insights, aggregate predictions across many comments.\n"
                    "- Replace the zero-shot model with a fine-tuned one for higher accuracy on your data."
                )

    analyze_button.click(
        analyze_comment,
        inputs=[comment_input, url_input],
        outputs=[summary_output, type_output, sentiment_output, preview_output, fetched_preview_output],
    )
    clear_button.click(
        _clear,
        inputs=None,
        outputs=[comment_input, url_input, summary_output, type_output, sentiment_output, preview_output, fetched_preview_output],
    )

if __name__ == "__main__":
    demo.queue(max_size=16).launch()
