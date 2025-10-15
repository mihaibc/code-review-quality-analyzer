---
title: Code Review Quality Analyzer
emoji: 🧠
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
---

# Code Review Quality Analyzer

A Hugging Face Space (Gradio) app that classifies individual code review comments to help engineering leaders understand the quality of their review culture.

## What It Does
- Accepts either pasted review comment text **or** a public GitHub pull request comment URL.
- Classifies the comment into one of five feedback types: Logic/Bug, Suggestion, Style/Nitpick, Question, Praise.
- Labels the overall sentiment as Positive, Neutral, or Negative.
- Runs entirely on CPU using the open-source `facebook/bart-large-mnli` zero-shot classifier.

## Quickstart (Local)
1. Create and activate a Python 3.9+ virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the Gradio app:
   ```bash
   python app.py
   ```
4. Open the local Gradio URL printed in the terminal and submit a comment or GitHub link.

### Optional environment variables
- `GITHUB_TOKEN` – supply a personal access token to increase GitHub rate limits when fetching comments via URL. This can be set locally or by adding a Space secret on Hugging Face.

## Deploying on Hugging Face Spaces
1. Create a new **Gradio** Space and select **CPU Basic** (no GPU needed).
2. Upload `app.py`, `requirements.txt`, and `huggingface.yml` to the Space repository.
3. Set the Space to auto-run; Gradio will launch the app automatically.
4. (Optional) Add a Hugging Face Space secret named `GITHUB_TOKEN` so the app can make authenticated API calls and avoid rate limits when fetching comments by URL.
5. (Optional) Enable outbound network access if you want to fetch comments directly from GitHub links. Without it, users should paste the comment text manually.

## Notes and Limitations
- GitHub URL support currently works for `#discussion_r<id>` (review comment) and `#issuecomment-<id>` fragments. Other comment types fall back to manual input.
- The zero-shot model is not fine-tuned on code review data; it provides a reasonable starting point that you can later replace with a custom fine-tuned model.
- For very long comments (>4,000 characters) the app asks users to shorten or summarize before analysis.

## Next Steps
- Swap in a custom fine-tuned classifier trained on your curated review dataset.
- Track reviewer trends by capturing predictions and aggregating over time.
- Extend URL support to additional platforms (e.g., GitLab, Bitbucket).
