# RT Speech Capabilities Demo

A Gradio web app demoing RT Speech Capabilities, using Speechmatics RT API and OpenAI ChatCompletion API.

## Getting Started

Install all the required Python dependencies with:

```
pip3 install -r requirements.txt
```

Define your Speechmatics and OpenAI API keys:

```bash
export SM_API_KEY=XYZ
export OPENAI_API_KEY=ABC
```

You will also need to install [FFmpeg](https://ffmpeg.org/download.html).

## Running

Start the app with

```bash
python3 -m app
```

and navigate to http://localhost:7860/ to view the app.

## Docs

The Speechmatics Python SDK and CLI is documented at https://speechmatics.github.io/speechmatics-python.

The Speechmatics API and product documentation can be found at https://docs.speechmatics.com.
