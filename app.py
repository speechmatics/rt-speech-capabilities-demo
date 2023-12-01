import asyncio
import logging
import os
from functools import partial

import gradio as gr
import speechmatics
from file_streaming import DelayedRealtimeStream, get_audio_stream_proc
from openai import AsyncOpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger()
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.INFO)

# Setup OAI client
OPENAI_CLIENT = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Setup SM client
SM_SETTINGS = speechmatics.models.AudioSettings(sample_rate=48000, chunk_size=1024, encoding="pcm_s16le")
SM_CONFIG = speechmatics.models.TranscriptionConfig(
    language="en",
    operating_point="enhanced",
    max_delay=5,
    enable_partials=True,
    enable_entities=True,
)
SM_WS_CLIENT = speechmatics.client.WebsocketClient(
    speechmatics.models.ConnectionSettings(
        url="wss://eu2.rt.speechmatics.com/v2",
        auth_token=os.environ["SM_API_KEY"],
    )
)


def update_transcript(msg):
    global transcript
    current_transcript = msg["metadata"]["transcript"].strip()
    if len(current_transcript) > 0:
        transcript += current_transcript + " "


def update_partial(msg):
    global partial_transcript
    partial_transcript = msg["metadata"]["transcript"].strip()


SM_WS_CLIENT.add_event_handler(speechmatics.models.ServerMessageType.AddTranscript, event_handler=update_transcript)
SM_WS_CLIENT.add_event_handler(speechmatics.models.ServerMessageType.AddPartialTranscript, event_handler=update_partial)

# Prompts for OAI ChatCompletion API
SENTIMENT_PROMPT = (
    "Analyze the sentiment of this conversation. "
    "The score should be between 1 to 10, where 1 is very negative, 5 is neutral, and 10 is very positive. "
    "The score should just be a single integer, DO NOT output any additional text.\n"
    "Text: {text}\n"
    "Sentiment:"
)
ENTITIES_PROMPT = (
    "Extract all named entities in this conversation. A named entity can be a person, product, organization, "
    "place, event, time. However, a product or event need not have a proper name, it can be a generic reference. "
    "Only respond with a comma-separated list of entities.\n"
    "Text: {text}\n"
    "Entities:"
)
SUMMARY_PROMPT = "Write a concise summary of this conversation in bullet points.\nText: {text}\nSummary:"

# Global vars to store transcript and partials
MIN_CONTEXT_WORDS = 10
transcript, prev_transcript, partial_transcript, summary = "", "", "", ""
sentiment = 5
entities = set()


###
# Methods for transcribing using SM RT API
###


async def transcribe_file(filepath_obj):
    filepath = filepath_obj["name"]
    LOGGER.info(f"Transcribing File: {filepath}")
    factory = partial(get_audio_stream_proc, filepath=filepath)
    with DelayedRealtimeStream(factory) as audio_source:
        await SM_WS_CLIENT.run(audio_source, SM_CONFIG, SM_SETTINGS)
    LOGGER.info("Done")


###
# Methods for processing text using OpenAI ChatCompletion API
###


async def call_chat_api(prompt):
    response = await OPENAI_CLIENT.chat.completions.create(
        messages=[{"role": "user", "content": prompt}], model="gpt-4-1106-preview", max_tokens=300
    )
    return response.choices[0].message.content


async def get_capabilities():
    return await asyncio.gather(
        call_chat_api(SUMMARY_PROMPT.format(text=transcript)),
        call_chat_api(SENTIMENT_PROMPT.format(text=transcript)),
        call_chat_api(ENTITIES_PROMPT.format(text=transcript)),
    )


def get_response():
    global transcript
    global prev_transcript
    global summary
    global sentiment
    global entities
    # Calculate difference in length between latest transcript and previous transcript
    new_context_len = len(transcript.split()) - len(prev_transcript.split())
    # Only call the OpenAI API if the difference is more than MIN_CONTEXT_WORDS
    if new_context_len > MIN_CONTEXT_WORDS:
        prev_transcript = transcript
        LOGGER.info(f"Transcript: {transcript}")
        summary, sentiment, entities = asyncio.run(get_capabilities())
    return summary, sentiment, entities


###
# Methods to display / handle responses in Gradio
###


def clear_responses():
    global transcript
    global partial_transcript
    global prev_transcript
    global summary
    global sentiment
    global entities
    transcript, prev_transcript, partial_transcript, summary = "", "", "", ""
    sentiment = 5
    entities = set()
    LOGGER.info("Cleared responses")


def convert_sentiment_to_color(sentiment):
    color_map = [
        "#f80000",
        "#f83e3e",
        "#f87c7c",
        "#f8baba",
        "#f8f8f8",
        "#f8f8f8",
        "#baf8ba",
        "#7cf87c",
        "#3ef83e",
        "#00f800",
    ]
    return f'<div class="sentiment_color" style="background-color:{color_map[sentiment-1]};"></div>'


def display_transcript():
    global transcript
    global partial_transcript
    return f"<div>{transcript} <span style='font-style: italic; opacity: 0.5;'>{partial_transcript}</span></div>"


def display_response():
    summary, sentiment, entities = get_response()
    return summary.replace("\n", "<br/>"), convert_sentiment_to_color(int(sentiment)), entities


def launch_app():
    global transcript
    global summary
    with gr.Blocks(
        title="RT Speech Capabilities",
        theme=gr.themes.Default(text_size=gr.themes.sizes.text_lg),
        css=".transcript, .summary, .entities {padding: 10px; overflow:auto; font-size: 20px !important;} "
        ".transcript {height: 600px;} .transcript .generating.svelte-zlszon.svelte-zlszon {height:600px;} "
        ".summary {height: 400px;} .summary .generating.svelte-zlszon.svelte-zlszon {height:400px;} "
        ".sentiment {height: 50px;} .sentiment .generating.svelte-zlszon.svelte-zlszon {display:None;} "
        ".entities {height: 150px;} .entities .generating.svelte-zlszon.svelte-zlszon {height:150px;} "
        ".sentiment_color {height: 50px; width: 100%;}",
    ) as demo:
        gr.Markdown(
            "# RT Speech Capabilities\n"
            "### Transcribe with Speechmatics RT API and obtain a summary, sentiment and list of entities "
            "in real-time! Upload an audio file or use the microphone to begin."
        )
        with gr.Row():
            with gr.Column():
                with gr.Tab("Audio File"):
                    file_audio = gr.Audio(source="upload", type="filepath", format="mp3")
                    gr.Examples(
                        examples=["example.mp3"],
                        inputs=[file_audio],
                        outputs=[],
                        fn=lambda _: None,
                        label="Examples",
                        cache_examples=True,
                    )
                gr.Markdown("### Transcript")
                transcript_box = gr.HTML(value=transcript, elem_classes="transcript")
            with gr.Column():
                gr.Markdown("### Summary")
                summary_box = gr.HTML(value=summary, elem_classes="summary")
                gr.Markdown("### Sentiment")
                sentiment_box = gr.HTML(value=sentiment, elem_classes="sentiment")
                gr.Markdown("### Entities")
                entities_box = gr.HTML(value=entities, elem_classes="entities")

        # Call `transcribe_file` when the file streams
        file_audio.play(transcribe_file, inputs=[file_audio], preprocess=False)
        # Watch for changes in transcript and call summarize
        transcript_box.change(display_transcript, inputs=None, outputs=[transcript_box], every=1)
        summary_box.change(display_response, inputs=None, outputs=[summary_box, sentiment_box, entities_box], every=1)
        # Clear outputs before running file
        file_audio.change(clear_responses)
        # Trigger a change in all outputs to start watching
        demo.load(
            lambda: ("a", "a", "a", "a"),
            inputs=None,
            outputs=[transcript_box, summary_box, sentiment_box, entities_box],
        )
    return demo


if __name__ == "__main__":
    demo = launch_app()
    demo.queue().launch(server_name="0.0.0.0")
