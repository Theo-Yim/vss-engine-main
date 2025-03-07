################################################################################
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#
################################################################################

import json
import tempfile
import uuid
from logging import Logger

import aiohttp
import gradio as gr
import pkg_resources
import yaml

pipeline_args = None
enable_logs = True
logger: Logger = None
appConfig = {}


STANDALONE_MODE = False

dummy_mr = """
#### just to create the space
"""
USER_AVATAR_ICON = tempfile.NamedTemporaryFile()
USER_AVATAR_ICON.write(
    pkg_resources.resource_string("__main__", "client/assets/user-icon-60px.png")
)
USER_AVATAR_ICON.flush()
CHATBOT_AVATAR_ICON = tempfile.NamedTemporaryFile()
CHATBOT_AVATAR_ICON.write(
    pkg_resources.resource_string("__main__", "client/assets/chatbot-icon-60px.png")
)
CHATBOT_AVATAR_ICON.flush()


def get_default_prompts():
    try:
        with open("/opt/nvidia/via/default_config.yaml") as f:
            ca_rag_config = yaml.safe_load(f)
            prompts = ca_rag_config["summarization"]["prompts"]
            return (
                prompts["caption"],
                prompts["caption_summarization"],
                prompts["summary_aggregation"],
                "/opt/nvidia/via/warehouse_graph_rag_config.yaml",
            )
    except Exception:
        return "", "", "", None


async def summarize_response_async(session: aiohttp.ClientSession, req_json, video_id, enable_chat):
    async with session.post(appConfig["backend"] + "/summarize", json=req_json) as resp:
        if resp.status >= 400:
            raise gr.Error((await resp.json())["message"])

        yield (
            gr.update(interactive=False),  # video, , , , ,
            gr.update(interactive=False),  # username, , , , ,
            gr.update(interactive=False),  # password, , , , ,
            gr.update(interactive=False),  # upload_button
            gr.update(interactive=False),  # enable_chat
            gr.update(interactive=enable_chat),  # ask_button
            gr.update(interactive=enable_chat),  # question_textbox
            video_id,  # stream_id
            "Waiting for first summary...",  # output_response
            "Waiting for alerts...",  # output_alerts
            gr.update(interactive=False),  # chunk_size
            gr.update(interactive=False),  # summary_duration
            gr.update(interactive=False),  # alerts
            gr.update(interactive=False),  # summary_prompt
            gr.update(interactive=False),  # caption_summarization_prompt
            gr.update(interactive=False),  # summary_aggregation_prompt
            gr.update(interactive=False),  # temperature
            gr.update(interactive=False),  # top_p
            gr.update(interactive=False),  # top_k
            gr.update(interactive=False),  # max_new_tokens
            gr.update(interactive=False),  # seed
            gr.update(interactive=False),  # num_frames_per_chunk
            gr.update(interactive=False),  # vlm_input_width
            gr.update(interactive=False),  # vlm_input_height
            gr.update(interactive=False),  # active_live_streams
            gr.update(interactive=False),  # refresh_list_button
            gr.update(interactive=False),  # reconnect_button
        )
        past_summaries = []
        past_alerts = []
        have_eos = False

        def get_output_string(header, items):
            return "\n--------\n".join([header] + items + [header]) if items else header

        while True:
            line = await resp.content.readline()
            if not line:
                if have_eos:
                    output_summaries = get_output_string("Live Stream Ended", past_summaries)
                    output_alerts = get_output_string("Live Stream Ended", past_alerts)
                else:
                    output_summaries = get_output_string(
                        "Disconnected from server. Reconnect to get latest summaries",
                        past_summaries,
                    )
                    output_alerts = get_output_string(
                        "Disconnected from server. Reconnect to get alerts", past_alerts
                    )
                break
            line = line.decode("utf-8")
            if not line.startswith("data: "):
                yield [gr.update()] * 27
                continue

            data = line.strip()[6:]

            if data == "[DONE]":
                output_summaries = get_output_string("Live Stream Ended", past_summaries)
                output_alerts = get_output_string("Live Stream Ended", past_alerts)
                break

            try:
                response = json.loads(data)
                if response["choices"][0]["finish_reason"] == "stop":
                    response_str_current = (
                        f'{response["media_info"]["start_timestamp"]}'
                        f' -> {response["media_info"]["end_timestamp"]}\n\n'
                        f'{response["choices"][0]["message"]["content"]}'
                    )
                    past_summaries = (
                        past_summaries[int(len(past_summaries) / 9) :] + [response_str_current]
                        if response_str_current
                        else []
                    )
                if response["choices"][0]["finish_reason"] == "tool_calls":
                    alert = response["choices"][0]["message"]["tool_calls"][0]["alert"]
                    alert_str = (
                        f"Alert Name: {alert['name']}\n"
                        f"Detected Events: {', '.join(alert['detectedEvents'])}\n"
                        f"NTP Time: {alert['ntpTimestamp']}\n"
                        f"Details: {alert['details']}\n"
                    )
                    past_alerts = past_alerts[int(len(past_alerts) / 99) :] + (
                        [alert_str] if alert_str else []
                    )
            except Exception:
                pass

            output_summaries = get_output_string(
                "Waiting for next summary..." if past_summaries else "Waiting for first summary...",
                past_summaries,
            )
            output_alerts = get_output_string(
                "Waiting for new alerts..." if past_alerts else "Waiting for alerts", past_alerts
            )

            yield (
                gr.update(interactive=False),  # video, , , , ,
                gr.update(interactive=False),  # username, , , , ,
                gr.update(interactive=False),  # password, , , , ,
                gr.update(interactive=False),  # upload_button
                gr.update(interactive=False),  # enable_chat
                gr.update(interactive=enable_chat),  # ask_button
                gr.update(interactive=enable_chat),  # question_textbox
                video_id,  # stream_id
                output_summaries,  # output_response
                output_alerts,  # output_alerts
                gr.update(interactive=False),  # chunk_size
                gr.update(interactive=False),  # summary_duration
                gr.update(interactive=False),  # alerts
                gr.update(interactive=False),  # summary_prompt
                gr.update(interactive=False),  # caption_summarization_prompt
                gr.update(interactive=False),  # summary_aggregation_prompt
                gr.update(interactive=False),  # temperature
                gr.update(interactive=False),  # top_p
                gr.update(interactive=False),  # top_k
                gr.update(interactive=False),  # max_new_tokens
                gr.update(interactive=False),  # seed
                gr.update(interactive=False),  # num_frames_per_chunk
                gr.update(interactive=False),  # vlm_input_width
                gr.update(interactive=False),  # vlm_input_height
                gr.update(interactive=False),  # active_live_streams
                gr.update(interactive=False),  # refresh_list_button
                gr.update(interactive=False),  # reconnect_button
            )

    yield (
        gr.update(interactive=True),  # video, , , , ,
        gr.update(interactive=True),  # username, , , , ,
        gr.update(interactive=True),  # password, , , , ,
        gr.update(interactive=True),  # upload_button
        gr.update(interactive=True),  # enable_chat
        gr.update(interactive=enable_chat),  # ask_button
        gr.update(interactive=enable_chat),  # question_textbox
        video_id,  # stream_id
        output_summaries,  # output_response
        output_alerts,  # output_alerts
        gr.update(interactive=True),  # chunk_size
        gr.update(interactive=True),  # summary_duration
        gr.update(interactive=True),  # alerts
        gr.update(interactive=True),  # summary_prompt
        gr.update(interactive=True),  # caption_summarization_prompt
        gr.update(interactive=True),  # summary_aggregation_prompt
        gr.update(interactive=True),  # temperature
        gr.update(interactive=True),  # top_p
        gr.update(interactive=True),  # top_k
        gr.update(interactive=True),  # max_new_tokens
        gr.update(interactive=True),  # seed
        gr.update(interactive=True),  # num_frames_per_chunk
        gr.update(interactive=True),  # vlm_input_width
        gr.update(interactive=True),  # vlm_input_height
        gr.update(
            interactive=False, choices=[], value="< Click Refresh List to fetch active streams >"
        ),  # active_live_streams
        gr.update(interactive=True),  # refresh_list_button
        gr.update(interactive=True),  # reconnect_button
    )


async def gradio_reset(stream_id, request: gr.Request):
    logger.info(f"gradio_reset. ip: {request.client.host}")

    if stream_id:
        session: aiohttp.ClientSession = appConfig["session"]
        async with session.delete(appConfig["backend"] + "/live-stream/" + stream_id):
            pass

    return (
        gr.update(value=None, interactive=True),  # video,
        gr.update(value=None, interactive=True),  # username, , , , ,
        gr.update(value=None, interactive=True),  # password, , , , ,
        gr.update(interactive=False),  # upload_button,
        gr.update(interactive=True),  # enable_chat
        gr.update(interactive=False),  # ask_button
        gr.update(interactive=False),  # reset_chat_button
        gr.update(interactive=False),  # question_textbox
        "",  # stream_id,
        None,  # output_response,
        None,  # output_alerts
        [],  # chatbot,
        gr.update(interactive=True),  # chunk_size,
        gr.update(interactive=True),  # summary_duration,
        gr.update(interactive=True),  # alerts,
        gr.update(interactive=True),  # summary_prompt,
        gr.update(interactive=True),  # caption_summarization_prompt,
        gr.update(interactive=True),  # summary_aggregation_prompt,
        gr.update(interactive=True),  # temperature,
        gr.update(interactive=True),  # top_p,
        gr.update(interactive=True),  # top_k,
        gr.update(interactive=True),  # max_new_tokens,
        gr.update(interactive=True),  # seed,
        gr.update(interactive=True),  # num_frames_per_chunk
        gr.update(interactive=True),  # vlm_input_width
        gr.update(interactive=True),  # vlm_input_height
        await refresh_active_stream_list(),  # active_live_streams,
        gr.update(interactive=True),  # refresh_list_button,
        gr.update(interactive=False),  # reconnect_button
    )


async def reset_chat(chatbot):
    # Reset all UI components to their initial state
    chatbot = []
    yield (chatbot)
    return


async def ask_question(
    question_textbox,
    ask_button,
    reset_chat_button,
    video,
    chatbot,
    media_ids,
    chunk_size,
    temperature,
    seed,
    num_frames_per_chunk,
    vlm_input_width,
    vlm_input_height,
    max_new_tokens,
    top_p,
    top_k,
):
    logger.debug(f"Question: {question_textbox}")
    session: aiohttp.ClientSession = appConfig["session"]
    # ask_button.interactive = False
    question = question_textbox
    video_id = media_ids
    reset_chat_triggered = True
    if question != "/clear":
        reset_chat_triggered = False
        chatbot = chatbot + [["<b>" + str(question) + " </b>", None]]

    yield chatbot, gr.update(), gr.update(), gr.update(value="", interactive=False)
    async with session.get(appConfig["backend"] + "/models") as resp:
        resp_json = await resp.json()
        if resp.status >= 400:
            chatbot = [[None, "<b>Error: </b><i>" + resp_json["message"] + "</i>"]]
            yield chatbot, gr.update(interactive=True), gr.update(interactive=True), gr.update(
                interactive=True
            )
            return

        model = resp_json["data"][0]["id"]

    req_json = {
        "id": video_id,
        "model": model,
        "chunk_duration": chunk_size,
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_new_tokens,
        "top_p": top_p,
        "top_k": top_k,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    # Not passing VLM specific params like num_frames_per_chunk, vlm_input_width
    req_json["messages"] = [{"content": str(question), "role": "user"}]
    session: aiohttp.ClientSession = appConfig["session"]
    async with session.post(appConfig["backend"] + "/chat/completions", json=req_json) as resp:
        if resp.status >= 400:
            resp_json = await resp.json()
            chatbot = []
            chatbot = chatbot + [[None, "<b>Error: </b><i>" + resp_json["message"] + "</i>"]]
            yield chatbot, gr.update(interactive=True), gr.update(interactive=True), gr.update(
                interactive=True
            )
            return
        response = await resp.text()
        logger.debug(f"response is {str(response)}")
        accumulated_responses = []
        lines = response.splitlines()
        for line in lines:
            data = line.strip()
            response = json.loads(data)
            if response["choices"]:
                accumulated_responses.append(response)
            if response["usage"]:
                usage = response["usage"]

        logger.debug(f"accumulated_responses: {accumulated_responses} usage: {usage}")

        if len(accumulated_responses) == 1:
            response_str = accumulated_responses[0]["choices"][0]["message"]["content"]
        else:
            response_str = ""
        if question != "/clear":
            chatbot = chatbot + [[None, response_str]]

        yield chatbot, gr.update(interactive=True), gr.update(
            interactive=not reset_chat_triggered
        ), gr.update(interactive=True)
        return


async def add_rtsp_stream(
    video,
    username,
    password,
    temperature,
    top_p,
    top_k,
    max_new_tokens,
    seed,
    num_frames_per_chunk,
    vlm_input_width,
    vlm_input_height,
    chunk_size,
    summary_duration,
    alerts,
    summary_prompt,
    caption_summarization_prompt,
    summary_aggregation_prompt,
    enable_chat,
    request: gr.Request,
):
    logger.info(f"upload_imgorvideo. ip: {request.client.host}")
    if not video:
        yield [
            gr.update(),
        ] * 27
        return
    elif video:
        video_id = ""
        try:
            session: aiohttp.ClientSession = appConfig["session"]
            async with session.get(appConfig["backend"] + "/models") as resp:
                resp_json = await resp.json()
                if resp.status >= 400:
                    raise gr.Error(resp_json["message"])
                model = resp_json["data"][0]["id"]

            req_json = {
                "liveStreamUrl": video.strip(),
                "username": username,
                "password": password,
                "description": "Added from Gradio UI",
            }
            async with session.post(appConfig["backend"] + "/live-stream", json=req_json) as resp:
                resp_json = await resp.json()
                if resp.status != 200:
                    raise gr.Error(resp_json["message"].replace("\\'", "'"))
                video_id = resp_json["id"]

            req_json = {
                "id": video_id,
                "model": model,
                "chunk_duration": chunk_size,
                "summary_duration": summary_duration,
                "temperature": temperature,
                "seed": seed,
                "max_tokens": max_new_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "stream": True,
                "enable_chat": enable_chat,
                "stream_options": {"include_usage": True},
                "num_frames_per_chunk": num_frames_per_chunk,
                "vlm_input_width": vlm_input_width,
                "vlm_input_height": vlm_input_height,
            }
            if summary_prompt:
                req_json["prompt"] = summary_prompt
            if caption_summarization_prompt:
                req_json["caption_summarization_prompt"] = caption_summarization_prompt
            if summary_aggregation_prompt:
                req_json["summary_aggregation_prompt"] = summary_aggregation_prompt

            parsed_alerts = []
            for alert in alerts.split(";"):
                alert = alert.strip()
                if not alert:
                    continue
                try:
                    alert_name, events = [word.strip() for word in alert.split(":")]
                    assert alert_name
                    assert events

                    parsed_events = [ev.strip() for ev in events.split(",") if ev.strip()]
                    assert parsed_events
                except Exception:
                    raise gr.Error(f"Failed to parse alert '{alert}'") from None
                parsed_alerts.append(
                    {
                        "type": "alert",
                        "alert": {"name": alert_name, "events": parsed_events},
                    }
                )
            if parsed_alerts:
                req_json["tools"] = parsed_alerts

            async for response in summarize_response_async(
                session, req_json, video_id, enable_chat
            ):
                yield response

        except Exception as ex:
            yield (
                gr.update(interactive=True),  # video, , , , ,
                gr.update(interactive=True),  # username, , , , ,
                gr.update(interactive=True),  # password, , , , ,
                gr.update(interactive=True),  # upload_button
                gr.update(interactive=True),  # enable_chat
                gr.update(interactive=enable_chat),  # ask_button
                gr.update(interactive=enable_chat),  # question_textbox
                video_id,  # stream_id
                "ERROR: " + ex.args[0],  # output_response
                "ERROR: " + ex.args[0],  # output_alerts
                gr.update(interactive=True),  # chunk_size
                gr.update(interactive=True),  # summary_duration
                gr.update(interactive=True),  # alerts
                gr.update(interactive=True),  # summary_prompt
                gr.update(interactive=True),  # caption_summarization_prompt
                gr.update(interactive=True),  # summary_aggregation_prompt
                gr.update(interactive=True),  # temperature
                gr.update(interactive=True),  # top_p
                gr.update(interactive=True),  # top_k
                gr.update(interactive=True),  # max_new_tokens
                gr.update(interactive=True),  # seed
                gr.update(interactive=True),  # num_frames_per_chunk
                gr.update(interactive=True),  # vlm_input_width
                gr.update(interactive=True),  # vlm_input_height
                gr.update(interactive=True),  # active_live_streams
                gr.update(interactive=True),  # refresh_list_button
                gr.update(interactive=True),  # reconnect_button
            )
    else:
        raise gr.Error("Only a single input is supported")


CHUNK_SIZES = [
    ("10 sec", 10),
    ("20 sec", 20),
    ("30 sec", 30),
    ("1 min", 60),
    ("2 min", 120),
    ("5 min", 300),
]

SUMMARY_DURATION = [
    ("1 min", 60),
    ("2 min", 120),
    ("5 min", 300),
    ("10 min", 600),
    ("30 min", 1800),
    ("1 hr", 3600),
]


async def refresh_active_stream_list():
    async with aiohttp.ClientSession() as session:
        async with session.get(appConfig["backend"] + "/live-stream") as resp:
            if resp.status >= 400:
                raise gr.Error(resp.json()["message"])
            resp_json = await resp.json()
            choices = [
                (f"{ls['liveStreamUrl']} ({ls['description']})", ls["id"])
                for ls in resp_json
                if ls["chunk_duration"] > 0
            ]
            return gr.update(
                choices=choices,
                value=(
                    f"< {len(choices)} active stream(s). Select an active stream >"
                    if choices
                    else "< No active streams found >"
                ),
                interactive=bool(choices),
            )


async def on_url_changed(video):
    return gr.update(interactive=bool(video))


async def reconnect_live_stream(video_id, enable_chat):
    if not video_id:
        yield [
            gr.update(),
        ] * 27
        return

    session: aiohttp.ClientSession = appConfig["session"]
    async with session.get(appConfig["backend"] + "/models") as resp:
        resp_json = await resp.json()
        if resp.status >= 400:
            raise gr.Error(resp_json["message"])
        model = resp_json["data"][0]["id"]

    req_json = {
        "id": video_id,
        "model": model,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    try:
        async for response in summarize_response_async(session, req_json, video_id, enable_chat):
            yield response
    except Exception as ex:
        yield (
            gr.update(interactive=False),  # video, , , , ,
            gr.update(interactive=False),  # username, , , , ,
            gr.update(interactive=False),  # password, , , , ,
            gr.update(interactive=False),  # upload_button
            gr.update(interactive=False),  # enable_chat
            gr.update(interactive=enable_chat),  # ask_button
            gr.update(interactive=enable_chat),  # question_textbox
            video_id,  # stream_id
            "ERROR: " + str(ex),  # output_response
            "ERROR: " + str(ex),  # output_alerts
            gr.update(interactive=False),  # chunk_size
            gr.update(interactive=False),  # summary_duration
            gr.update(interactive=False),  # alerts
            gr.update(interactive=False),  # summary_prompt
            gr.update(interactive=False),  # caption_summarization_prompt
            gr.update(interactive=False),  # summary_aggregation_prompt
            gr.update(interactive=False),  # temperature
            gr.update(interactive=False),  # top_p
            gr.update(interactive=False),  # top_k
            gr.update(interactive=False),  # max_new_tokens
            gr.update(interactive=False),  # seed
            gr.update(interactive=False),  # num_frames_per_chunk
            gr.update(interactive=False),  # vlm_input_width
            gr.update(interactive=False),  # vlm_input_height
            gr.update(interactive=False),  # active_live_streams
            gr.update(interactive=False),  # refresh_list_button
            gr.update(interactive=False),  # reconnect_button
        )


def live_stream_selected(active_live_stream):
    try:
        uuid.UUID(active_live_stream)
        return gr.update(interactive=True)
    except Exception:
        pass
    return gr.update(interactive=False)


def build_rtsp_stream(args, app_cfg, logger_):
    global appConfig, logger, pipeline_args
    appConfig = app_cfg
    logger = logger_
    pipeline_args = args

    (
        default_prompt,
        default_caption_summarization_prompt,
        default_summary_aggregation_prompt,
        _,
    ) = get_default_prompts()

    # need it here for example
    show_configuration = gr.Button(
        "Show parameters", variant="primary", size="sm", render=False, elem_classes="gray-btn"
    )
    stream_id = gr.State("")
    with gr.Row(equal_height=True):

        with gr.Column(scale=1):
            video = gr.Textbox(
                label="STREAM",
                placeholder="rtsp://",
                interactive=True,
                visible=True,
                container=True,
                elem_classes="white-background",
            )
            with gr.Accordion(
                label="RTSP Credentials",
                elem_classes="white-background",
                open=False,
            ):
                username = gr.Textbox(
                    label="RTSP Username",
                    placeholder="(OPTIONAL)",
                    interactive=True,
                    visible=True,
                    container=True,
                    elem_classes="white-background",
                )
                password = gr.Textbox(
                    label="RTSP Password",
                    placeholder="(OPTIONAL)",
                    interactive=True,
                    visible=True,
                    container=True,
                    elem_classes="white-background",
                )

            chunk_size = gr.Dropdown(
                choices=CHUNK_SIZES,
                label="CHUNK SIZE",
                value=10,
                interactive=True,
                visible=True,
                elem_classes=["white-background", "bold-header"],
            )
            summary_duration = gr.Dropdown(
                choices=SUMMARY_DURATION,
                label="SUMMARY DURATION",
                value=60,
                interactive=True,
                visible=True,
                elem_classes=["white-background", "bold-header"],
            )
            alerts = gr.TextArea(
                label="ALERTS (OPTIONAL)",
                placeholder="<alert1_name>:<event1>,<event2>;<alert2_name>:<event3>,<event4>;",
                elem_classes=["white-background", "bold-header"],
                lines=3,
                max_lines=3,
                value="",
            )
            summary_prompt = gr.TextArea(
                label="PROMPT (OPTIONAL)",
                elem_classes=["white-background", "bold-header"],
                lines=3,
                max_lines=3,
                value=default_prompt,
            )
            caption_summarization_prompt = gr.TextArea(
                label="CAPTION SUMMARIZATION PROMPT (OPTIONAL)",
                elem_classes=["white-background", "bold-header"],
                lines=3,
                max_lines=3,
                value=default_caption_summarization_prompt,
            )
            summary_aggregation_prompt = gr.TextArea(
                label="SUMMARY AGGREGATION PROMPT (OPTIONAL)",
                elem_classes=["white-background", "bold-header"],
                lines=3,
                max_lines=3,
                value=default_summary_aggregation_prompt,
            )

            with gr.Row(equal_height=True):
                gr.Markdown(dummy_mr, visible=True)
                enable_chat = gr.Checkbox(value=False, label="Enable chat for the stream")
                upload_button = gr.Button(
                    value="Start streaming & summarization",
                    interactive=False,
                    variant="primary",
                    size="sm",
                    scale=1,
                )

            gr.Markdown("##")
            active_live_streams = gr.Dropdown(
                label="ACTIVE LIVE STREAMS",
                interactive=False,
                visible=True,
                elem_classes=["white-background", "bold-header"],
                allow_custom_value=True,
                value=lambda: "< Please Refresh List >",
                # choices= refresh_active_stream_list,
            )
            with gr.Row(equal_height=True):
                refresh_list_button = gr.Button(
                    value="Refresh List", interactive=True, variant="primary", size="sm", scale=0
                )
                reconnect_button = gr.Button(
                    value="Reconnect", interactive=False, variant="primary", size="sm", scale=0
                )

        with gr.Column(scale=3):
            with gr.Column():
                with gr.Tabs(elem_id="via-tabs"):
                    with gr.Tab("SUMMARIES"):
                        output_response = gr.TextArea(
                            interactive=False, max_lines=30, lines=30, show_label=False
                        )
                    with gr.Tab("ALERTS"):
                        output_alerts = gr.TextArea(
                            interactive=False, max_lines=30, lines=30, show_label=False
                        )
                    with gr.Tab("CHAT"):
                        chatbot = gr.Chatbot(
                            [],
                            label="RESPONSE",
                            bubble_full_width=False,
                            avatar_images=(USER_AVATAR_ICON.name, CHATBOT_AVATAR_ICON.name),
                            height=550,
                            elem_classes="white-background",
                        )
                        with gr.Row(equal_height=True, variant="default"):
                            question_textbox = gr.Textbox(
                                label="Ask a question",
                                interactive=False,
                                scale=3,  # This makes it take up 3 parts of the available space
                            )
                            with gr.Column(
                                scale=1
                            ):  # This column takes up 1 part of the available space
                                ask_button = gr.Button("Ask", interactive=False)
                                reset_chat_button = gr.Button("Reset Chat", interactive=False)

            with gr.Row(equal_height=False, variant="default"):
                gr.HTML("", visible=True).scale = 1
                show_configuration.render()

                clear = gr.Button("Stop Summarization & Delete Live Stream", size="sm")
                show_configuration_state = gr.State(False)
            configuration = gr.Column(variant="compact", visible=False)

            def toggle_configuration(show_configuration_state):
                return (
                    gr.update(visible=not show_configuration_state),
                    gr.update(
                        value="Show parameters" if show_configuration_state else "Hide parameters"
                    ),
                    not show_configuration_state,
                )

            show_configuration.click(
                toggle_configuration,
                inputs=[show_configuration_state],
                outputs=[configuration, show_configuration, show_configuration_state],
                show_progress="hidden",
            )
            with configuration:
                with gr.Accordion("VLM Parameters"):
                    with gr.Row():
                        num_frames_per_chunk = gr.Number(
                            label="num_frames_per_chunk",
                            interactive=True,
                            precision=0,
                            minimum=0,
                            maximum=10,
                            value=0,
                            info=("The number of frames to choose from chunk"),
                            elem_classes="white-background",
                        )
                        vlm_input_width = gr.Number(
                            label="VLM Input Width",
                            interactive=True,
                            precision=0,
                            minimum=0,
                            maximum=4096,
                            value=0,
                            info=("Provide VLM frame's width details"),
                            elem_classes="white-background",
                        )
                        vlm_input_height = gr.Number(
                            label="VLM Input Height",
                            interactive=True,
                            precision=0,
                            minimum=0,
                            maximum=4096,
                            value=0,
                            info=("Provide VLM frame's height details"),
                            elem_classes="white-background",
                        )
                    with gr.Row():
                        temperature = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.4,
                            interactive=True,
                            label="Temperature",
                            step=0.05,
                            info=(
                                "The sampling temperature to use for text generation."
                                " The higher the temperature value is, the less deterministic"
                                " the output text will be. It is not recommended to modify both"
                                " temperature and top_p in the same call."
                            ),
                            elem_classes="white-background",
                        )
                        top_p = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=1,
                            interactive=True,
                            label="Top P",
                            step=0.05,
                            info=(
                                "The top-p sampling mass used for text generation. The top-p value"
                                " determines the probability mass that is sampled at sampling time."
                                " For example, if top_p = 0.2, only the most likely tokens"
                                " (summing to 0.2 cumulative probability) will be sampled."
                                " It is not recommended to modify both temperature and top_p"
                                " in the same call."
                            ),
                            elem_classes="white-background",
                        )
                        top_k = gr.Number(
                            label="Top K",
                            interactive=True,
                            precision=0,
                            minimum=1,
                            maximum=1000,
                            value=100,
                            info=(
                                "The number of highest probability vocabulary"
                                " tokens to keep for top-k-filtering"
                            ),
                            elem_classes="white-background",
                        )
                    with gr.Row():
                        max_new_tokens = gr.Slider(
                            minimum=1,
                            maximum=1024,
                            value=512,
                            interactive=True,
                            label="Max Tokens",
                            step=1,
                            info=(
                                "The maximum number of tokens to generate in any given call."
                                " Note that the model is not aware of this value,"
                                " and generation will simply stop at the number of"
                                " tokens specified."
                            ),
                            elem_classes="white-background",
                        )
                        seed = gr.Number(
                            label="Seed",
                            interactive=True,
                            precision=0,
                            minimum=1,
                            maximum=2**32 - 1,
                            value=1,
                            info=(
                                "Seed value to use for sampling. Repeated requests "
                                "with the same seed "
                                "and parameters should return the same result."
                            ),
                            elem_classes="white-background",
                        )

        upload_button.click(
            lambda enable_chat, request: (
                gr.update(interactive=bool(enable_chat)),
                gr.update(interactive=bool(enable_chat)),
            ),
            inputs=[ask_button, question_textbox],
            outputs=[ask_button, question_textbox],
        ).then(
            add_rtsp_stream,
            inputs=[
                video,
                username,
                password,
                temperature,
                top_p,
                top_k,
                max_new_tokens,
                seed,
                num_frames_per_chunk,
                vlm_input_width,
                vlm_input_height,
                chunk_size,
                summary_duration,
                alerts,
                summary_prompt,
                caption_summarization_prompt,
                summary_aggregation_prompt,
                enable_chat,
            ],
            outputs=[
                video,
                username,
                password,
                upload_button,
                enable_chat,
                ask_button,
                question_textbox,
                stream_id,
                output_response,
                output_alerts,
                chunk_size,
                summary_duration,
                alerts,
                summary_prompt,
                caption_summarization_prompt,
                summary_aggregation_prompt,
                temperature,
                top_p,
                top_k,
                max_new_tokens,
                seed,
                num_frames_per_chunk,
                vlm_input_width,
                vlm_input_height,
                active_live_streams,
                refresh_list_button,
                reconnect_button,
            ],
        )

        refresh_list_button.click(fn=refresh_active_stream_list, outputs=[active_live_streams])
        active_live_streams.change(
            live_stream_selected, inputs=[active_live_streams], outputs=[reconnect_button]
        )
        reconnect_button.click(
            lambda enable_chat, request: (
                gr.update(interactive=bool(enable_chat)),
                gr.update(interactive=bool(enable_chat)),
            ),
            inputs=[ask_button, question_textbox],
            outputs=[ask_button, question_textbox],
        ).then(
            fn=reconnect_live_stream,
            inputs=[active_live_streams, enable_chat],
            outputs=[
                video,
                username,
                password,
                upload_button,
                enable_chat,
                ask_button,
                question_textbox,
                stream_id,
                output_response,
                output_alerts,
                chunk_size,
                summary_duration,
                alerts,
                summary_prompt,
                caption_summarization_prompt,
                summary_aggregation_prompt,
                temperature,
                top_p,
                top_k,
                max_new_tokens,
                seed,
                num_frames_per_chunk,
                vlm_input_width,
                vlm_input_height,
                active_live_streams,
                refresh_list_button,
                reconnect_button,
            ],
        )

        ask_button.click(
            ask_question,
            inputs=[
                question_textbox,
                ask_button,
                reset_chat_button,
                video if video is not None else active_live_streams,
                chatbot,
                stream_id,
                chunk_size,
                temperature,
                seed,
                num_frames_per_chunk,
                vlm_input_width,
                vlm_input_height,
                max_new_tokens,
                top_p,
                top_k,
            ],
            outputs=[chatbot, ask_button, reset_chat_button, question_textbox],
        )

        reset_chat_button.click(
            ask_question,
            inputs=[
                gr.State("/clear"),
                ask_button,
                reset_chat_button,
                video if video is not None else active_live_streams,
                chatbot,
                stream_id,
                chunk_size,
                temperature,
                seed,
                num_frames_per_chunk,
                vlm_input_width,
                vlm_input_height,
                max_new_tokens,
                top_p,
                top_k,
            ],
            outputs=[chatbot, ask_button, reset_chat_button, question_textbox],
        ).then(
            reset_chat,
            inputs=[chatbot],
            outputs=[chatbot],
        )

        video.change(on_url_changed, inputs=[video], outputs=[upload_button])

        clear.click(
            gradio_reset,
            [stream_id],
            [
                video,
                username,
                password,
                upload_button,
                enable_chat,
                ask_button,
                reset_chat_button,
                question_textbox,
                stream_id,
                output_response,
                output_alerts,
                chatbot,
                chunk_size,
                summary_duration,
                alerts,
                summary_prompt,
                caption_summarization_prompt,
                summary_aggregation_prompt,
                temperature,
                top_p,
                top_k,
                max_new_tokens,
                seed,
                num_frames_per_chunk,
                vlm_input_width,
                vlm_input_height,
                active_live_streams,
                refresh_list_button,
                reconnect_button,
            ],
            queue=None,
        )
