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

import atexit
import json
import os
import shutil
import tempfile
from logging import Logger
from pathlib import Path

import aiohttp
import gradio as gr
import pkg_resources
import yaml

from utils import MediaFileInfo

pipeline_args = None
logger: Logger = None
appConfig = {}

DEFAULT_CHUNK_SIZE = 0
DEFAULT_VIA_TARGET_RESPONSE_TIME = 2 * 60  # in seconds
DEFAULT_VIA_TARGET_USECASE_EVENT_DURATION = 10  # in seconds

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
                # "/opt/nvidia/via/warehouse_graph_rag_config.yaml",
            )
    except Exception:
        return "", "", "", None


async def enable_button(gallery_data):
    yield (gr.update(interactive=True, value="Summarize"))
    return


def remove_icon_files():
    USER_AVATAR_ICON.close()
    CHATBOT_AVATAR_ICON.close()


atexit.register(remove_icon_files)


async def remove_all_media(session: aiohttp.ClientSession, media_ids):
    for media_id in media_ids:
        async with session.delete(appConfig["backend"] + "/files/" + media_id):
            pass


async def add_assets(
    gr_video,
    chatbot,
    image_mode,
    dc_json_path,
    request: gr.Request,
):
    logger.info(f"summarize. ip: {request.client.host}")
    if not gr_video:
        return [
            gr.update(),
        ] * 19
    else:
        url = appConfig["backend"] + "/files"
        session: aiohttp.ClientSession = appConfig["session"]

        media_ids = []
        if image_mode is True:
            media_paths = []
            for tup in gr_video:
                media_paths.append(tup[0])
                async with session.post(
                    url,
                    data={
                        "filename": (None, tup[0]),
                        "purpose": (None, "vision"),
                        "media_type": (None, "image"),
                    },
                ) as resp:
                    resp_json = await resp.json()
                    if resp.status >= 400:
                        chatbot = [[None, "<b>Error: </b><i>" + resp_json["message"] + "</i>"]]
                        await remove_all_media(session, media_ids)
                        return (
                            chatbot,
                            [],
                            *[
                                gr.update(),
                            ]
                            * 17,
                        )
                    media_ids.append(resp_json["id"])
            logger.debug(f"multi-img; media_paths is {str(media_paths)}")

        else:
            media_path = os.path.abspath(gr_video)
            # Copy dense caption json if its present
            enable_dense_caption = bool(os.environ.get("ENABLE_DENSE_CAPTION", False))
            if enable_dense_caption:
                if os.path.exists(dc_json_path):
                    dc_path = media_path + ".dc.json"
                    shutil.copy(dc_json_path, dc_path)

            async with session.post(
                url,
                data={
                    "filename": media_path,
                    "purpose": "vision",
                    "media_type": "video",
                },
            ) as resp:
                resp_json = await resp.json()
                if resp.status >= 400:
                    chatbot = [[None, "<b>Error: </b><i>" + resp_json["message"] + "</i>"]]
                    return (
                        chatbot,
                        [],
                        *[
                            gr.update(),
                        ]
                        * 17,
                    )
                media_ids.append(resp_json["id"])

        chatbot = []
        chatbot = chatbot + [
            [None, "Processing the image(s) ..." if image_mode else "Processing the video ..."]
        ]

        return (
            chatbot,
            media_ids,
            resp,
            *[
                gr.update(interactive=False),
            ]
            * 16,
        )


def convert_seconds_to_string(seconds, need_hour=False, millisec=False):
    seconds_in = seconds
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    if need_hour or hours > 0:
        ret_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        ret_str = f"{minutes:02d}:{seconds:02d}"

    if millisec:
        ms = int((seconds_in * 100) % 100)
        ret_str += f".{ms:02d}"
    return ret_str


def get_response_table(responses):
    return (
        "<table><thead><th>Duration</th><th>Response</th></thead><tbody>"
        + "".join(
            [
                f'<tr><td>{convert_seconds_to_string(item["media_info"]["start_offset"])} '
                f'-> {convert_seconds_to_string(item["media_info"]["end_offset"])}</td>'
                f'<td>{item["choices"][0]["message"]["content"]}</td></tr>'
                for item in responses
            ]
        )
        + "</tbody></table>"
    )


async def reset_chat(chatbot):
    # Reset all UI components to their initial state
    chatbot = []
    yield (chatbot)
    return


async def close_asset(chatbot, question_textbox, video, media_ids, image_mode):
    session: aiohttp.ClientSession = appConfig["session"]
    await remove_all_media(session, media_ids)
    # Reset all UI components to their initial state
    chatbot = []
    yield (
        chatbot,
        gr.update(interactive=False, value=""),  # question_textbox
        gr.update(interactive=False),  # ask_button
        gr.update(interactive=False),  # reset_chat_button
        gr.update(interactive=False),  # close_asset_button
        gr.update(value=None),  # video
        gr.update(
            interactive=False,
            value=f"Select/Upload {'image(s)' if image_mode else 'video'} to summarize",
        ),  # summarize_button
        gr.update(interactive=True),  # chat_button
        None,  # output_alerts
        gr.update(interactive=True),  # alerts
        gr.update(interactive=True, value=0),  # num_frames_per_chunk
        gr.update(interactive=True, value=0),  # vlm_input_width
        gr.update(interactive=True, value=0),  # vlm_input_height
        gr.update(interactive=True, value=0.4),  # temprature
        gr.update(interactive=True, value=1),  # top_p
        gr.update(interactive=True, value=100),  # top_k,
        gr.update(interactive=True, value=512),  # max_new_tokens
        gr.update(interactive=True, value=1),  # seed
    )
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
    max_new_tokens,
    top_p,
    top_k,
    num_frames_per_chunk,
    vlm_input_width,
    vlm_input_height,
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
        elif len(accumulated_responses) > 1:
            response_str = get_response_table(accumulated_responses)
        else:
            response_str = ""

        response_str = response_str.replace("\\n", "<br>").replace("\n", "<br>")
        if question != "/clear":
            chatbot = chatbot + [[None, response_str]]
        yield chatbot, gr.update(interactive=True), gr.update(
            interactive=not reset_chat_triggered
        ), gr.update(interactive=True)
        return


def get_output_string(header, items):
    return "\n--------\n".join([header] + items + [header]) if items else header


async def summarize(
    image_mode,
    gr_video,
    chatbot,
    media_ids,
    chunk_size,
    temperature,
    seed,
    max_new_tokens,
    top_p,
    top_k,
    summary_prompt,
    caption_summarization_prompt,
    summary_aggregation_prompt,
    response_obj,
    request: gr.Request,
    summarize=None,
    enable_chat=True,
    alerts=None,
    enable_cv_metadata=False,
    num_frames_per_chunk=0,
    vlm_input_width=0,
    vlm_input_height=0,
    graph_rag_prompt_yaml=None,
):
    logger.info(f"summarize. ip: {request.client.host}")
    if gr_video is None:
        yield (
            [
                gr.update(),
            ]
            * 18
        )
        return
    elif gr_video is not None and response_obj and media_ids:
        session: aiohttp.ClientSession = appConfig["session"]
        async with session.get(appConfig["backend"] + "/models") as resp:
            resp_json = await resp.json()
            if resp.status >= 400:
                chatbot = [[None, "<b>Error: </b><i>" + resp_json["message"] + "</i>"]]
                yield (
                    chatbot,
                    *[
                        gr.update(interactive=True),
                    ]
                    * 17,
                )
                return
            model = resp_json["data"][0]["id"]

        req_json = {
            "id": media_ids,
            "model": model,
            "chunk_duration": chunk_size,
            "temperature": temperature,
            "seed": seed,
            "max_tokens": max_new_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "stream": True,
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
        # if graph_rag_prompt_yaml:
        #     with open(graph_rag_prompt_yaml, "r") as f:
        #         yaml_data = yaml.safe_load(f)
        #     string_data = yaml.dump(yaml_data, default_flow_style=False)
        #     req_json["graph_rag_prompt_yaml"] = string_data
        req_json["enable_chat"] = enable_chat
        # req_json["enable_cv_metadata"] = enable_cv_metadata

        parsed_alerts = []
        accumulated_responses = []
        past_alerts = []
        if parsed_alerts:
            output_alerts = get_output_string(
                "Waiting for new alerts..." if past_alerts else "Waiting for alerts", past_alerts
            )
            yield (
                chatbot,
                output_alerts,
                *[
                    gr.update(),
                ]
                * 16,
            )
        else:
            output_alerts = ""
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

        async with session.post(appConfig["backend"] + "/summarize", json=req_json) as resp:
            if resp.status >= 400:
                resp_json = await resp.json()
                chatbot = []
                chatbot = chatbot + [[None, "<b>Error: </b><i>" + resp_json["message"] + "</i>"]]
                await remove_all_media(session, media_ids)
                yield (
                    chatbot,
                    *[
                        gr.update(interactive=True),
                    ]
                    * 17,
                )
                return
            while True:
                line = await resp.content.readline()
                if not line:
                    break
                line = line.decode("utf-8")
                if not line.startswith("data: "):
                    continue

                data = line.strip()[6:]

                if data == "[DONE]":
                    break
                response = json.loads(data)
                if response["choices"] and response["choices"][0]["finish_reason"] == "stop":
                    accumulated_responses.append(response)
                if response["usage"]:
                    usage = response["usage"]
                if (
                    parsed_alerts
                    and response["choices"]
                    and response["choices"][0]["finish_reason"] == "tool_calls"
                ):
                    alert = response["choices"][0]["message"]["tool_calls"][0]["alert"]
                    alert_str = (
                        f"Alert Name: {alert['name']}\n"
                        f"Detected Events: {', '.join(alert['detectedEvents'])}\n"
                        f"Time: {alert['offset']} seconds\n"
                        f"Details: {alert['details']}\n"
                    )
                    past_alerts = past_alerts[int(len(past_alerts) / 99) :] + (
                        [alert_str] if alert_str else []
                    )
                    output_alerts = get_output_string(
                        "Waiting for new alerts..." if past_alerts else "Waiting for alerts",
                        past_alerts,
                    )
                    yield (
                        chatbot,
                        output_alerts,
                        *[
                            gr.update(),
                        ]
                        * 16,
                    )

        if len(accumulated_responses) == 1:
            response_str = accumulated_responses[0]["choices"][0]["message"]["content"]
        elif len(accumulated_responses) > 1:
            response_str = get_response_table(accumulated_responses)
        else:
            response_str = ""
        if "Summarization failed" in response_str:
            chatbot = [[None, "<i>" + response_str + "</i>"]]
        elif response_str:
            if summarize is True:
                chatbot = [
                    [
                        None,
                        f"<b>Here is a summary of the {'image(s)' if image_mode else 'video'}</b>",
                    ],
                    [None, response_str],
                ]
            else:
                chatbot = [[None, f"<b> {'Image(s)' if image_mode else 'Video'} processed</b>"]]
            if usage:
                start_time = accumulated_responses[0]["media_info"]["start_offset"]
                end_time = accumulated_responses[0]["media_info"]["end_offset"]
                duration = end_time - start_time
                chatbot = chatbot + [
                    [
                        None,
                        f"<i>Processing Time: {usage['query_processing_time']:.2f} seconds\n"
                        f"{'' if image_mode else f'Stream Duration: {duration:.1f} seconds'}</i>",
                    ]
                ]
        else:
            chatbot = [[None, "<b>No summary was generated for given request</b>"]]

        # await remove_all_media(session, media_ids)

        if parsed_alerts and not past_alerts:
            output_alerts = "No alerts were generated for this input media"

        yield (
            chatbot,
            output_alerts,
            *[
                gr.update(interactive=True),
            ]
            * 16,
        )
        return
    else:
        yield (
            [
                gr.update(),
            ]
            * 18
        )
        return


CHUNK_SIZES = [
    ("No chunking", 0),
    ("5 sec", 5),
    ("10 sec", 10),
    ("20 sec", 20),
    ("30 sec", 30),
    ("1 min", 60),
    ("2 min", 120),
    ("5 min", 300),
    ("10 min", 600),
    ("20 min", 1200),
    ("30 min", 1800),
]


def validate_example_file(path, is_img=False):
    try:
        if (
            ".dc.json" in str(path)
            or ".prompts.json" in str(path)
            # or ".graph_rag.yaml" in str(path)
        ):
            return False
        media_info = MediaFileInfo.get_info(path)
        if media_info.video_codec:
            return bool(media_info.is_image) == bool(is_img)
    except Exception as ex:
        print(ex)
        return False
    return bool(media_info.video_codec)


def get_closest_chunk_size(CHUNK_SIZES, x):
    """
    Returns the integer value from CHUNK_SIZES that is closest to x.

    Args:
        CHUNK_SIZES (list of tuples): A list of tuples containing chunk size labels and values.
        x (int): The target value to find the closest chunk size to.

    Returns:
        int: The integer value from CHUNK_SIZES that is closest to x.
    """
    _, values = zip(*CHUNK_SIZES)  # extract just the values from CHUNK_SIZES
    closest_value = min(values, key=lambda v: abs(v - x))  # find the value closest to x
    return closest_value


async def get_recommended_chunk_size(video_length):
    # In seconds:
    target_response_time = DEFAULT_VIA_TARGET_RESPONSE_TIME
    usecase_event_duration = DEFAULT_VIA_TARGET_USECASE_EVENT_DURATION
    recommended_chunk_size = 0

    session: aiohttp.ClientSession = appConfig["session"]
    async with session.post(
        appConfig["backend"] + "/recommended_config",
        json={
            "video_length": int(video_length),
            "target_response_time": int(target_response_time),
            "usecase_event_duration": int(usecase_event_duration),
        },
    ) as response:
        if response.status < 400:
            # Success response from API:
            resp_json = await response.json()
            recommended_chunk_size = int(resp_json.get("chunk_size"))
        if recommended_chunk_size == 0:
            # API fail to provide non-zero chunk size
            # Choose the largest chunk-size in favor of quick VIA execution
            recommended_chunk_size = video_length
        return get_closest_chunk_size(CHUNK_SIZES, recommended_chunk_size)


async def chat_checkbox_selected(chat_checkbox):
    logger.debug("Chat box state updtaed to {}", chat_checkbox)
    return (
        gr.update(visible=chat_checkbox),
        gr.update(visible=chat_checkbox),
        gr.update(visible=chat_checkbox),
    )


async def video_changed(video, image_mode):
    if video:
        if image_mode:
            new_value = 0
        else:
            video_length = (await MediaFileInfo.get_info_async(video)).video_duration_nsec / (
                1000 * 1000 * 1000
            )
            logger.info(f"Video length: {video_length:.2f} seconds")
            new_value = await get_recommended_chunk_size(video_length)
        return [
            gr.update(interactive=True, value="Summarize"),
            gr.update(value=new_value),
        ]
    else:
        return [
            gr.update(
                interactive=False,
                value=f"Select/Upload {'image(s)' if image_mode else 'video'} to summarize",
            ),
            gr.update(value=0),
        ]


def get_example_details(f):
    dc_path = str()
    prompt, caption_summarization_prompt, summary_aggregation_prompt = get_default_prompts()

    if Path(str(f) + ".dc.json").exists():
        dc_path = str(f) + ".dc.json"

    # if Path(str(f) + ".graph_rag.yaml").exists():
    #     graph_rag_prompt_yaml = str(f) + ".graph_rag.yaml"

    try:
        if Path(str(f) + ".prompts.json").exists():
            with open(str(f) + ".prompts.json") as f:
                prompts = json.load(f)
                prompt = prompts["prompt"]
                caption_summarization_prompt = prompts["caption_summarization_prompt"]
                summary_aggregation_prompt = prompts["summary_aggregation_prompt"]
    except Exception:
        pass

    return (
        dc_path,
        prompt,
        caption_summarization_prompt,
        summary_aggregation_prompt,
        # graph_rag_prompt_yaml,
    )


def build_summarization(args, app_cfg, logger_):
    global appConfig, logger, pipeline_args
    appConfig = app_cfg
    logger = logger_
    pipeline_args = args

    (
        default_prompt,
        default_caption_summarization_prompt,
        default_summary_aggregation_prompt,
        # default_graph_rag_yaml,
    ) = get_default_prompts()

    # need it here for example
    show_configuration = gr.Button(
        "Show parameters", variant="primary", size="sm", render=False, elem_classes="gray-btn"
    )
    media_ids = gr.State("")
    response_obj = gr.State(None)
    with gr.Row(equal_height=False):

        with gr.Column(scale=1):

            if args.image_mode is False:
                video = gr.Video(
                    autoplay=True,
                    elem_classes=["white-background", "summary-video"],
                    sources=["upload"],
                    show_download_button=False,
                )
                chunk_size = gr.Dropdown(
                    choices=CHUNK_SIZES,
                    label="CHUNK SIZE",
                    value=DEFAULT_CHUNK_SIZE,
                    interactive=True,
                    visible=True,
                    elem_classes=["white-background", "bold-header"],
                )
            else:
                video = gr.Gallery(show_label=False, type="filepath")
                display_image = gr.Image(visible=False, type="filepath")
                chunk_size = gr.State(0)

            stream_name = gr.Textbox(show_label=False, visible=False)
            dc_json_path = gr.Textbox(show_label=False, visible=False)

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
            # with gr.Row():
            #     gr.Markdown("## Upload GraphRAG Prompt config yaml file")
            #     graph_rag_prompt_yaml = gr.File(
            #         type="filepath", file_count="single", value=default_graph_rag_yaml
            #     )  # , file_types=['.yaml', '.yml'])

            with gr.Row(equal_height=True):
                gr.Markdown(dummy_mr, visible=True)

            enable_cv_metadata = gr.State(False)

            chat_checkbox = gr.Checkbox(value=True, label="Enable Chat for the file")
            summarize_button = gr.Button(
                interactive=False,
                value=f"Select/Upload {'image(s)' if args.image_mode else 'video'} to summarize",
                variant="primary",
                size="sm",
                scale=1,
            )
            gr.Examples(
                examples=[
                    [f, f.stem, *get_example_details(f)]
                    for f in sorted(Path(args.examples_streams_directory).glob("*"))
                    if f.is_file() and validate_example_file(f.absolute(), args.image_mode)
                ],
                inputs=[
                    display_image if args.image_mode else video,
                    stream_name,
                    dc_json_path,
                    summary_prompt,
                    caption_summarization_prompt,
                    summary_aggregation_prompt,
                    # graph_rag_prompt_yaml,
                ],
                label="SELECT A SAMPLE",
                elem_id="example",
            )

        with gr.Column(scale=3):
            with gr.Tabs(elem_id="via-tabs"):
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
                with gr.Tab("ALERTS"):
                    output_alerts = gr.TextArea(
                        interactive=False, max_lines=30, lines=30, show_label=False
                    )

            with gr.Row(equal_height=False, variant="default"):
                gr.HTML("", visible=True).scale = 1
                show_configuration.render()
                show_configuration_state = gr.State(False)
            configuration = gr.Column(variant="compact", visible=False)

            with gr.Row(equal_height=False, variant="default"):
                close_asset_button = gr.Button(
                    "Close Session", variant="primary", interactive=False, size="sm"
                )

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
                                "The top-p sampling mass used for text generation."
                                " The top-p value determines the probability mass that is sampled"
                                " at sampling time. For example, if top_p = 0.2,"
                                " only the most likely"
                                " tokens (summing to 0.2 cumulative probability) will be sampled."
                                " It is not recommended to modify both temperature and top_p in the"
                                " same call."
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
                                "The number of highest probability vocabulary "
                                "tokens to keep for top-k-filtering"
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
                                " and generation will"
                                " simply stop at the number of tokens specified."
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
                                "Seed value to use for sampling. "
                                "Repeated requests with the same seed"
                                " and parameters should return the same result."
                            ),
                            elem_classes="white-background",
                        )

        ask_button.click(
            ask_question,
            inputs=[
                question_textbox,
                ask_button,
                reset_chat_button,
                video,
                chatbot,
                media_ids,
                chunk_size,
                temperature,
                seed,
                max_new_tokens,
                top_p,
                top_k,
                num_frames_per_chunk,
                vlm_input_width,
                vlm_input_height,
            ],
            outputs=[chatbot, ask_button, reset_chat_button, question_textbox],
        )

        reset_chat_button.click(
            ask_question,
            inputs=[
                gr.State("/clear"),
                ask_button,
                reset_chat_button,
                video,
                chatbot,
                media_ids,
                chunk_size,
                temperature,
                seed,
                max_new_tokens,
                top_p,
                top_k,
                num_frames_per_chunk,
                vlm_input_width,
                vlm_input_height,
            ],
            outputs=[chatbot, ask_button, reset_chat_button, question_textbox],
        ).then(
            reset_chat,
            inputs=[chatbot],
            outputs=[chatbot],
        )

        summarize_button.click(
            add_assets,
            inputs=[
                video,
                chatbot,
                gr.State(args.image_mode),
                dc_json_path,
            ],
            outputs=[
                chatbot,
                media_ids,
                response_obj,
                summarize_button,
                chat_checkbox,
                video,
                temperature,
                top_p,
                top_k,
                max_new_tokens,
                seed,
                chunk_size,
                summary_prompt,
                caption_summarization_prompt,
                summary_aggregation_prompt,
                enable_cv_metadata,
                num_frames_per_chunk,
                vlm_input_width,
                vlm_input_height,
            ],
            show_progress=False,
        ).then(
            summarize,
            inputs=[
                gr.State(args.image_mode),
                video,
                chatbot,
                media_ids,
                chunk_size,
                temperature,
                seed,
                max_new_tokens,
                top_p,
                top_k,
                summary_prompt,
                caption_summarization_prompt,
                summary_aggregation_prompt,
                response_obj,
                gr.State(True),  # summarize
                chat_checkbox,  # enable_chat
                alerts,
                gr.State(False),  # enable_cv_metadata
                num_frames_per_chunk,
                vlm_input_width,
                vlm_input_height,
            ],
            outputs=[
                chatbot,
                output_alerts,
                summarize_button,
                chat_checkbox,
                video,
                temperature,
                top_p,
                top_k,
                max_new_tokens,
                seed,
                chunk_size,
                summary_prompt,
                caption_summarization_prompt,
                summary_aggregation_prompt,
                enable_cv_metadata,
                num_frames_per_chunk,
                vlm_input_width,
                vlm_input_height,
            ],
            show_progress=False,
        ).then(
            lambda chat_checkbox: (
                gr.update(interactive=False),
                gr.update(interactive=chat_checkbox),
                gr.update(interactive=chat_checkbox),
                gr.update(interactive=chat_checkbox),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
            ),
            inputs=[chat_checkbox],
            outputs=[
                chat_checkbox,
                ask_button,
                reset_chat_button,
                question_textbox,
                close_asset_button,
                num_frames_per_chunk,
                vlm_input_width,
                vlm_input_height,
                temperature,
                top_p,
                top_k,
                max_new_tokens,
                seed,
            ],
        )

        chat_checkbox.select(
            chat_checkbox_selected,
            inputs=[chat_checkbox],
            outputs=[ask_button, reset_chat_button, question_textbox],
        )

        video.change(
            video_changed,
            inputs=[video, gr.State(args.image_mode)],
            outputs=[summarize_button, chunk_size],
        )
        video.upload(
            fn=enable_button,
            inputs=[video],
            outputs=[
                summarize_button,
            ],
        )

        if args.image_mode:

            def on_select_example(selected_images, image):
                selected_images = selected_images or []
                return (
                    selected_images + [image],
                    gr.update(interactive=True, value="Summarize"),
                    gr.update(interactive=True, value="Chat"),
                )

            display_image.change(
                on_select_example,
                inputs=[video, display_image],
                outputs=[video, summarize_button, ask_button],
            )

        close_asset_button.click(
            fn=close_asset,
            inputs=[chatbot, question_textbox, video, media_ids, gr.State(args.image_mode)],
            outputs=[
                chatbot,
                question_textbox,
                ask_button,
                reset_chat_button,
                close_asset_button,
                video,
                summarize_button,
                chat_checkbox,
                output_alerts,
                alerts,
                num_frames_per_chunk,
                vlm_input_width,
                vlm_input_height,
                temperature,
                top_p,
                top_k,
                max_new_tokens,
                seed,
            ],
        )
