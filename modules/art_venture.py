import os
import io
import time
import base64
import pathlib
import requests
import threading
import traceback
from uuid import uuid4
from typing import Callable, Dict, List, Union
from types import MethodType
from aiohttp import web
from PIL import Image

from ..config import config
from .logger import logger
from .utils import get_task_from_av, upload_to_av

import folder_paths
from server import PromptServer
from execution import validate_prompt, PromptExecutor

Callback = Union[str, Callable]


def handle_task_finished(
    success: bool, images: List[str] = None, callback: Callback = None
):
    if callback is None:
        return

    data = {"success": str(success).lower()}
    if not isinstance(callback, str):
        data["images"] = images
        callback(data)
        return

    files = None
    if images is not None:
        files = []
        for img in images:
            img_path = pathlib.Path(img)
            ext = img_path.suffix.lower()
            content_type = f"image/{ext[1:]}"
            files.append(
                (
                    "files",
                    (img_path.name, open(os.path.abspath(img), "rb"), content_type),
                )
            )

    return upload_to_av(
        files,
        additional_data=data,
        upload_url=callback,
    )


def patch_comfy():
    # monky patch PromptQueue
    orig_task_done = PromptServer.instance.prompt_queue.task_done

    def handle_task_done(queue, item_id, outputs):
        item = queue.currently_running.get(item_id, None)
        if item:
            task_id = item[1]
            ArtVentureRunner.instance.on_task_finished(task_id, outputs)

        orig_task_done(item_id, outputs)

    PromptServer.instance.prompt_queue.task_done = MethodType(
        handle_task_done, PromptServer.instance.prompt_queue
    )

    # monky patch PromptExecutor
    PromptExecutor.orig_handle_execution_error = PromptExecutor.handle_execution_error

    def handle_execution_error(
        self, prompt_id, prompt, current_outputs, executed, error, ex
    ):
        node_id = error["node_id"]
        class_type = prompt[node_id]["class_type"]
        mes = {
            "type": error.get("exception_type", ex.__class__.__name__),
            "message": error.get("exception_message", str(ex)),
        }
        details = {
            "node_id": node_id,
            "node_type": class_type,
            "traceback": error.get("traceback", None),
        }
        ArtVentureRunner.instance.current_task_exception = {
            "error": mes,
            "details": details,
        }
        self.orig_handle_execution_error(
            prompt_id, prompt, current_outputs, executed, error, ex
        )

    PromptExecutor.handle_execution_error = handle_execution_error


class ArtVentureRunner:
    instance: "ArtVentureRunner" = None

    def __init__(self) -> None:
        self.current_task_id: str = None
        self.callback: Callback = None
        self.current_task_exception = None
        self.current_thread: threading.Thread = None
        ArtVentureRunner.instance = self

        patch_comfy()

    def register_new_task(self, prompt: Dict, callback: Callback = None):
        if self.current_task_id is not None:
            return (callback, Exception("Already running a task"))

        valid = validate_prompt(prompt)
        if not valid[0]:
            logger.error(f"Invalid recipe: {valid[3]}")
            return (callback, Exception("Invalid recipe"))

        task_id = str(uuid4())
        outputs_to_execute = valid[2]
        PromptServer.instance.prompt_queue.put(
            (0, task_id, prompt, {}, outputs_to_execute)
        )

        logger.info(f"Task registered with id {task_id}")
        self.current_task_id = task_id
        self.callback = callback
        self.current_task_exception = None
        return (callback, None)

    def get_new_task(self):
        if config.get("runner_enabled", False) != True:
            return (None, None)

        if self.current_task_id is not None:
            return (None, None)

        try:
            data = get_task_from_av()
        except Exception as e:
            logger.error(f"Error while getting new task {e}")
            return (None, e)

        if data["has_task"] != True:
            return (None, None)

        prompt = data.get("prompt")
        callback: str = data.get("callback_url")
        logger.info(f"Got new task")
        logger.debug(prompt)

        return self.register_new_task(prompt, callback)

    def watching_for_new_task(self, get_task: Callable):
        logger.info("Watching for new task")

        failed_attempts = 0
        while True:
            if self.current_task_id is not None:
                time.sleep(2)
                continue

            try:
                callback, e = get_task()
                if callback and e is not None:
                    logger.error("Error while getting new task")
                    logger.error(e)
                    handle_task_finished(callback, False)
                    failed_attempts += 1
                else:
                    failed_attempts = 0
            except requests.exceptions.ConnectionError:
                logger.error("Connection error while getting new task")
                failed_attempts += 1
            except Exception as e:
                logger.error("Error while getting new task")
                logger.error(e)
                logger.debug(traceback.format_exc())
                failed_attempts += 1

            # increase sleep time based on failed attempts
            time.sleep(min(3 + 5 * failed_attempts, 60))

    def watching_for_new_task_threading(self):
        if config.get("runner_enabled", False) != True:
            logger.info("Runner is disabled")
            return

        if self.current_thread is not None and self.current_thread.is_alive():
            return

        self.current_thread = threading.Thread(
            target=self.watching_for_new_task, args=(self.get_new_task,)
        )
        self.current_thread.daemon = True
        self.current_thread.start()

    def on_task_finished(
        self,
        task_id: str,
        outputs: Dict,
    ):
        if task_id != self.current_task_id:
            return

        if self.current_task_exception is not None:
            logger.info(f"Task {task_id} failed: {self.current_task_exception}")
            handle_task_finished(callback=self.callback, success=False)
        else:
            images = []
            outdir = folder_paths.get_output_directory()
            for k, v in outputs.items():
                files = v.get("images", [])
                for image in files:
                    type = image.get("type", None)
                    if type == "output":
                        filename = image.get("filename")
                        subfolder = image.get("subfolder", "")
                        images.append(os.path.join(outdir, subfolder, filename))

            logger.info(f"Task {task_id} finished with {len(images)} image(s)")
            handle_task_finished(self.callback, True, images)
            if config.get("remove_runner_images_after_upload", False):
                for img in images:
                    if os.path.exists(img):
                        os.remove(img)

        self.current_task_id = None
        self.callback = None
        self.current_task_exception = None


@PromptServer.instance.routes.post("/av/task/register")
async def register_new_task(request):
    prompt: Dict = await request.json()

    def callback(data: Dict):
        images = data.pop("images", [])
        base64_images = []
        for f in images:
            img = Image.open(os.path.abspath(f))
            with io.BytesIO() as output_bytes:
                img.save(output_bytes, format="PNG")
                bytes_data = output_bytes.getvalue()
                base64_images.append("data:image/png;base64," + base64.b64encode(bytes_data).decode("utf-8"))

        data["images"] = base64_images

        return web.json_response(data)

    ArtVentureRunner.instance.register_new_task(prompt, callback)
