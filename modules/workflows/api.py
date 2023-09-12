import os
import gc
import json
import time
import uuid
from aiohttp import web
from typing import Dict
from types import MethodType

import folder_paths
from server import PromptServer
from execution import validate_prompt, PromptExecutor
from comfy.model_management import soft_empty_cache

from ..utils import set_dict_attribute

current_dir = os.path.dirname(os.path.abspath(__file__))

# look for json files
workflows = {}
for file in os.listdir(current_dir):
    if file.endswith(".json"):
        workflows[file[:-5]] = os.path.join(current_dir, file)


executor = PromptExecutor(PromptServer.instance)
orig_handle_execution_error = executor.handle_execution_error


async def handler(prompt):
    valid = validate_prompt(prompt)
    if not valid[0]:
        execution_ex = {"error": valid[1], "details": valid[3]}
        return web.json_response(execution_ex, status=400)

    prompt_id = str(uuid.uuid4())
    execution_ex = None

    def on_finish(outputs: Dict):
        print("on_finish callback", outputs)

        outdir = folder_paths.get_output_directory()

        images = []
        for k, v in outputs.items():
            files = v.get("images", [])
            for image in files:
                type = image.get("type", None)
                if type == "output":
                    filename = image.get("filename")
                    subfolder = image.get("subfolder", "")
                    images.append(os.path.join(outdir, subfolder, filename))

        return images

    def on_error(self, prompt_id, prompt, current_outputs, executed, error, ex):
        nonlocal execution_ex
        node_id = error["node_id"]
        class_type = prompt[node_id]["class_type"]
        mes = {
            "type": error["exception_type"],
            "message": error["exception_message"],
        }
        details = {
            "node_id": node_id,
            "node_type": class_type,
            "executed": list(executed),
            "traceback": error["traceback"],
        }
        execution_ex = {"error": mes, "details": details}
        orig_handle_execution_error(
            prompt_id, prompt, current_outputs, executed, error, ex
        )

    execution_start_time = time.perf_counter()
    executor.handle_execution_error = MethodType(on_error, executor)
    executor.execute(prompt, prompt_id, execute_outputs=valid[2])

    gc.collect()
    soft_empty_cache()

    if execution_ex is None:
        print(
            "Prompt executed in {:.2f} seconds".format(
                time.perf_counter() - execution_start_time
            )
        )
        images = on_finish(executor.outputs_ui)

        return web.FileResponse(images[0])
    else:
        return web.json_response(execution_ex, status=500)


@PromptServer.instance.routes.post("/av/workflows/{name:.+}")
async def workflow_handler(request):
    workflow_name = request.match_info["name"]
    print("workflow_name", workflow_name, workflows)
    if workflow_name not in workflows:
        return web.Response(status=404)

    workflow = json.load(open(workflows[workflow_name], "r"))
    params: Dict = await request.json()

    prompt = workflow["prompt"]
    args = workflow.get("args", {})
    for key, value in args.items():
        if key in params:
            for p in value:
                set_dict_attribute(prompt, p, params[key])

    print("prompt", prompt)

    return await handler(prompt)