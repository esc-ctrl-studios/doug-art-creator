import logging
import time
from asyncio import Task, run
from typing import Any, Sequence

import civitai
from pydantic import BaseModel


_DOUG_PROMPT_EXAMPLE = ("score_9, (Doug Funnie), journal, glowing, "
                        "mecha_battle, giant_robot, cityscape, high_angle, "
                        "dynamic, anime90s, action")
_NEGATIVE_PROMPT = ("low quality, line art, deformed, ugly, sad, anxious, "
                    "depressing,  old, full hair, 3d, sketch, monochrome, "
                    "ecstacy, suggestive")

# AutismMix SDXL: https://civitai.com/models/288584?modelVersionId=324619
_MODEL = "urn:air:sdxl:checkpoint:civitai:288584@324619"
_DOUG_LORA_MODEL = "urn:air:sdxl:lora:civitai:659532@737982"

class Result(BaseModel):
    blobKey: str
    available: bool
    blobUrl: str = ''
    blobUrlExpirationDate: str = ''


class Job(BaseModel):
    jobId: str
    cost: float
    result: Result
    scheduled: bool

class ImageResponse(BaseModel):
    token: str
    jobs: Sequence[Job]

class JobResponse(BaseModel):
    token: str
    jobs: Sequence[Job]


async def extract_urls(resp: ImageResponse) -> None:
    all_ready = False
    job_resp: JobResponse|None = None
    while (not all_ready):
        job_resp_dict = civitai.jobs.get(token=resp.token)
        if isinstance(job_resp_dict, Task):
            job_resp = JobResponse(**(await job_resp_dict))
        else:
            job_resp = JobResponse(**job_resp_dict)
        all_ready = all(job.result.available for job in job_resp.jobs)
        if not all_ready:
            time.sleep(5) # Sleep for 5 seconds
    if job_resp is None:
        logging.info("Bad state. job_resp should not be None here.")
    logging.info(job_resp)
    


def view_job(token: str) -> str:
    return civitai.jobs.get(token=token).__str__()


async def generate_image() -> ImageResponse | None:
    input: dict[str, Any] = {
        "model": _MODEL,
        "params": {
            # TODO: Make this accept themed Doug prompts, not fixed.
            "prompt": _DOUG_PROMPT_EXAMPLE ,
            "negativePrompt": _NEGATIVE_PROMPT,
            "scheduler": "EulerA",
            "steps": 30,
            "cfgScale": 7,
            "width": 832,
            # Only works if you locally modify civitai/schemas/__init__.py to 
            # lift the 1024 limit.
            # https://github.com/civitai/civitai-python/issues/8
            "height": 1216,
            "clipSkip": 2
        },
        "additionalNetworks": {
            _DOUG_LORA_MODEL: {
                "type": "Lora",
                "strength": 0.85
            }
        }

    }

    response = civitai.image.create(input)

    # To mock the above call, comment it out and use the following... 
    # TODO: Do this more professionally ;)
    # resp_str = """
    # {"token": "eyJKb2JzIjpbIjMyOThkOGE3LWUwNzUtNDMzMC05M2M1LWViMDMyZmM1NTEyYyJdfQ==", "jobs": [{"jobId": "3298d8a7-e075-4330-93c5-eb032fc5512c", "cost": 0.6400000000000001, "result": {"blobKey": "B6EE40547244BDEC2ADC15C837ED763BBC494F6B1D11592991EB989859B54212", "available": "False"}, "scheduled": "True"}]}
    # """
    # response = json.loads(resp_str)

    if not response:
        logging.info("No response returned.")
        return None
    elif isinstance(response, Task):
        logging.info("Task returned, awaiting..")
        response_val = await response
        if response_val is None:
            logging.info("Task resulted in None")
            return None
        return ImageResponse(**response_val)
    else:
        return ImageResponse(**response)


async def main():
    # TODO
    # Read prompts from a Google Spreadsheet.
    # For each prompt generate a doug and write it into Google drive
    # Mark each prompt done somehow so we don't regenerate work.

    # Make generate_image actually generate Dougs
    # Somehow make it decide a themed prompt. Maybe calls to Gemini?

    # Currently this code will execute a hardcoded prompt and await for the URL of the image.
    response = await generate_image()
    if response is None:
        logging.info("Failed to generate image.")
        return None

    logging.info(f"response is {response}")

    await extract_urls(response)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')
    run(main())