import logging
from asyncio import Task, run, sleep
from typing import Any, Sequence

import civitai
import asyncio
from pydantic import BaseModel

from datetime import timedelta
import aiohttp
import os

_DOUG_PROMPT_EXAMPLE = ("score_9, (Doug Funnie), journal, glowing, "
                        "mecha_battle, giant_robot, cityscape, high_angle, "
                        "dynamic, anime90s, action")
_NEGATIVE_PROMPT = ("low quality, line art, deformed, ugly, sad, anxious, "
                    "depressing,  old, full hair, 3d, sketch, monochrome, "
                    "ecstacy, suggestive")

_MAX_DURATION: timedelta = timedelta(minutes=15)

# AutismMix SDXL: https://civitai.com/models/288584?modelVersionId=324619
_MODEL = "urn:air:sdxl:checkpoint:civitai:288584@324619"
_DOUG_LORA_MODEL = "urn:air:sdxl:lora:civitai:659532@737982"


class Result(BaseModel):
    blobKey: str
    available: bool = False
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


class ImageGenerator:
    def __init__(self):
        pass

    async def generate_image(self, prompt: str) -> str:
        resp = await generate_image()
        if resp is not None:
            await extract_urls(resp.token)
        return ""


async def do_get_job_from_token(token: str) -> JobResponse:
    # Have to implement my own poll because asyncio does not play well with polling2
    success = False
    total_sleep = 0
    sleep_duration = 15
    resp: JobResponse | None = None
    while (True):
        resp = await get_job_from_token(token)

        if resp is not None and all(job.result.available for job in resp.jobs):
            return resp

        if resp is None:
            raise ValueError("Received an empty response.")

        if total_sleep >= _MAX_DURATION.total_seconds():
            raise TimeoutError(
                f"Resp did not return in {_MAX_DURATION.total_seconds()} seconds.")

        await sleep(sleep_duration)
        total_sleep = total_sleep + sleep_duration


async def get_job_from_token(token: str) -> JobResponse:
    logging.info("Polling...")
    job_resp_dict = civitai.jobs.get(token=token)
    if not job_resp_dict:
        raise ValueError("Response not yet ready.")
    elif isinstance(job_resp_dict, Task):
        job_resp = JobResponse(**(await job_resp_dict))
    else:
        job_resp = JobResponse(**job_resp_dict)

    logging.info(f"job_resp: {job_resp}")
    logging.info(
        f"job_resp is all ready? {all(job.result.available for job in job_resp.jobs)}")
    return job_resp


async def extract_urls(resp_token: str, sleep_duration: timedelta = timedelta(seconds=15)) -> Sequence[str]:
    job_resp: JobResponse = await do_get_job_from_token(resp_token)
    return [job.result.blobUrl for job in job_resp.jobs]


def view_job(token: str) -> str:
    return civitai.jobs.get(token=token).__str__()


async def generate_image(prompt: str = _DOUG_PROMPT_EXAMPLE) -> ImageResponse | None:
    input: dict[str, Any] = {
        "model": _MODEL,
        "params": {
            # TODO: Make this accept themed Doug prompts, not fixed.
            "prompt": prompt,
            "negativePrompt": _NEGATIVE_PROMPT,
            "scheduler": "EulerA",
            "steps": 30,
            "cfgScale": 7,
            # "width": 832,
            # Only works if you locally modify civitai/schemas/__init__.py to
            # lift the 1024 limit.
            # https://github.com/civitai/civitai-python/issues/8
            # "height": 1216,

            # Cheap params for testing
            "width": 512,
            "height": 512,

            "clipSkip": 2
        },
        # "additionalNetworks": {
        #    _DOUG_LORA_MODEL: {
        #        "type": "Lora",
        #        "strength": 0.85
        #    }
        # }
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


def total_costs(resp: ImageResponse | None) -> float:
    if not resp:
        return 0

    total_cost = 0
    all_costs = [job.cost for job in resp.jobs]
    for cost in all_costs:
        total_cost = total_cost + cost
    return total_cost


async def main():
    # TODO
    # Read prompts from a Google Spreadsheet.
    # For each prompt generate a doug and write it into Google drive
    # Mark each prompt done somehow so we don't regenerate work.

    # Make generate_image actually generate Dougs
    # Somehow make it decide a themed prompt. Maybe calls to Gemini?

    # Currently this code will execute a hardcoded prompt and await for the URL of the image.
    response: ImageResponse | None = await generate_image(_DOUG_PROMPT_EXAMPLE)

    logging.info(f"Total cost is {total_costs(response)}")

    if response is None:
        logging.info("Failed to generate image.")
        return None

    logging.info(f"response is {response}")

    urls = await extract_urls(response.token)
    await download_images(urls)


async def download_images(urls: Sequence[str], prefix: str = "image_") -> None:
    # AI Generated
    for i, url in enumerate(urls):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    file_path = f"{prefix}{i}.jpeg"
                    with open(file_path, 'wb') as f:
                        f.write(content)
                    logging.info(f"Downloaded {file_path}")
                else:
                    logging.error(
                        f"Failed to download {url}, status code: {resp.status}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')
    run(main())
