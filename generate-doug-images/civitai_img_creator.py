import logging
from asyncio import Task, run, sleep
from typing import Any, Sequence

import civitai
import asyncio
from pydantic import BaseModel

from datetime import timedelta
import aiohttp
import os


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

    def __init__(self):
        pass

    async def generate_image(self, prompt: str) -> Sequence[str]:
        """ Generates an image on civitai and returns the URL of the image. """
        resp = await self._generate_image(prompt)
        logging.info(f"Total costs: {self._total_costs(resp)}")
        if resp is not None:
            return await self._extract_urls(resp.token)
        return []


    async def _extract_urls(self, resp_token: str, sleep_duration: timedelta = timedelta(seconds=15)) -> Sequence[str]:
        job_resp: JobResponse = await self._do_get_job_from_token(resp_token)
        return [job.result.blobUrl for job in job_resp.jobs]

    async def _generate_image(self, prompt: str = _DOUG_PROMPT_EXAMPLE) -> ImageResponse | None:
        input: dict[str, Any] = {
            "model": self._MODEL,
            "params": {
                # TODO: Make this accept themed Doug prompts, not fixed.
                "prompt": prompt,
                "negativePrompt": self._NEGATIVE_PROMPT,
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

    def _total_costs(self, resp: ImageResponse | None) -> float:
        if not resp:
            return 0

        total_cost = 0
        all_costs = [job.cost for job in resp.jobs]
        for cost in all_costs:
            total_cost = total_cost + cost
        return total_cost

    async def _do_get_job_from_token(self, token: str) -> JobResponse:
        # Have to implement my own poll because asyncio does not play well with polling2
        success = False
        total_sleep = 0
        sleep_duration = 15
        resp: JobResponse | None = None
        while (True):
            resp = await self._get_job_from_token(token)

            if resp is None:
                raise ValueError("Received an empty response.")

            if resp is not None and all(job.result.available for job in resp.jobs):
                return resp

            if total_sleep >= self._MAX_DURATION.total_seconds():
                raise TimeoutError(
                    f"Resp did not return in {self._MAX_DURATION.total_seconds()} seconds.")

            await sleep(sleep_duration)
            total_sleep = total_sleep + sleep_duration

    async def _get_job_from_token(self, token: str) -> JobResponse:
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
