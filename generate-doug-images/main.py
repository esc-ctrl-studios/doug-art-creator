import logging
from asyncio import Task, run, sleep
from typing import Any, Sequence

import civitai
import asyncio
from pydantic import BaseModel

from datetime import timedelta
import aiohttp
import os
from civitai_img_creator import ImageGenerator

_DOUG_PROMPT_EXAMPLE = ("score_9, (Doug Funnie), journal, glowing, "
                        "mecha_battle, giant_robot, cityscape, high_angle, "
                        "dynamic, anime90s, action")


async def main():
    # TODO
    # Read prompts from a Google Spreadsheet.
    # For each prompt generate a doug and write it into Google drive
    # Mark each prompt done somehow so we don't regenerate work.

    # Make generate_image actually generate Dougs
    # Somehow make it decide a themed prompt. Maybe calls to Gemini?

    # Currently this code will execute a hardcoded prompt and await for the URL of the image.
    gen = ImageGenerator()
    urls = await gen.generate_image(_DOUG_PROMPT_EXAMPLE)
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
