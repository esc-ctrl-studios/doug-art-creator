# Doug Art Creator

General idea: Two components - one to generate prompts from themes, another to execute those prompts on civit.ai.

In between the current idea is to have a Google spreadsheet serve as repository of the prompts, then the results can be dumped into a Google Drive format.


## Environment Setup steps (1st time only)

The following steps were done with Python 3.12.

```
# Create virtual environment
python -m venv .venv

# Use the virtual environment
source .venv/bin/activate

# From https://developer.civitai.com/docs/api/python-sdk
# Install civitai-py
python -m pip install civitai-py mypy logging
```

## Environment Setup steps (after initialized)

```
# Use the virtual environment
source .venv/bin/activate
```

## VSCode

Python will probably complain about imports. Need to select the right interpreter from the `.venv` directory.

## Design FAQ

### Why use pydandic instead of datamodels.dataclass?

The response type of both `civitai.image.create` and `civitai.job.get` are both nested dicts. I didn't know a good way to automatically unpack the nested dicts into dataclasses, so I went the way of pydantic.
