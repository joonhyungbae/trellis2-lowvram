"""Hugging Face token and gated-model access.

DINOv3 (the image encoder TRELLIS.2 uses) is gated with manual approval by Meta. Each student
requests access with their own account; we never redistribute the weights (docs/PLAN.md 0).
The token is stored under our HF_HOME (models/huggingface/token), not in the user's global setup.
"""

from __future__ import annotations

import logging
from enum import Enum

log = logging.getLogger(__name__)

DINOV3_REPO = "facebook/dinov3-vitl16-pretrain-lvd1689m"
JOIN_URL = "https://huggingface.co/join"
TOKEN_URL = "https://huggingface.co/settings/tokens"
DINOV3_URL = f"https://huggingface.co/{DINOV3_REPO}"


class Access(str, Enum):
    OK = "ok"
    NO_TOKEN = "no_token"
    BAD_TOKEN = "bad_token"
    NOT_APPROVED = "not_approved"  # not requested yet, or waiting for Meta's approval
    NETWORK = "network"


def get_token() -> str | None:
    from huggingface_hub import get_token as _get

    return _get()


def save_token(token: str) -> None:
    from huggingface_hub import login

    login(token=token, add_to_git_credential=False)


def check_access(repo_id: str = DINOV3_REPO, token: str | None = None) -> Access:
    from huggingface_hub import auth_check, whoami
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    token = token or get_token()
    if not token:
        return Access.NO_TOKEN
    try:
        whoami(token=token)
    except HfHubHTTPError as e:
        if e.response is not None and e.response.status_code == 401:
            return Access.BAD_TOKEN
        log.debug("whoami failed: %r", e)
        return Access.NETWORK
    except Exception as e:  # connection errors
        log.debug("whoami failed: %r", e)
        return Access.NETWORK
    try:
        auth_check(repo_id, token=token)
    except GatedRepoError:
        return Access.NOT_APPROVED
    except RepositoryNotFoundError:
        return Access.BAD_TOKEN
    except Exception as e:
        log.debug("auth_check failed: %r", e)
        return Access.NETWORK
    return Access.OK
