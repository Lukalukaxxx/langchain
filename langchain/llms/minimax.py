"""Wrapper around MiniMaxCompletion API."""
import logging
from typing import Any, Dict, List, Mapping, Optional

import requests
from pydantic import BaseModel, Extra, root_validator

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class _MinimaxEndpointClient(BaseModel):
    """An API client that talks to a Minimax llm endpoint."""

    group_id: str
    api_key: str
    api_url: str

    @root_validator(pre=True)
    def set_api_url(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "api_url" not in values:
            host = "https://api.minimax.chat"
            group_id = values["group_id"]
            api_url = f"{host}/v1/text/chatcompletion?GroupId={group_id}"
            values["api_url"] = api_url
        return values

    def post(self, request: Any) -> Any:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(self.api_url, headers=headers, json=request)
        # TODO: error handling and automatic retries
        if not response.ok:
            raise ValueError(f"HTTP {response.status_code} error: {response.text}")
        if response.json()["base_resp"]["status_code"] > 0:
            raise ValueError(
                f"API {response.json()['base_resp']['status_code']}"
                f" error: {response.json()['base_resp']['status_msg']}"
            )
        return response.json()["reply"]


class _MinimaxCommon(BaseModel):
    client: _MinimaxEndpointClient = None  #: :meta private:

    model_name: str = "abab5-chat"
    """Model name to use, support 'abab5-chat' and 'abab5.5-chat'"""

    temperature: float = 0.9
    """What sampling temperature to use"""

    tokens_to_generate: int = 256
    """The maximum number of tokens to generate in the completion."""

    top_p: float = 0.95
    """Total probability mass of tokens to consider at each step."""

    skip_info_mask: bool = False
    """De-sensitize text information in the output that might involve privacy issues."""

    minimax_group_id: Optional[str] = None
    """Group ID for MiniMax API."""

    minimax_api_key: Optional[str] = None
    """API Key for MiniMax API."""

    class Config:
        """Configuration for this pydantic config."""

        extra = Extra.ignore

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["minimax_group_id"] = get_from_dict_or_env(
            values, "minimax_group_id", "MINIMAX_GROUP_ID"
        )
        values["minimax_api_key"] = get_from_dict_or_env(
            values, "minimax_api_key", "MINIMAX_API_KEY"
        )
        values["client"] = _MinimaxEndpointClient(
            api_key=values["minimax_api_key"],
            group_id=values["minimax_group_id"],
        )
        return values

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling GooseAI API."""
        normal_params = {
            "model": self.model_name,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "tokens_to_generate": self.tokens_to_generate,
            "skip_info_mask": self.skip_info_mask,
        }
        return {**normal_params}

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{"model_name": self.model_name}, **self._default_params}


class MiniMaxCompletion(LLM, _MinimaxCommon):
    """Wrapper around Minimax large language models.

    To use, you should have the environment variable ``MINIMAX_GROUP_ID`` and
    ``MINIMAX_API_KEY`` set with your API token, or pass it as a named parameter to
    the constructor.

    Example:
        .. code-block:: python

            from langchain.llms import MiniMaxCompletion
            llm = MiniMaxCompletion(model_name="abab5-chat")

    """

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "minimax"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the MiniMax API."""
        payload = self._default_params
        payload["messages"] = [{"sender_type": "USER", "text": prompt}]
        text = self.client.post(payload)

        # This is required since the stop are not enforced by the model parameters
        return text if stop is None else enforce_stop_tokens(text, stop)
