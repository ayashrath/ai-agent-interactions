"""
Class to make gemini chat work

Not making it asynchronous as even in multi-agent stuff, I want the stuff to be synchronous.

TODO: Calculate input and output token values - and limit accordingly
TODO: Add proper type casting
TODO: Async possibility
"""

from dotenv import load_dotenv
from google import genai
from google.genai import types
import os
import time
from datetime import datetime
from termcolor import cprint
from db_manager import dump_history
from typing import List, Dict, Any


# consts
VALID_SAFETY_VALUES: List[str] = [
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_CIVIC_INTEGRITY"
]  # these 5 are the only supported ones for gemini
VALID_GOOGLE_TOOLS: List[str] = []  # TODO: fill in when needed
SAFETY_THRESHOLDS: List[str] = [
    "BLOCK_LOW_AND_ABOVE",
    "BLOCK_MEDIUM_AND_ABOVE",
    "BLOCK_ONLY_HIGH",
]
TOOL_CONFIG_VALUES: List[str] = [
    "AUTO",
    "ANY",
    "NONE",
]
AVAILABLE_MODELS: List[str] = [
    "gemini-3-pro-preview",  # most expensive
    "gemini-2.5-flash",  # ok-ish
    "gemini-2.5-flash-lite",  # cheapest
]  # these are the only ones I care about for now
available_config_options = [
    "system_instruction",
    "tools",
    "tool_config_mode",
    "allowed_tool_for_config",
    "safety_setting",
    "temperature",
    "top_p",
    "top_k",
    "max_output_tokens",
    "response_mime_type",
]

class GeminiAgent:
    """
    Creates and manages a Gemini Agent

    name: name of the agent
    model_name: model to use
    chat: the AsyncChat object
    """
    def __init__(self, name: str, model_name: str, client: genai.client.Client, config: Dict[str, Any] = None):
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"{model_name} is not valid!")
        if config is None:
            chat = client.chats.create(model=model_name)
        else:
            chat = client.chats.create(model=model_name, config=self._parse_config(config))

        self.name = name
        self.model_name = model_name
        self.context = []  # context data
        self.history = []  # history of stuff with the agent

        self.chat_obj = chat  # the thing that makes it a different instance with Gemini servers

        cprint(f"Remember to close the client and also dump history when done with the agent {self.name}", "red")

    @staticmethod
    def _parse_config(config: Dict[str, Any]) -> types.GenerateContentConfig:
        """
        Parses the config into something Gemini can understand
        :param config: config dict
        :return: Gemini config object

        Example config:
        ```
        config_example = {
            "system_instruction": "abc",
            "tools": ["tool1", "tool2"],
            "safety_setting": {
                "HARM_CONTENT_HARASSMENT":"BLOCK_ONLY_HIGH",
                "HARM_CATEGORY_HATE_SPEECH":"BLOCK_ONLY_HIGH",
            },
            "tool_config_mode": "AUTO",
            "allowed_tool_for_config": ["tool1"],
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 1024,
            "response_mime_type": "text/plain"
        }
        ```
        """

        extra = set(config.keys()) - set(available_config_options)
        if extra:
            raise ValueError(f"Unknown config keys: {sorted(extra)}")

        # system_instruction
        system_instruction = config.get("system_instruction")
        if system_instruction is not None and not isinstance(system_instruction, str):
            raise ValueError("system_instruction must be a string")

        # tools
        tools = config.get("tools")
        if tools is not None:
            if not isinstance(tools, list) or not all(isinstance(t, str) for t in tools):
                raise ValueError("tools must be a list of strings")

        # safety_setting
        safety_setting = config.get("safety_setting")
        safety_setting_collected = None
        if safety_setting is not None:
            if not isinstance(safety_setting, dict):
                raise ValueError("safety_setting must be a dict")
            safety_setting_collected = []
            for category, threshold in safety_setting.items():
                if category not in VALID_SAFETY_VALUES:
                    raise ValueError(f"{category} is not a valid safety category (use uppercase)")
                if threshold not in SAFETY_THRESHOLDS:
                    raise ValueError(f"{threshold} is not a valid safety threshold (use uppercase)")
                safety_setting_collected.append(
                    types.SafetySetting(category=category, threshold=threshold)
                )

        # tool_config_mode
        tool_config_mode = config.get("tool_config_mode")
        if tool_config_mode is not None:
            if not isinstance(tool_config_mode, str) or tool_config_mode not in TOOL_CONFIG_VALUES:
                raise ValueError(f"{tool_config_mode} Tool Config mode is invalid (maybe use all uppercase)")

        # allowed_tool_for_config
        allowed_tool_for_config = config.get("allowed_tool_for_config")
        if allowed_tool_for_config is not None:
            if not isinstance(allowed_tool_for_config, list) or not all(
                    isinstance(t, str) for t in allowed_tool_for_config):
                raise ValueError("allowed_tool_for_config must be a list of strings")
            if tools is not None and not set(allowed_tool_for_config).issubset(set(tools)):
                raise ValueError("allowed_tool_for_config must be a subset of tools")

        # temperature
        temperature = config.get("temperature")
        if temperature is not None:
            if not isinstance(temperature, (int, float)) or not (0.0 <= float(temperature) <= 2.0):
                raise ValueError("temperature must be a number between 0.0 and 2.0")

        # top_p
        top_p = config.get("top_p")
        if top_p is not None:
            if not isinstance(top_p, (int, float)) or not (0.0 <= float(top_p) <= 1.0):
                raise ValueError("top_p must be a number between 0.0 and 1.0")

        # top_k
        top_k = config.get("top_k")
        if top_k is not None:
            if not isinstance(top_k, int) or top_k < 0:
                raise ValueError("top_k must be a non-negative integer")

        # max_output_tokens
        max_output_tokens = config.get("max_output_tokens")
        if max_output_tokens is not None:
            if not isinstance(max_output_tokens, int) or max_output_tokens <= 0:
                raise ValueError("max_output_tokens must be a positive integer")

        # response_mime_type
        response_mime_type = config.get("response_mime_type")
        if response_mime_type is not None and not isinstance(response_mime_type, str):
            raise ValueError("response_mime_type must be a string")

        # Build config kwargs
        config_kwargs = {}
        if temperature is not None:
            config_kwargs["temperature"] = float(temperature)
        if top_p is not None:
            config_kwargs["top_p"] = float(top_p)
        if top_k is not None:
            config_kwargs["top_k"] = int(top_k)
        if max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = int(max_output_tokens)
        if system_instruction is not None:
            config_kwargs["system_instruction"] = system_instruction
        if tools is not None:
            config_kwargs["tools"] = tools
        if response_mime_type is not None:
            config_kwargs["response_mime_type"] = response_mime_type
        if safety_setting_collected is not None:
            config_kwargs["safety_config"] = safety_setting_collected
        if tool_config_mode is not None and allowed_tool_for_config is None:
            config_kwargs["tool_config"] = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode = tool_config_mode)
            )
        if tool_config_mode is not None and allowed_tool_for_config is not None:
            config_kwargs["tool_config"] = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode = tool_config_mode,
                    allowed_function_names=allowed_tool_for_config,
                )
            )

        return types.GenerateContentConfig(**config_kwargs)

    def reset_context(self):
        """
        Reset context memory
        """
        self.context = []

    def add_context(self, info_name: str, info: str):
        """
        Allows to add stuff the agent must know - like say some other agent's responses (for example)

        :param info_name: change to another agents response if you want to add it (for example)
        :param info: the main part of the thing
        """
        self.context.append(f"[{info_name}]{info}")

    def _join_context(self, prompt: str) -> str:
        """
            Adds context and message - using gemini's suggested way to add it now

            :param prompt: Prompt (can be "")
            """

        full_context = "\n".join(self.context)
        if full_context:
            return f"Context:\n{full_context}\n\nUser Query:\n{prompt}".strip()
        return prompt

    def send_message(self, prompt: str, print_response: bool = True) -> str:
        """
        Send prompt over to the servers

        :param prompt: Message to be sent ("" if it just needs to feed off context)
        :param print_response: Do you want it to print responses
        """

        prompt = self._join_context(prompt)
        if print_response:
            cprint(f"Input: {prompt}\n\n", "green")
        cprint("Output:", "blue")

        st_time = time.time()
        # try 3 times and then give up
        for i in range(1, 4):
            try:
                response = ""
                for chunk in self.chat_obj.send_message_stream(prompt):
                    response += chunk.text
                    if print_response:
                        cprint(chunk.text, end="", flush=True, color="blue")
                self.record_history(prompt, response)
                self.reset_context()
                if print_response:
                    cprint(f"\n\nTime Elapsed: {(time.time() - st_time):.2f} seconds\n", "yellow")  # newline
                return response
            except Exception as e:
                cprint(f"Some Exception {e}", "red")
                time.sleep(20 * 2 ** i)  # have 25 seconds as my Wi-Fi drops sometimes - thus makes sure it persists
                continue
        raise RuntimeError("Couldn't connect with the server or timeout")

    def record_history(self, prompt: str, response: str):
        """
        Records the history in a list of dicts
        :param prompt: prompt str
        :param response: response str
        :return:
        """
        timestamp = datetime.now().isoformat()
        self.history.append(
            {
                "timestamp": timestamp,
                "model": self.model_name,
                "name": self.name,
                "prompt": prompt,  # should not contain context, as if needed it should be added manually
                "response": response,
            }
        )

    def dump_history(self, project_name: str = ""):
        """
        Dumps the history into a sqlite db
        :param project_name: sqlite table name (default = default_project)
        :return:
        """
        # dump history
        dump_history(self.history, project_name=project_name)
        self.history = []


def create_gemini_client():
    """
    Creates the Gemini client
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key is None:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    client = genai.Client(api_key=api_key)

    return client

def close_client(client):
    client.close()


# python
if __name__ == "__main__":
    client = create_gemini_client()
    cfg_poet = {
        "system_instruction": "You are a lyrical poet. Keep responses short.",
        "temperature": 0.7,
        "max_output_tokens": 200,
        "response_mime_type": "text/plain",
    }

    cfg_helper = {
        "system_instruction": "You are a concise technical assistant.",
        "temperature": 0.2,
        "max_output_tokens": 300,
        "response_mime_type": "text/plain",
    }

    # create two agents with dict configs
    poet_agent = GeminiAgent("poet", "gemini-2.5-flash", client, cfg_poet)
    helper_agent = GeminiAgent("helper", "gemini-2.5-flash", client, cfg_helper)

    # add distinct context for each (info_name first, info second)
    poet_agent.add_context("Background", "The user loves robots and sonnets.")
    helper_agent.add_context("Guideline", "Prefer short, actionable answers.")

    print("\nPoet response:\n")
    resp_poet = poet_agent.send_message("Write a two-line poem about robots.", print_response=True)

    print("\nHelper response:\n")
    resp_helper = helper_agent.send_message("How to optimize a Python loop?", print_response=True)

    # persist histories
    poet_agent.dump_history(project_name="default_project")
    helper_agent.dump_history(project_name="default_project")

    close_client(client)