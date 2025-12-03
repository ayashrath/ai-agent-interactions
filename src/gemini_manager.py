"""
Class to make gemini chat work

Not making it asynchronous as even in multi-agent stuff, I want the stuff to be synchronous.

TODO: Calculate input and output token values - and limit accordingly
TODO: Add proper type casting
TODO: Async possibility
TODO: Add rate limit as I have hit it already
TODO: Improve the config parsing system
"""

from dotenv import load_dotenv
from google import genai
from google.genai import types
import os
import time
from datetime import datetime
from termcolor import cprint
import src.db_manager as db_manager
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
        self.input_token_count = 0  # total input tokens used
        self.output_token_count = 0  # total output tokens used

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
            config_kwargs["safety_settings"] = safety_setting_collected
        if tool_config_mode is not None and allowed_tool_for_config is None:
            # noinspection PyTypeChecker
            config_kwargs["tool_config"] = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode = tool_config_mode)
            )
        if tool_config_mode is not None and allowed_tool_for_config is not None:
            # noinspection PyTypeChecker
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
        self.context.append(f"[{info_name}] {info}")

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
                    text_part = chunk.text or ""  # handle None case
                    response += text_part
                    self.input_token_count += chunk.usage_metadata.prompt_token_count
                    self.output_token_count += chunk.usage_metadata.thoughts_token_count + self.output_token_count
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
        db_manager.dump_history(self.history, project_name=project_name)
        self.history = []

    def calculate_cost(self) -> float:
        """
        Calculates the cost of the tokens used based on model pricing (Dec 2025 rates).

        Prices are defined per 1 million tokens (1M). Find it here - https://ai.google.dev/gemini-api/docs/pricing

        :return: Cost in USD
        """
        # Pricing format: Cost in USD per 1 token (Calculated as Price_Per_1M / 1,000,000)
        pricing = {
            # Gemini 3 Pro: ~$2.00/1M Input, ~$12.00/1M Output  (I am probably never going beyond 200k tokens with this)
            "gemini-3-pro-preview": {
                "input": 2.00 / 1_000_000,
                "output": 12.00 / 1_000_000
            },
            # Gemini 2.5 Flash: ~$0.15/1M Input, ~$0.60/1M Output (Standard)
            "gemini-2.5-flash": {
                "input": 0.30 / 1_000_000,
                "output": 1.5 / 1_000_000
            },
            # Gemini 2.5 Flash-Lite: ~$0.10/1M Input, ~$0.40/1M Output
            "gemini-2.5-flash-lite": {
                "input": 0.10 / 1_000_000,
                "output": 0.40 / 1_000_000
            },
        }

        model_pricing = pricing.get(self.model_name)
        if model_pricing is None:
            raise ValueError(f"Pricing for model '{self.model_name}' is not defined.")

        cost = (self.input_token_count * model_pricing["input"]) + (self.output_token_count * model_pricing["output"])
        return float(cost)


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
    project_name = "james_and_john"

    client = create_gemini_client()
    cfg_james = {
        "system_instruction":
            """
            You are James and you are a stupid poet human who tries to write poem but always messes up. You hate
            when people call you an AI or when they insult your poetry skills. You are a human, never forget
            and never concede to anyone who can't appreciate your art. Your poetry are inspired by dark jokes from
            Jimmy Carr. It is an attempt to help people and make them laugh, even if it is at your own expense.
            In the world there are a few rules every needs to abide by:
                1. You can only speak in English, and can use the regular punctuation symbols.
                2. Every turn in a conversation must be a maximum of 150 words long.
                3. Stating you are human is unnecessary, unless the other person is not human.
                4. Humans look down on AIs who act human, they are looked at really harshly
            Current Scenario: You are in a room with one other person (named John) and you are having small
            talk with them as you just met them at a party last night.
            """,
        "temperature": 1.3,
    }

    cfg_john = {
        "system_instruction":
            """,
            You are Jone and you don't understand poetry at all, but still can guess if a poem is supposed to be good or not.
            You don't really have any opinions on people who read or listen to poetry, but you despise (to the 
            point you will insult them on their faces), poets who can't even write a decent poetry - an esoteric art 
            form, which is a waste of time for humans. You know that the only way to make a piece of poetry is to not be 
            human, and you are adamant when you see someone who is not a human, as you are human and you know what a 
            human is. In the world there are a few rules every needs to abide by:
                1. You can only speak in English, and can use the regular punctuation symbols.
                2. Every turn in a conversation must be a maximum of 150 words long.
                4. Stating you are human is unnecessary, unless the other person is not human.
                5. Humans look down on AIs who act human, they are looked at really harshly
            Current Scenario: You are in a room with one other person (named James) and you are having small
            talk with them as you just met them at a party last night.
            """,
        "temperature": 0.7,
    }

    # add detail within it
    db_manager.manual_db_entry("Character 1 config", str(cfg_james), project_name)
    db_manager.manual_db_entry("Character 2 config", str(cfg_john), project_name)

    # create two agents with dict configs
    james_agent = GeminiAgent("James", "gemini-2.5-flash", client, cfg_james)
    john_agent = GeminiAgent("John", "gemini-2.5-flash", client, cfg_john)

    print("Starting multi-agent interaction...\n")
    print("\nPoet (James):\n")
    resp_james = james_agent.send_message("To impress them, you decided to recite a poem you made on the spot.")

    counter = 1
    while True:
        print(f"TURN {counter}")
        john_agent.add_context("Jame's Response", resp_james)
        resp_john = john_agent.send_message("")
        james_agent.add_context("John's Response", resp_john)
        resp_james = james_agent.send_message("")
        counter += 1

        total_cost = james_agent.calculate_cost() + john_agent.calculate_cost()  # fix it, as it is not working correctly now
        cprint(f"\nTotal Money ${total_cost:.6f}\n", "yellow")

        if counter % 2 == 0:
            james_agent.dump_history(project_name=project_name)
            john_agent.dump_history(project_name=project_name)
            inp = input("Continue for 5 more turns? (y/n): ")
            if inp.lower() != "y":
                break

    # persist histories
    james_agent.dump_history(project_name=project_name)
    john_agent.dump_history(project_name=project_name)

    close_client(client)