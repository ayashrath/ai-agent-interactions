"""
Class to make gemini chat work

Not making it asynchronous as even in multi-agent stuff, I want the stuff to be synchronous.

TODO: Add a rate limiter or something maybe?
TODO: Maybe more colourful cli output
TODO: Calculate input and output token values
TODO: Add proper type casting
"""

from dotenv import load_dotenv
from google import genai
from google.genai.chats import Chat
from google.genai import types
import yaml
import os
import time
from datetime import datetime
from termcolor import cprint
from db_manager import dump_history
from typing import List, Tuple, Any, Dict


# consts
VALID_SAFETY_VALUES: List[str] = [
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_CIVIC_INTEGRITY"
]  # these 5 are the only supported ones for gemini
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


class GeminiAgent:
    """
    A single agent (chat) struct

    name: name of the agent
    model_name: model to use
    chat: the AsyncChat object
    """
    def __init__(self, name: str, model_name: str, chat: Chat):
        self.name = name
        self.model_name = model_name
        self.history = []  # history
        self.chat_obj = chat  # the thing that makes it a different instance with Gemini servers
        self.context = []  # context data

    def reset_context(self):
        """
        Reset context memory
        """
        self.context = []

    def add_context(self, info_name: str, info: str):
        """
        Adds context within context

        :param info_name: change to another agents response if you want to add it (for example)
        :param info: the main part of the thing
        """
        self.context.append(f"[{info_name}]{info}")

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
                "prompt": prompt,
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

class GeminiConfig:
    """
    Helper function to create configs
    """
    def __init__(
            self,
            system_instruction: str = None,  # overriding personality
            tools: List[str] = None,  # tools like maps, google search, url, etc.; or even user created function
            tool_config_mode: types.FunctionCallingConfigMode = None,  # the mode of config
            allowed_tool_for_config: List[str] = None,  # lst of stuff effected by config_mode
            safety_setting: Dict[types.HarmCategory, types.HarmBlockThreshold] = None,  # safety settings
            temperature: float = None,  # temperature (randomness)
            top_p: float = None,  # considers the smallest set of top tokens whose cumulative probability is p
            top_k: float = None,  # model only considers the top K most tokens
            max_output_tokens: int = None,  # max tokens that will be consumed by output,
            response_mime_type: str = "text/plain",  # the kind of response
    ):
        # Implement thinking level can be used, but it is recommended to get high in 3.0 (as it states it is prone to
        # error below that

        # safety is a list of dict with key being the category and the value being the threshold (there are other
        # measures, but I have ignored them (ignoring safety rating and harm probability for now)

        safety_setting_collected = None

        if safety_setting is not None:
            safety_setting_collected = []
            for category, threshold in safety_setting.items():
                if category not in VALID_SAFETY_VALUES:
                    raise ValueError(f"{category} is not a valid safety category (use uppercase)")
                if threshold not in SAFETY_THRESHOLDS:
                    raise ValueError(f"{threshold} is not a valid safety threshold (use uppercase)")
                safety_setting_collected.append(
                    types.SafetySetting(category=category, threshold=threshold)
                )

        # check toolconfig options - you can have only one
        if tool_config_mode is None and allowed_tool_for_config is not None:
            raise ValueError(
                f"You need to provide a tool config mode if you want to use tool_config (maybe use all uppercase)"
            )
        if tool_config_mode is not None and tool_config_mode not in TOOL_CONFIG_VALUES:
            raise ValueError(f"{tool_config_mode} Tool Config mode is invalid (maybe use all uppercase)")

        config_kwargs = {}
        if temperature is not None:
            config_kwargs["temperature"] = temperature
        if top_p is not None:
            config_kwargs["top_p"] = top_p
        if top_k is not None:
            config_kwargs["top_k"] = top_k
        if max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = max_output_tokens
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
                function_calling_config=types.FunctionCallingConfig(mode=tool_config_mode)
            )
        if tool_config_mode is not None and allowed_tool_for_config is not None:
            config_kwargs["tool_config"] = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=tool_config_mode,
                    allowed_function_names=allowed_tool_for_config,
                )
            )

        self.config = types.GenerateContentConfig(**config_kwargs)

        # for using later if needed
        self.tools = tools
        self.response_mime_types = response_mime_type

    def get_config_data(self):
        """
        Returns config data
        :return: config data for AI
        """
        return self.config


class GeminiManager:
    """
    Manages Gemini Stuff
    """
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key is None:
            raise ValueError("API KEY not found!")
        self.client = genai.Client(api_key=api_key)
        self.agents = {}

        with open("models.yaml", "r") as fh:
            models = yaml.safe_load(fh)
            self.valid_models = models["gemini"]

    def create_agent(self, agent_name: str, model_name: str, config:GeminiConfig = None):
        """
        Creates the agent (chat)

        :param agent_name: Name of agent
        :param model_name: Model to use
        :param config: Gemini config
        """
        if agent_name in self.agents:
            raise ValueError(f"{agent_name} already exists!")
        elif model_name not in self.valid_models:
            raise ValueError(f"{model_name} is not valid!")

        if config is None:
            chat = self.client.chats.create(model=model_name)
        else:
            chat = self.client.chats.create(model=model_name, config=config.get_config_data())
        agent = GeminiAgent(agent_name, model_name, chat)

        self.agents[agent_name] = agent

        return agent

    def add_info(self, agent, info: str, info_name: str = "info"):
        """
        Allows to add stuff the agent must know - like say some other agent's responses (for example)

        :param agent: The agent where you want to add context
        :param info: The extra context
        :param info_name: Name of info (say some agent's name or something) - default = "info"
        """
        if agent.name not in self.agents:
            raise ValueError(f"{agent.name} doesn't exist!")

        agent.add_context(info_name, info)

    @staticmethod
    def _build_prompt(agent: GeminiAgent, prompt: str):
        """
        Adds context and message - using gemini's suggested way to add it now

        :param agent: Name of Agent
        :param prompt: Prompt (can be "")
        """

        full_context = "\n".join(agent.context)
        if full_context:
            return f"Context:\n{full_context}\n\nUser Query:\n{prompt}".strip()
        return prompt

    def send_message(self, agent: GeminiAgent, prompt_str: str, print_response: bool = True):
        """
        Sends prompt over to the servers

        :param agent: The agent
        :param prompt_str: Message to be sent ("" if it just needs to feed off context)
        :param print_response: Do you want it to print responses
        """
        if agent.name not in self.agents:
            raise ValueError(f"{agent.name} doesn't exist")

        prompt = self._build_prompt(agent, prompt_str)
        if print_response:
            cprint(f"Input: {prompt}\n\n", "green")
        cprint("Output:", "blue")

        response = ""

        st_time = time.time()
        # try 3 times and then give up
        for i in range(1,4):
            try:
                response = ""
                for chunk in agent.chat_obj.send_message_stream(prompt):
                    response += chunk.text
                    if print_response:
                        cprint(chunk.text, end = "", flush = True, color = "blue")
                agent.record_history(prompt, response)
                agent.reset_context()
                if print_response:
                    cprint(f"\n\nTime Elapsed: {(time.time() - st_time):.2f} seconds\n", "yellow")  # newline
                return response
            except Exception as e:
                cprint(f"Some Exception {e}", "red")
                time.sleep(20 * 2 ** i)  # have 25 seconds as my Wi-Fi drops sometimes - thus makes sure it persists
                continue

        raise RuntimeError("Couldn't connect with the server or timeout")

    def close_client(self, project_name: str = ""):
        for agent in self.agents.values():
            agent.dump_history(project_name=project_name)
        self.client.close()


if __name__ == "__main__":
    # Prepare two configs and two agents, then send one message to each (no error catching)
    cfg_poet = GeminiConfig(
        system_instruction="You are a lyrical poet. Keep responses short.",
        temperature=0.7,
        max_output_tokens=200,
    )

    cfg_helper = GeminiConfig(
        system_instruction="You are a concise technical assistant.",
        temperature=0.2,
        max_output_tokens=300,
    )

    mgr = GeminiManager()

    # create two agents with configs
    poet_agent = mgr.create_agent("poet", "gemini-2.5-flash", config=cfg_poet)
    helper_agent = mgr.create_agent("helper", "gemini-2.5-flash", config=cfg_helper)

    # add distinct context for each
    mgr.add_info(poet_agent, "The user loves robots and sonnets.", "Background")
    mgr.add_info(helper_agent, "Prefer short, actionable answers.", "Guideline")

    # send one message to each agent and print responses
    resp_poet = mgr.send_message(poet_agent, "Write a two-line poem about robots.", print_response=True)
    print("\nPoet response:\n", resp_poet)

    resp_helper = mgr.send_message(helper_agent, "How to optimize a Python loop?", print_response=True)
    print("\nHelper response:\n", resp_helper)

    # close and persist histories
    mgr.close_client()
