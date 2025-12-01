"""
Class to make gemini chat work

Not making it asynchronous as even in multi-agent stuff, I want the stuff to be synchronous.

TODO: Add a rate limiter or something maybe?
TODO: Maybe more colourful cli output
TODO: Calculate input and output token values
TODO: Add config for agents
TODO: Allow tool access
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
from db_manager import dump_history


# consts
VALID_SAFETY_VALUES = [
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
    "HARM_CATEGORY_CIVIC_INTEGRITY"
]  # these 5 are the only supported ones for gemini
SAFETY_THRESHOLDS = [
    "BLOCK_LOW_AND_ABOVE",
    "BLOCK_MEDIUM_AND_ABOVE",
    "BLOCK_ONLY_HIGH",
]
TOOL_CONFIG_VALUES = [
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
        self.context = []

    def add_context(self, info_name: str, info: str):
        """
        Adds context within context

        :param info_name: change to another agents response if you want to add it (for example)
        :param info: the main part of the thing
        """
        self.context.append(f"[{info_name}]{info}")

    def record_history(self, prompt: str, response: str):
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
        # dump history
        dump_history(self.history, project_name=project_name)
        self.history = []

class GeminiConfig:
    """
    Helper function to create configs
    """
    def __init__(
            self,
            system_instruction = None,  # overriding personality
            tools = None,  # tools like maps, google search, url, etc.; or even user created function
            tool_config_mode = None,  # the mode of config
            allowed_tool_for_config = None,  # lst of stuff effected by config_mode
            safety_setting = None,  # safety settings
            temperature = None,  # temperature (randomness)
            top_p = None,  # considers the smallest set of top tokens whose cumulative probability is p
            top_k = None,  # model only considers the top K most tokens
            max_output_tokens = None,  # max tokens that will be consumed by output,
            response_mime_type = "text/plain",  # the kind of response
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

    def add_info(self, agent_name, info: str, info_name: str = "info"):
        """
        Allows to add stuff the agent must know - like say some other agent's responses (for example)

        :param agent_name: Name of agent the context must be known to
        :param info: The extra context
        :param info_name: Name of info (say some agent's name or something) - default = "info"
        """
        if agent_name not in self.agents:
            raise ValueError(f"{agent_name} doesn't exist!")

        self.agents[agent_name].add_context(info_name, info)

    @staticmethod
    def _build_prompt(agent: GeminiAgent, prompt: str):
        """
        Adds context and message - using gemini's suggested way to add it now
        TODO: Add a char limit and stuff so that I don't overwhelm my usage

        :param agent: Name of Agent
        :param prompt: Prompt (can be "")
        """

        full_context = "\n".join(agent.context)
        if full_context:
            return f"Context:\n{full_context}\n\nUser Query:\n{prompt}".strip()
        return prompt

    def send_message(self, agent_name: str, prompt_str: str, print_response: bool = True):
        """
        Sends prompt over to the servers

        :param agent_name: Name of the agent
        :param prompt_str: Message to be sent ("" if it just needs to feed off context)
        :param print_response: Do you want it to print responses
        """
        if agent_name not in self.agents:
            raise ValueError(f"{agent_name} doesn't exist")

        agent = self.agents[agent_name]
        prompt = self._build_prompt(agent, prompt_str)
        if print_response:
            print(f"Input: {prompt}\n\nOutput:")

        response = ""

        st_time = time.time()
        # try 3 times and then give up
        for i in range(1,4):
            try:
                response = ""
                for chunk in agent.chat_obj.send_message_stream(prompt):
                    response += chunk.text
                    if print_response:
                        print(chunk.text, end="", flush=True)
                agent.record_history(prompt, response)
                agent.reset_context()
                if print_response:
                    print(f"\nTime Elapsed: {(time.time() - st_time):.2f} seconds\n")  # newline
                return response
            except Exception as e:
                print(f"Some Exception {e}")
                time.sleep(20 * 2 ** i)  # have 25 seconds as my Wi-Fi drops sometimes - thus makes sure it persists
                continue

        raise RuntimeError("Couldn't connect with the server or timeout")

    def close_client(self, project_name=""):
        for agent in self.agents.values():
            agent.dump_history(project_name=project_name)
        self.client.close()


if __name__ == "__main__":
    # test
    try:
        mgr = GeminiManager()
        mgr.create_agent("poet", "gemini-2.5-flash")  # Example model

        # Test context
        mgr.add_info("poet", "The user loves robots.", "Background")

        mgr.send_message("poet", "Death is the best!.")
    except Exception as e:
        print(f"Fatal Error: {e}")
    finally:
        if 'mgr' in locals():
            mgr.close_client()
