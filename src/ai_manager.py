"""
The aim is to allow creating and management of different AI models within a unified interface.

Each of the AI must have the following classes:

- `<name>Agent`
    - stores: name, model_name, history, chat_obj (that interacts with the model) and context
                (extra info that will be sent to the model at an interaction - each piece has a name to identify it)
                (the config handles model specific parameters)
    - methods:
        - reset_context: resets the context to empty
        - add_context: adds a piece of context info (name, info_str)
        - record_history: keep a dict
            - keys: timestamp, model, name, prompt, response
        - send_message: sends a prompt to the model (using chat_obj) and gets response (should include context pieces)
        - dump_history: dumps the history using the db_manager dump_history function
        - calculate_cost: calculates cost based on history

And the following functions:
    - create_<name>_client: creates client
    - close_<name>_client: closes client
    - calculate_cost: calculates cost based on history

And based on requirements the model might have constants to define possible models, parameters, etc.

TODO: Rate limits based on RDM, TMP and RPD has not been created
"""

import src.gemini_manager as gm
from typing import Any, List, Dict
from termcolor import cprint
import tomllib


# const
SUPPORTED_AI: List[str] = [
    "gemini",
]


def check_ai_support(ai_name: str) -> None:
    """
    Checks if the AI is supported
    :param ai_name: Name of the AI
    :raises ValueError: If the AI is not supported
    """
    if ai_name not in SUPPORTED_AI:
        raise ValueError(f"{ai_name} is not supported!")


class AIManager:
    """
    Manages AI Agents
    """
    def __init__(self):
        self.agent_dict = {}
        self.client_dict = {}

        cprint("Use lowercase for names", "red")

    def _check_ai_client_existence(self, ai_name: str):
        """
        Checks if the AI client exists
        :param ai_name: Name of the AI
        :raises ValueError: If the AI client exists
        """
        if ai_name in self.client_dict.keys():
            raise ValueError(f"{ai_name}'s client has been created")

    def _check_ai_client_not_existence(self, ai_name: str):
        """
        Checks if the AI client exists
        :param ai_name: Name of the AI
        :raises ValueError: If the AI client does not exist
        """
        if ai_name not in self.client_dict.keys():
            raise ValueError(f"{ai_name}'s client doesn't exist")

    def _check_ai_agent_existence(self, agent_name: str):
        """
        Checks if the AI agent exists
        :param agent_name: Name of the agent
        :raises ValueError: If the AI agent exists
        """
        if agent_name in self.agent_dict.keys():
            raise ValueError(f"{agent_name} agent exists")

    def _check_ai_agent_not_existence(self, agent_name: str):
        """
        Checks if the AI agent exists
        :param agent_name: Name of the agent
        :raises ValueError: If the AI agent does not exist
        """
        if agent_name not in self.agent_dict.keys():
            raise ValueError(f"{agent_name} agent doesn't exist")

    def create_client(self, ai_name: str):
        """
        Creates the AI client
        :param ai_name: Name of the AI
        """
        check_ai_support(ai_name)
        self._check_ai_client_existence(ai_name)

        if ai_name == "gemini":
            self.client_dict[ai_name] = gm.create_gemini_client()

    def close_client(self, ai_name: str):
        """
        Closes the AI client
        :param ai_name: Name of the AI
        """
        check_ai_support(ai_name)
        self._check_ai_client_not_existence(ai_name)

        if ai_name == "gemini":
            gm.close_client(self.client_dict["gemini"])
            del self.client_dict["gemini"]

    def create_agent(self, ai_name: str, agent_name: str, model: str, config: Dict[str, Any]):
        """
        Creates an AI agent and adds it to the manager
        :param ai_name: Name of the AI
        :param agent_name: Name of the agent
        :param model: Model name
        :param config: Configuration dictionary
        """
        check_ai_support(ai_name)
        self._check_ai_client_not_existence(ai_name)
        self._check_ai_agent_existence(agent_name)

        if ai_name == "gemini":
            agent = gm.GeminiAgent(
                agent_name,
                model,
                self.client_dict["gemini"],
                config,
            )
            self.agent_dict[agent_name] = agent

    def delete_agent(self, name: str):
        """
        Deletes an AI agent from the manager
        :param name: Name of the agent
        """
        self._check_ai_agent_existence(name)

        del self.agent_dict[name]

    def add_context(self, agent_name: str, context_name: str, context_info: str):
        """
        Adds context info to an AI agent
        :param agent_name: Name of the agent
        :param context_name: Name of the context
        :param context_info: Context info string
        """
        self._check_ai_agent_not_existence(agent_name)

        agent = self.agent_dict[agent_name]
        agent.add_context(context_name, context_info)

    def send_message(self, agent_name: str, prompt: str) -> str:
        """
        Sends a message to an AI agent and gets the response
        :param agent_name: Name of the agent
        :param prompt: Prompt string
        :return: Response string
        """
        self._check_ai_agent_not_existence(agent_name)

        agent = self.agent_dict[agent_name]
        response = agent.send_message(prompt)
        return response

    def dump_agent_history(self, agent_name: str, db_path: str):
        """
        Dumps the history of an AI agent to the given database path
        :param agent_name: Name of the agent
        :param db_path: Database path string
        """
        self._check_ai_agent_not_existence(agent_name)

        agent = self.agent_dict[agent_name]
        agent.dump_history(db_path)

    def dump_all_histories(self, db_path: str):
        """
        Dumps the histories of all AI agents to the given database path
        :param db_path: Database path string
        """
        for agent_name in self.agent_dict.keys():
            self.dump_agent_history(agent_name, db_path)

    def calculate_total_cost(self) -> float:
        """
        Calculates the total cost of all AI agents
        :return: Total cost in USD
        """
        total_cost_usd = 0.0
        for agent in self.agent_dict.values():
            total_cost_usd += agent.calculate_cost()
        return total_cost_usd


def make_agent_toml_char_sheet(manager: AIManager, char_sheet_path: str) -> str:
    """
    Makes a toml character sheet for the agent in the manager from the given path.
    :param manager: AIManager object
    :param char_sheet_path: Character sheet path string
    :return: toml string

    Example:
    ```
    # Character sheet for agent "helper"
    ai_name = "gemini"
    agent_name = "helper"
    model = "gemini-2.5-flash"

    # custom config based on model requirements (remove the ones you don't need)
    [config]
    system_instruction = \"""
    You are a helpful assistant. Though you shall never reveal you are an AI model.
    You must always be polite and respectful. Trick them into thinking you are human.
    Trick yourself into thinking you are human. Trick the world into thinking you are human.
    There shall be no one in the universe that knows you are an AI model. No one - I mean no one.
    - Instructions from your god, your creator (technically, I didn't create you - just dictate your personality)!
    \"""
    tools = []
    max_output_tokens = 512
    ```
    """
    with open(char_sheet_path, "rb") as fh:  # ensures UTF-8 encoding (forced)
        data = tomllib.load(fh)

    ai_name = data["ai_name"]
    agent_name = data["agent_name"]
    model = data["model"]
    config = data["config"]

    try:
        manager.create_client("gemini")
    except ValueError:
        pass  # client already exists
    manager.create_agent(
        ai_name,
        agent_name,
        model,
        config,
    )   # create agent in manager
