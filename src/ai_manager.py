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

And the following functions:
    - create_<name>_client: creates client
    - close_<name>_client: closes client

And based on requirements the model might have constants to define possible models, parameters, etc.
"""

import gemini_manager as gm
from typing import Any, List, Dict
from termcolor import cprint

# const
SUPPORTED_AI = [
    "gemini",
]


def check_ai_support(ai_name: str):
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
        if ai_name not in self.client_dict.keys():
            raise ValueError(f"{ai_name}'s client has not been created")

    def _check_ai_agent_existence(self, agent_name: str):
        if agent_name not in self.agent_dict.keys():
            raise ValueError(f"{agent_name} agent does not exist")

    def _check_ai_agent_not_existence(self, agent_name: str):
        if agent_name in self.agent_dict.keys():
            raise ValueError(f"{agent_name} agent already exists")

    def create_client(self, ai_name: str):
        check_ai_support(ai_name)
        self._check_ai_client_existence(ai_name)

        if ai_name == "gemini":
            self.client_dict[ai_name] = gm.create_gemini_client()

    def close_client(self, ai_name: str):
        check_ai_support(ai_name)
        self._check_ai_client_existence(ai_name)

        if ai_name == "gemini":
            gm.close_client(self.client_dict["gemini"])
            del self.client_dict["gemini"]

    def create_agent(self, ai_name: str, name: str, model: str, config: Dict[str, Any]):
        check_ai_support(ai_name)
        self._check_ai_client_existence(ai_name)
        self._check_ai_agent_not_existence(name)

        if ai_name == "gemini":
            agent = gm.GeminiAgent(
                name,
                model,
                self.client_dict["gemini"],
                config,
            )
            self.agent_dict[name] = agent

    def delete_agent(self, name: str):
        self._check_ai_agent_existence(name)

        del self.agent_dict[name]

    def add_context(self, agent_name: str, context_name: str, context_info: str):
        self._check_ai_agent_existence(agent_name)

        agent = self.agent_dict[agent_name]
        agent.add_context(context_name, context_info)

    def send_message(self, agent_name: str, prompt: str) -> str:
        self._check_ai_agent_existence(agent_name)

        agent = self.agent_dict[agent_name]
        response = agent.send_message(prompt)
        return response

    def dump_agent_history(self, agent_name: str, db_path: str):
        self._check_ai_agent_existence(agent_name)

        agent = self.agent_dict[agent_name]
        agent.dump_history(db_path)

    def dump_all_histories(self, db_path: str):
        for agent_name in self.agent_dict.keys():
            self.dump_agent_history(agent_name, db_path)

