"""
Class to make gemini chat work

Not making it asynchronous as even in multi-agent stuff, I want the stuff to be synchronous

TODO: Add a rate limiter or something maybe?
TODO: Maybe more colourful cli output
"""

from dotenv import load_dotenv
from google import genai
from google.genai.chats import Chat
import yaml
import os
import asyncio
import time
from datetime import datetime


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
        self.history.append({
            "timestamp": timestamp,
            "model": self.model_name,
            "name": self.name,
            "prompt": prompt,
            "response": response,
        }
        )

    def dump_history(self):
        # dump history
        self.history = []

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

    def create_agent(self, agent_name: str, model_name: str):
        """
        Creates the agent (chat)

        :param agent_name: Name of agent
        :param model_name: Model to use
        """
        if agent_name in self.agents:
            raise ValueError(f"{agent_name} already exists!")
        elif model_name not in self.valid_models:
            raise ValueError(f"{model_name} is not valid!")
        chat = self.client.chats.create(model=model_name)
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

        self.agents[agent_name].add_context(info, info_name)

    @staticmethod
    def _build_prompt(self, agent: GeminiAgent, prompt: str):
        """
        Adds context and message
        TODO: Add a char limit and stuff so that I don't overwhelm my usage

        :param agent: Name of Agent
        :param prompt: Prompt (can be "")
        """

        if prompt != "":
            prompt = "\n\n".join(agent.context) + "\n\n[user]" + prompt
        else:
            prompt = "\n\n".join(agent.context)

        return prompt

    def send_message(self, agent_name: str, prompt: str, print_response: bool = True):
        """
        Sends prompt over to the servers

        :param agent_name: Name of the agent
        :param prompt: Message to be sent ("" if it just needs to feed off context)
        :param print_response: Do you want it to print responses
        """
        if agent_name not in self.agents:
            raise ValueError(f"{agent_name} doesn't exist")

        agent = self.agents[agent_name]
        prompt = self._build_prompt(agent, prompt)
        agent.reset_context()
        if print_response:
            print(f"Input: {prompt}\n\nOutput:")

        response = ""

        st_time = time.time()
        # try twice and then give up
        for i in range(1,4):
            try:
                response = ""
                for chunk in agent.chat_obj.send_message_stream(prompt):
                    response += chunk.text
                    if print_response:
                        print(chunk.text, end="", flush=True)
                break
            except Exception as e:
                print(f"Some Exception {e}")
                time.sleep(15 * i)
                continue

        if response == "":
            raise TimeoutError("Couldn't connect with the server")

        if print_response:
            print(f"\nTime Elapsed: {(time.time() - st_time):.2f} seconds\n")  # newline

        agent.record_history(prompt, response)
        return response

    def close_client(self):
        self.client.close()


if __name__ == "__main__":
    # Example synchronous usage for `src/gemini_manager.py`
    manager = None
    try:
        manager = GeminiManager()
        print("Creating agents...")
        manager.create_agent("story_writer", "gemini-2.5-flash")
        manager.create_agent("code_reviewer", "gemini-2.5-flash")
        print("Agents created.")
        print("-" * 40)

        manager.send_message("story_writer", "Tell me a short story about a brave knight within 10 words.")
        print("-" * 40)

        manager.send_message("code_reviewer", "What is a python list comprehension? Give a short example within 10 words.")
        print("-" * 40)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if manager:
            manager.close_client()
            print("Client closed.")
