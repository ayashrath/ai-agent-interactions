"""
Class to make gemini chat work

TODO: Add a rate limiter or something maybe?
"""

from dotenv import load_dotenv
from google import genai
from google.genai.chats import AsyncChat
import os
import asyncio


class GeminiAgent:
    """
    A single agent (chat) struct

    name: name of the agent
    model_name: model to use
    chat: the AsyncChat object
    """
    def __init__(self, name: str, model_name: str, chat: AsyncChat):
        self.name = name
        self.model_name = model_name
        self.history = {}
        self.agent = chat


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

    async def create_agent(self, agent_name: str, model_name: str):
        """
        Creates the agent (chat)

        :param agent_name: Name of agent
        :param model_name: Model to use
        """
        if agent_name in self.agents:
            raise ValueError(f"{agent_name} already exists!")
        agent = self.client.aio.chats.create(model=model_name)
        agent = GeminiAgent(agent_name, model_name, agent)

        self.agents[agent_name] = agent

    async def add_info(self, agent_name, message: str):
        pass  # make it such that it provides info to the agent, just to provide info - that's all!

    async def send_message(self, agent_name: str, message: str, print_response: bool = True):
        if agent_name not in self.agents:
            raise ValueError(f"{agent_name} doesn't exist")

        agent = self.agents[agent_name]

        response = ""

        async for chunk in await agent.agent.send_message_stream(message):
            response += chunk.text
            if print_response:
                print(chunk.text, end="", flush=True)

        if print_response:
            print()  # newline

        self.agents[agent_name].history[message] = response

    def close_client(self):
        self.client.close()

if __name__ == "__main__":
    async def run_tests():
        manager = GeminiManager()
        print("Creating agents...")
        await manager.create_agent("story_writer", "gemini-2.5-flash")
        await manager.create_agent("code_reviewer", "gemini-2.5-flash")
        print("Agents created.")
        print("-" * 20)

        await manager.send_message("story_writer", "Tell me a short story about a brave knight within 10 words.")
        print("-" * 20)

        await manager.send_message("code_reviewer", "What is a python list comprehension? Give a short example. Within 10 words.")
        print("-" * 20)

        await manager.send_message("story_writer", "What was the knight's name? Within 10 words!")
        print("-" * 20)

    asyncio.run(run_tests())
