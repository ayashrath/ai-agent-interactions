"""
The James and John Game
"""

import src.ai_manager as ai
from termcolor import cprint
import pathlib

manager = ai.AIManager()
project_name = "james_and_johns"

# add characters
current_dir_path = str(pathlib.Path(__file__).parent.resolve())
ai.make_agent_toml_char_sheet(manager, current_dir_path + "/james.toml")
ai.make_agent_toml_char_sheet(manager, current_dir_path + "/johns.toml")

print("Starting multi-agent interaction...\n")
print("\nPoet (James):\n")

resp_james = manager.send_message("james", "To impress them, you decided to recite a poem you made on the spot.")

counter = 1
while True:
    print(f"--- Interaction Round {counter} ---\n")
    print("James:\n")
    manager.add_context("johns", "James's Response", resp_james)
    resp_john = manager.send_message("johns", "")
    manager.add_context("james", "Johns's Response", resp_john)

    total_cost = manager.calculate_total_cost()
    cprint(f"\nTotal Money ${total_cost:.6f}\n", "yellow")

    if counter % 1 == 0:
        manager.dump_all_histories(project_name)
        inp = input("Continue for 1 more turns? (y/n): ")
        if inp.lower() != "y":
            break
    counter += 1

manager.dump_all_histories(project_name)
manager.close_client("gemini")
