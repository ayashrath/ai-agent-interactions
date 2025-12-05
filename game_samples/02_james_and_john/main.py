"""
The James and John Game
"""

import src.ai_manager as ai
import src.text_to_speech_manager as tts
from termcolor import cprint
import pathlib

ai_manager = ai.AIManager()
sound_manager = tts.TextToSpeechManager()
project_name = "james_and_johns"
voice_playback_rate = 2.0  # 1.0 is normal speed

# add characters
current_dir_path = str(pathlib.Path(__file__).parent.resolve())
ai.make_agent_toml_char_sheet(ai_manager, current_dir_path + "/james.toml")
ai.make_agent_toml_char_sheet(ai_manager, current_dir_path + "/johns.toml")
james_sound = "en-US-DerekMultilingualNeural"
johns_sound = "en-US-GuyNeural"

print("Starting multi-agent interaction...\n")
print("\nPoet (James):\n")

resp_james = ai_manager.send_message("james", "To impress them, you decided to recite a poem you made on the spot.")
sound_manager.convert_text_to_speech(resp_james, speech_voice=james_sound, rate=voice_playback_rate)

counter = 1
while True:
    print(f"--- Interaction Round {counter} ---\n")
    print("James:\n")
    ai_manager.add_context("johns", "James's Response", resp_james)
    resp_john = ai_manager.send_message("johns", "")
    sound_manager.convert_text_to_speech(resp_john, speech_voice=johns_sound, rate=voice_playback_rate)

    print("\nJohn:\n")
    ai_manager.add_context("james", "Johns's Response", resp_john)
    resp_james = ai_manager.send_message("james", "")
    sound_manager.convert_text_to_speech(resp_james, speech_voice=james_sound, rate=voice_playback_rate)

    total_cost = ai_manager.calculate_total_cost()
    cprint(f"\nTotal Money ${total_cost:.6f}\n", "yellow")

    if counter % 5 == 0:
        ai_manager.dump_all_histories(project_name)
        inp = input("Continue for 5 more turns? (y/n): ")
        if inp.lower() != "y":
            break
    counter += 1

ai_manager.dump_all_histories(project_name)
ai_manager.close_client("gemini")
