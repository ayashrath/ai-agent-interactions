# AI Interactions

I was inspired by [DougDoug's](https://www.youtube.com/@DougDoug) AI videos, and wanted to make something similar.
Thus, I am making this. Shall add more detail on the project as I move further.

Currently the scripts are able to provide ability to use Gemini API to have multi-agent conversation interactions.

## The files

```
├── data  # shall be created by default when you run the scripts
│   └── chat_history.sqlite  # the sqlite db where the data is stored
├── game_samples  # sample code provided on how the tool can be used - three scenarios
│   ├── 01_simple_chatbot  
│   ├── 02_james_and_john
│   │   ├── james.toml
│   │   ├── johns.toml
│   │   ├── main.py
│   │   └── README.md
│   └── 03_board_discussion
│       └── README.md
├── README.md
├── requirements.txt
├── setup.py
└── src
    ├── ai_manager.py  # unifies the interface with all the different ai managers
    ├── db_manager.py  # the interface to the db
    ├── gemini_manager.py  # utilises the gemini api and provided an interface
    ├── __init__.py
    └── text_to_speech_manager.py  # the azure text-to-speech (tts) interace
```

## Setup

- Install Python Packages within a virtual environemnt
- Set up environment variables


## Future Plan

- Create an GUI to view the conversations of the agents
- Add support for 
  - OpenAI
  - Claude
  - DeepSeek
  - Grok
  - Meta
- Make the code good

## Note

- There is no definite timeline for me to add new features
- The scripts will break as the APIs are always changing. I am not a commitment to maintain these - it just an interesting thing I wanted to make.