"""
The aim is to allow creating and management of different AI models within a unified interface.

Each of the AI must have the following classes:

- `<name>Agent`
    - stores: name, model_name, history, chat_obj (that interacts with the model) and context
                (extra info that will be sent to the model at an interaction - each piece has a name to identify it)
    - methods:
        - reset_context: resets the context to empty
        - add_context: adds a piece of context info (name, info_str)
        - record_history: keep a dict
            - keys: timestamp, model, name, prompt, response
        - send_message: sends a prompt to the model (using chat_obj) and gets response (should include context pieces)
        - dump_history: dumps the history using the db_manager dump_history function
- `<name>Config`
    - stores: stores the config parameters for the agent
    - methods:
        - get_config_data: returns the config data as a dict or appropriate data structure (based on the lib requirement)

And the following functions:
    - create_<name>_client: creates client
    - close_<name>_client: closes client

And based on requirements the model might have constants to define possible models, parameters, etc.
"""


