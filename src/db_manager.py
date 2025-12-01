"""
The DB Manager
"""

import time
import sqlite3

class DBManager:
    def __init__(self, table_name: str = "", db: str = "chat_history.sqlite"):
        self.conn = sqlite3.connect("data/" + db)
        self.cur = self.conn.cursor()

        if table_name == "":
            self.table_name = "default_chat"
        else:
            self.table_name = table_name

    def create_table(self):
        self.cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                time    datetime PRIMARY KEY,
                model   text,
                name    text,
                prompt  text,
                response    text
            )
            """
        )

    def insert_history(self, history):
        for ind in range(len(history)):
            self.cur.execute(
                f"""
                INSERT INTO {self.table_name}
                VALUES (?, ?, ?, ?, ?)
                """, (
                    history[ind]["timestamp"],
                    history[ind]["model"],
                    history[ind]["name"],
                    history[ind]["prompt"],
                    history[ind]["response"]
                )
            )

    def close(self):
        self.conn.commit()
        self.conn.close()


def dump_history(history, project_name = ""):
    """
    Dump history stuff

    :param history: History - List of dict with keys - (timestamp, model, name, prompt, response)
    :param project_name: Name of project (table name for table)
    """
    dbmanager = DBManager(project_name)
    dbmanager.create_table()
    dbmanager.insert_history(history)
    dbmanager.close()
