"""
The DB Manager
"""

from datetime import datetime
import sqlite3
import pathlib

class DBManager:
    """
    The DB Manager class
    """
    def __init__(self, table_name: str = "", db: str = "chat_history.sqlite"):
        file_dir_path = str(pathlib.Path(__file__).parent.resolve())
        self.conn = sqlite3.connect(file_dir_path + "/../data/" + db)
        self.cur = self.conn.cursor()

        if table_name == "":
            self.table_name = "default_chat"  # not unique, debug stuff
        else:
            table_name = table_name.replace("-", "_").replace(" ", "_") # as I usually forget
            self.table_name = table_name

    def create_table(self):
        """
        Create the table if it does not exist
        """
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

    def manual_entry(self, name: str, message: str):
        """
        Insert a manual entry into the database
        :param name: Name of the entry
        :param message: Message of the entry
        """
        timestamp = datetime.now().isoformat()
        self.cur.execute(
            f"""
            INSERT INTO {self.table_name}
            VALUES (?, ?, ?, ?, ?)
            """, (
                timestamp,
                "manual",
                name,
                message,
                ""
            )
        )


    def insert_history(self, history):
        """
        Insert history into the database
        :param history: History - List of dict with keys - (timestamp, model, name, prompt, response)
        """
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
        """
        Close the database connection
        """
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

def manual_db_entry(name: str, message: str, project_name = ""):
    """
    Insert a manual entry into the database

    :param name: Name of the entry
    :param message: Message of the entry
    :param project_name: Name of project (table name for table)
    """
    dbmanager = DBManager(project_name)
    dbmanager.create_table()
    dbmanager.manual_entry(name, message)
    dbmanager.close()
