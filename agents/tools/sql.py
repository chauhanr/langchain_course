import sqlite3 
from langchain.tools import Tool 

# Tool to run a sql when gpt model requests for the query to be run. 

conn = sqlite3.connect("db.sqlite")
def run_sql(sql):
    cursor = conn.cursor()
    cursor.execute(sql)
    conn.commit()
    return cursor.fetchall()

run_query_tool = Tool.from_function(
    name="run_sqlite_query",
    description="Run a query on the sqlite database",
    func=run_sql
)
