import psycopg2
from fastapi import FastAPI

app = FastAPI()

@app.get('/topkpostgres')
async def get_top_k_from_postgres(Q : str, k : int) -> dict:

    conn = psycopg2.connect(
        database='db2project2',
        port='5432',
        user='postgres',
        password='postgres'
    )

    cursor = conn.cursor()
    postgreSQL_select = \
            f"SELECT title, abstract, ts_rank_cd(indexed, query) rank\
             FROM documents, phraseto_tsquery('english', '{Q}') query\
             WHERE indexed @@ query ORDER BY rank DESC LIMIT {k};"
    
    cursor.execute(postgreSQL_select)
    response = cursor.fetchall()

    conn.close()
    cursor.close()
    return response

@app.get('/topkinvindex')
async def get_top_k_from_index(Q: str, k : int) -> dict:
    pass
