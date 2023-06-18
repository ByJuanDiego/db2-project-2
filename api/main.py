import psycopg2
from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post('/obtener_datos')
async def obtener_datos(data: dict) ->dict:
    Q = data.get('query')
    k = int(data.get('k'))
    conn = psycopg2.connect(
        database='db2project2',
        port='5432',
        user='postgres',
        password='postgres'
    )

    cursor = conn.cursor()
    postgreSQL_select = \
            f"SELECT title, abstract, ts_rank(indexed, query) rank\
             FROM documents, phraseto_tsquery('english', '{Q}') query\
             WHERE indexed @@ query ORDER BY rank DESC LIMIT {k};"
    
    cursor.execute(postgreSQL_select)
    response = cursor.fetchall()

    conn.close()
    cursor.close()
    return {'data' : response}

@app.get('/topkinvindex')
async def get_top_k_from_index(Q: str, k : int) -> dict:
    pass
