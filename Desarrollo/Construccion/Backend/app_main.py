from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
import os
import re
import json
import pyodbc

# --- 2026 Langchain Modern Imports ---
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document
from groq import Groq

from schema_metadata import schema  
from sql_examples import sql_examples  

# Load environment variables dynamically
base_dir = Path(__file__).resolve().parent.parent
dotenv_path = base_dir / "Server" / ".env"
load_dotenv(dotenv_path=dotenv_path)

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SQL Server Config ---
SQL_SERVER_CONFIG = {
    "server": os.getenv("DB_SERVER"),
    "database": os.getenv("DB_DATABASE"),
    "username": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "driver": "{ODBC Driver 17 for SQL Server}",
}

class Query(BaseModel):
    prompt: str

# --- Schema Context Builders ---
def build_schema_context(user_query: str) -> str:
    relevant_tables = []
    for table, info in schema.items():
        if any(col.lower() in user_query.lower() for col in info["columns"].keys()):
            relevant_tables.append(table)
    if not relevant_tables:
        relevant_tables = list(schema.keys())

    return "\n".join(
        f"Table: {table}\nColumns: {', '.join(schema[table]['columns'].keys())}\n"
        for table in relevant_tables
    )

def build_detailed_schema_context(user_query: str) -> str:
    relevant_tables = []
    for table, info in schema.items():
        if any(col.lower() in user_query.lower() for col in info["columns"].keys()):
            relevant_tables.append(table)
    if not relevant_tables:
        relevant_tables = list(schema.keys())

    lines = []
    for table in relevant_tables:
        source = schema[table].get("source_system", "dbo")
        fq_table = f"{source}.{table}"
        lines.append(f"Tabla: {fq_table} - {schema[table]['description']}")
        for col_name, col_info in schema[table]['columns'].items():
            fq_col = f"{fq_table}.{col_name}"
            lines.append(f"  - {fq_col} ({col_info['type']}): {col_info['description']}")
        lines.append("")
    return "\n".join(lines)

# --- SQL Safety ---
def is_safe_sql(sql: str) -> bool:
    sql = sql.strip().lower()
    return sql.startswith("select") and not re.search(r'\b(update|delete|insert|drop|alter|truncate|create)\b', sql)

# --- LLM Setup ---
llm_sql = OllamaLLM(
    model="deepseek-r1:8b",
    model_kwargs={
        "temperature": 0,
        "top_p": 0.95,
        "num_predict": 1024,
        "repeat_penalty": 1.2,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
    }
)

llm_rag = OllamaLLM(
    model="deepseek-r1:8b"
)

# Groq client setup
groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key)
groq_model = "openai/gpt-oss-120b" # CORREGIDO: Se actualiaz el modelo porque el anterior esta deprecated

# --- SQL Generator: DeepSeek + Groq Cleaner ---
def generate_sql_query(natural_language_query: str, return_raw=False):
    schema_description = build_schema_context(natural_language_query)
    full_schema_description = build_detailed_schema_context(natural_language_query)
    examples_text = "\n\n".join(
        f"Pregunta: {ex['question']}\nSQL:\n{ex['query'].strip()}"
        for ex in sql_examples
    )

    ds_prompt = f"""
Eres un asistente que convierte preguntas en lenguaje natural a SQL Server válido.

INSTRUCCIONES:
- Devuelve SOLO código SQL válido.
- No incluyas explicaciones, etiquetas ni comentarios.

Esquema:
{schema_description}

Ejemplos:
{examples_text}

Pregunta: {natural_language_query}

SQL:
""".strip()

    try:
        raw_response = llm_sql.invoke(ds_prompt)
        raw_sql = raw_response.strip()
    except Exception as e:
        return ("SELECT 'DeepSeek error' AS error_message", "") if return_raw else "SELECT 'DeepSeek error' AS error_message"

    groq_prompt = f"""
Corrige y valida esta consulta SQL generada por otro modelo. Usa el esquema proporcionado.

Instrucciones:
- Devuelve un JSON con una sola clave "sql".
- Usa los nombres completos de tablas: esquema.nombre_tabla (por ejemplo, Gastos.FACT_GASTOS, Enapres.FACT_ENAPRES, ATM.FACT_ATM, Generico.DIM_FECHA, Generico.DIM_UBIGEO).
- Verifica que columnas y tablas existan y sean válidas según el esquema.

Consulta original:
{raw_sql}

Esquema detallado:
{full_schema_description}
""".strip()

    try:
        response = groq_client.chat.completions.create(
            model=groq_model,
            messages=[{"role": "user", "content": groq_prompt}],
            response_format={"type": "json_object"}
        )
        parsed_json = json.loads(response.choices[0].message.content)
        final_sql = parsed_json.get("sql", "").strip()

        if not re.match(r"(?i)^\s*(select|with)\b", final_sql):
            final_sql = f"SELECT * FROM {list(schema.keys())[0]} WHERE 1=0 -- SQL inválido"

        return (final_sql, raw_sql) if return_raw else final_sql

    except Exception as e:
        print("Groq JSON mode error:", e)
        return ("SELECT 'Groq error' AS error_message", raw_sql) if return_raw else "SELECT 'Groq error' AS error_message"


# --- SQL Execution ---
def execute_sql_query(query: str):
    conn_str = (
        f"DRIVER={SQL_SERVER_CONFIG['driver']};"
        f"SERVER={SQL_SERVER_CONFIG['server']};"
        f"DATABASE={SQL_SERVER_CONFIG['database']};"
        f"UID={SQL_SERVER_CONFIG['username']};"
        f"PWD={SQL_SERVER_CONFIG['password']}"
    )
    with pyodbc.connect(conn_str) as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return [dict(zip(columns, row)) for row in rows]

# --- RAG Setup (Modernizado a LCEL y Langchain_Chroma) ---
def fetch_data_from_sql():
    conn_str = (
        f"DRIVER={SQL_SERVER_CONFIG['driver']};"
        f"SERVER={SQL_SERVER_CONFIG['server']};"
        f"DATABASE={SQL_SERVER_CONFIG['database']};"
        f"UID={SQL_SERVER_CONFIG['username']};"
        f"PWD={SQL_SERVER_CONFIG['password']}"
    )

    queries = {
        "ATM_Contexto": """
            SELECT DISTINCT 
                ANIO, DEPARTAMENTO, PROVINCIA, ATM, ALCALDE, UNIDAD_ORGANICA 
            FROM [dbo].[Vista_ATM_Completa] 
            WHERE UNIDAD_ORGANICA IS NOT NULL 
              AND UNIDAD_ORGANICA NOT IN ('sin datos', 'NO INDICA')
        """,
        "Gastos_Proyectos": """
            SELECT DISTINCT 
                ANIO, DEPARTAMENTO, PROVINCIA, DISTRITO, 
                EJECUTORA_NOMBRE, 
                PRODUCTO_PROYECTO_NOMBRE, 
                ACTIVIDAD_ACCION_OBRA_NOMBRE, 
                META_NOMBRE 
            FROM [dbo].[Vista_Gastos_Completa] 
            WHERE PRODUCTO_PROYECTO_NOMBRE != 'SIN PRODUCTO'
              AND PRODUCTO_PROYECTO_NOMBRE IS NOT NULL
        """
    }



    documents = []
    with pyodbc.connect(conn_str) as conn:
        cursor = conn.cursor()
        for name, sql in queries.items():
            print(f">>> Cargando datos de: {name}")
            cursor.execute(sql)
            cols = [col[0] for col in cursor.description]
            for row in cursor.fetchall():
                row_dict = dict(zip(cols, row))
                # MEJORA SILENCIOSA: Formato de texto semántico en lugar de JSON duro para mejores embeddings
                semantic_text = ", ".join([f"{k}: {v}" for k, v in row_dict.items() if v is not None])
                documents.append(Document(page_content=semantic_text, metadata={"source": name}))
    return documents

def setup_vectorstore(documents, persist_directory="chroma_db"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_path = Path(persist_directory)
    if persist_path.exists() and not documents:
        print(">>> Cargando vectorstore existente")
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    print(">>> Creando nuevo vectorstore")
    return Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)

def create_rag_pipeline(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # MEJORA SILENCIOSA: LCEL Chain, reemplazando el deprecado RetrievalQA
    prompt_template = ChatPromptTemplate.from_template(
        "Eres un experto en gestión de recursos hídricos en el Perú. Usa el siguiente contexto para responder la pregunta de forma clara y profesional.\n\nContexto:\n{context}\n\nPregunta: {input}\n\nRespuesta:"
    )
    combine_docs_chain = create_stuff_documents_chain(llm_rag, prompt_template)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    return retrieval_chain

# --- Init pipeline
rag_pipeline = None
VECTORSTORE_PATH = "chroma_db"

def initialize_rag_pipeline():
    global rag_pipeline
    if not Path(VECTORSTORE_PATH).exists():
        print(">>> Primera vez: cargando datos SQL y creando vectorstore...")
        docs = fetch_data_from_sql()
        vectorstore = setup_vectorstore(docs, persist_directory=VECTORSTORE_PATH)
    else:
        vectorstore = setup_vectorstore([], persist_directory=VECTORSTORE_PATH)

    rag_pipeline = create_rag_pipeline(vectorstore)

if __name__ == "__main__" or os.getenv("RUNNING_TESTS") != "1":
    initialize_rag_pipeline()

# --- Endpoints ---
@app.post("/ask")
async def chat_endpoint(query: Query):
    try:
        # LCEL usa 'input' en lugar de 'query'
        result = rag_pipeline.invoke({"input": query.prompt})
        response_text = result.get("answer") if isinstance(result, dict) else str(result)
        return {"response": response_text}
    except Exception as e:
        return {"error": f"Error procesando consulta RAG: {str(e)}"}

@app.post("/query")
async def sql_query_endpoint(query: Query):
    try:
        sql, raw_sql = generate_sql_query(query.prompt, return_raw=True)

        if re.search(r'<[^<>]+>', sql):
            return {
                "error": "La consulta generada contiene placeholders con '<...>'. Reformula la pregunta.",
                "sql": sql,
                "raw_sql": raw_sql
            }

        sql = re.sub(
            r"(?i)\b([A-Za-z0-9_]+\.)?ANIO\s*=\s*(\d{4})",
            lambda m: (m.group(1) or "") + "ANIO = '" + m.group(2) + "'",
            sql
        )

        if not is_safe_sql(sql):
            return {
                "error": "Consulta bloqueada por seguridad. Solo se permiten SELECT.",
                "sql": sql,
                "raw_sql": raw_sql
            }

        try:
            results = execute_sql_query(sql)
        except Exception as db_error:
            return {
                "error": f"Error ejecutando SQL: {str(db_error)}",
                "sql": sql,
                "raw_sql": raw_sql
            }

        return {
            "sql": sql,
            "raw_sql": raw_sql,
            "results": results[:10]
        }

    except Exception as e:
        return {
            "error": f"Error generando o ejecutando SQL: {str(e)}",
            "sql": None,
            "raw_sql": None
        }