"""
Parte 1: Indexación de artículos en ChromaDB usando embeddings de OpenAI.
Soporta indexación incremental: no reindexia artículos ya existentes.
"""
import os
import tiktoken
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from articulos import articulos

load_dotenv()

MODELO_EMBEDDING = "text-embedding-3-small"
COSTO_POR_MILLON_TOKENS = 0.02  # USD
COLECCION_NOMBRE = "articulos_tech"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma = chromadb.PersistentClient(path="./chroma_db")
coleccion = chroma.get_or_create_collection(name=COLECCION_NOMBRE)


def contar_tokens(textos: list[str]) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return sum(len(enc.encode(t)) for t in textos)


def obtener_ids_existentes() -> set[str]:
    resultado = coleccion.get(include=[])
    return set(resultado["ids"])


def crear_embedding(texto: str) -> list[float]:
    respuesta = client.embeddings.create(input=texto, model=MODELO_EMBEDDING)
    return respuesta.data[0].embedding


def indexar():
    ids_existentes = obtener_ids_existentes()

    pendientes = [a for a in articulos if a["id"] not in ids_existentes]
    if not pendientes:
        print("Todos los artículos ya están indexados. Nada que hacer.")
        return

    textos = [a["contenido"] for a in pendientes]
    total_tokens = contar_tokens(textos)
    costo_estimado = (total_tokens / 1_000_000) * COSTO_POR_MILLON_TOKENS

    print(f"Artículos a indexar: {len(pendientes)}")
    print(f"Tokens a procesar:   {total_tokens}")
    print(f"Coste estimado:      ${costo_estimado:.6f} USD")
    print()

    embeddings = [crear_embedding(t) for t in textos]

    coleccion.add(
        ids=[a["id"] for a in pendientes],
        embeddings=embeddings,
        documents=textos,
        metadatas=[{"titulo": a["titulo"], "id": a["id"]} for a in pendientes],
    )

    print(f"Indexados {len(pendientes)} artículos correctamente.")
    print(f"Total en colección: {coleccion.count()}")


if __name__ == "__main__":
    indexar()
