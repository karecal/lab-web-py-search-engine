"""
Parte 2: Motor de búsqueda semántica sobre la colección indexada en ChromaDB.
"""
import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODELO_EMBEDDING = "text-embedding-3-small"
COLECCION_NOMBRE = "articulos_tech"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma = chromadb.PersistentClient(path="./chroma_db")
coleccion = chroma.get_or_create_collection(name=COLECCION_NOMBRE)


def buscar(query: str, n_resultados: int = 3) -> list[dict]:
    respuesta = client.embeddings.create(input=query, model=MODELO_EMBEDDING)
    query_embedding = respuesta.data[0].embedding

    resultados = coleccion.query(
        query_embeddings=[query_embedding],
        n_results=n_resultados,
        include=["documents", "metadatas", "distances"],
    )

    salida = []
    for i in range(len(resultados["ids"][0])):
        # ChromaDB devuelve distancia euclidiana al cuadrado; convertimos a similitud coseno aproximada
        distancia = resultados["distances"][0][i]
        similitud = 1 - distancia / 2  # normalización para embeddings unitarios

        salida.append({
            "id": resultados["ids"][0][i],
            "titulo": resultados["metadatas"][0][i]["titulo"],
            "contenido": resultados["documents"][0][i],
            "score": round(similitud, 4),
        })

    return salida


def imprimir_resultados(query: str, resultados: list[dict]):
    print(f"\nQuery: {query!r}")
    print("-" * 60)
    for r in resultados:
        print(f"  [{r['score']:.4f}] {r['titulo']}")
        print(f"           {r['contenido'][:80]}...")
    print()


if __name__ == "__main__":
    queries = [
        "¿cómo hacer una API en Python?",
        "diferencias entre frameworks de frontend",
        "cómo funciona la autenticación en aplicaciones web",
        "herramientas para trabajar con modelos de lenguaje",
    ]

    for q in queries:
        resultados = buscar(q)
        imprimir_resultados(q, resultados)
