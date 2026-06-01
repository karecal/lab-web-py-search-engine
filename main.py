"""
Bonus: Endpoint FastAPI GET /buscar?q= sobre el motor de búsqueda semántica.
Ejecutar con: uvicorn main:app --reload
"""
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from buscar import buscar

app = FastAPI(title="Motor de Búsqueda Semántica", version="1.0.0")


class Resultado(BaseModel):
    id: str
    titulo: str
    contenido: str
    score: float


class RespuestaBusqueda(BaseModel):
    query: str
    n_resultados: int
    resultados: list[Resultado]


@app.get("/buscar", response_model=RespuestaBusqueda)
def endpoint_buscar(
    q: str = Query(..., min_length=1, description="Texto de la búsqueda"),
    n: int = Query(3, ge=1, le=8, description="Número de resultados"),
):
    if not q.strip():
        raise HTTPException(status_code=400, detail="La query no puede estar vacía")

    resultados = buscar(q, n_resultados=n)
    return RespuestaBusqueda(query=q, n_resultados=len(resultados), resultados=resultados)


@app.get("/")
def root():
    return {"mensaje": "Motor de búsqueda semántica activo", "endpoints": ["/buscar?q=tu-consulta"]}
