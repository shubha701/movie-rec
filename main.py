import os
import pickle
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx 
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv


# .env, cors and path config
load_dotenv()

OMDB_API_KEY = os.getenv("OMDB_API_KEY")

OMDB_BASE = "http://www.omdbapi.com/"

if not OMDB_API_KEY:
    raise RuntimeError("OMDB_API_KEY missing. Put it in .env as OMDB_API_KEY=xxxx")

app = FastAPI(title="Movie Recommender API", description="API for movie recommendations based on user ratings.", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # for local Streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# path and Global vars configs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "DF.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None


# Models
class OMDBMovieCard(BaseModel):
    omdb_id: str
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    year: Optional[str] = None
    movie_type: Optional[str] = None
    vote_average: Optional[float] = None
    
class OMDBMovieDetail(BaseModel):
    omdb_id: str
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    year: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: Optional[List[str]] = None
    imdb_rating: Optional[str] = None
    runtime: Optional[str] = None
    director: Optional[str] = None
    actors: Optional[str] = None
    
class TFIDFRecItem(BaseModel):
    title: str
    score: float
    omdb: Optional[OMDBMovieCard] = None
    
class SearchBundleResponse(BaseModel):
    query: str
    movie_details: OMDBMovieDetail
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TFIDFRecItem]
 
 
 # Utility functions   
def _norm_title_url(t:str) -> str:
    return str(t).strip().lower()

def get_poster_url(movie_data):
    poster = movie_data.get("Poster")

    if not poster or poster == "N/A":
        return None

    return poster

async def omdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    q = dict(params)
    q["apikey"] = OMDB_API_KEY
    
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{OMDB_BASE}{path}", params=q)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"OMDB request error: {type(e).__name__} | {repr(e)}",)   
    
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OMDB error: {r.status_code} | {r.text}",)   
    
    return r.json()  

async def omdb_cards_from_results(
    results: List[dict],
    limit: int = 20
) -> List[OMDBMovieCard]:

    out: List[OMDBMovieCard] = []

    for m in (results or [])[:limit]:
        out.append(
            OMDBMovieCard(
                omdb_id=m.get("imdbID", ""),
                title=m.get("Title", ""),
                poster_url=m.get("Poster", ""),
                year=m.get("Year", ""),
                movie_type=m.get("Type", "")
            )
        )

    return out         

async def omdb_movie_details(imdb_id: str) -> OMDBMovieDetail:
    data = await omdb_get(
        "",
        {
            "i": imdb_id,
            "plot": "full"
        }
    )

    return OMDBMovieDetail(
        omdb_id=data.get("imdbID", ""),
        title=data.get("Title", ""),
        overview=data.get("Plot", ""),
        year=data.get("Year", ""),
        release_date=data.get("Released", None),
        poster_url=get_poster_url(data),
        backdrop_url=data.get("Backdrop", None),
        genres=data.get("Genre", "").split(", ") if data.get("Genre") else [],
        imdb_rating=data.get("imdbRating", ""),
        runtime=data.get("Runtime", ""),
        director=data.get("Director", ""),
        actors=data.get("Actors", "")
    )
    
async def omdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    """
    Raw OMDb response for keyword search.
    Streamlit will use this for suggestions and movie grid.
    """
    return await omdb_get(
        "",
        {
            "s": query,
            "page": page,
            "type": "movie"
        }
    )    

async def omdb_search_first(query: str) -> Optional[dict]:
    data = await omdb_search_movies(query=query, page=1)

    results = data.get("Search", [])

    return results[0] if results else None    

# Help of function
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    """
    indices.pkl can be:
    - dict (title -> index)
    - pandas Series (index=title, value=index)
    we normalize into TITLE_TO_IDX.
    """
    title_to_idx: Dict[str, int] = {}
    
    if isinstance(indices, dict):
        for k, v in indices.items():
            title_to_idx[_norm_title_url(k)] = int(v)
        return title_to_idx
    
    #pandas Series or similar mapping
    try:
        for k, v in indices.items():
            title_to_idx[_norm_title_url(k)] = int(v)
        return title_to_idx
    except Exception:
        # Last resort: if it's a list-like etc.
        raise RuntimeError(
            "indices.pkl must be dict or pandas Series-like (with.items())"
        )

def get_local_idx_by_title(title: str) -> int:
    global TITLE_TO_IDX
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF index map not initialized")
    key = _norm_title_url(title)
    if key not in TITLE_TO_IDX:
        raise HTTPException(status_code=404, detail=f"title not found in local dataset: '{title}'")
    return int(TITLE_TO_IDX[key])

def tfidf_recommend_titles(query_title: str, top_n: int = 10) -> List[Tuple[str, float]]:
    global df, tfidf_matrix
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded")
    
    idx = get_local_idx_by_title(query_title)
    
    # query vector
    qv = tfidf_matrix[idx]
    scores = (tfidf_matrix @ qv.T).toarray().ravel()
    
    # sort descending
    order = np.argsort(-scores)
    
    out: List[Tuple[str, float]] = []
    for i in order:
        if int(i) == int(idx):
            continue
        try:
            title_i = str(df.iloc[int(i)]["title"])
        except Exception:
            continue
        out.append((title_i, float(scores[int(i)])))
        if len(out) >= top_n:
            break
    return out

async def attach_omdb_card_by_title(title: str) -> Optional[OMDBMovieCard]:
    try:
        m = await omdb_search_first(title)

        if not m:
            return None

        return OMDBMovieCard(
            omdb_id=m.get("imdbID", ""),
            title=m.get("Title") or title,
            poster_url=m.get("Poster", ""),
            year=m.get("Year"),
            movie_type=m.get("Type"),
        )

    except Exception:
        return None

async def genre_recommendations_from_dataset(genres: List[str], limit: int = 12) -> List[TFIDFRecItem]:
    if not genres:
        return []

    genre = genres[0]
    data = await omdb_search_movies(query=genre, page=1)
    cards = await omdb_cards_from_results(
        data.get("Search", []),
        limit=limit
    )

    return [
        TFIDFRecItem(
            title=card.title,
            score=0.0,
            omdb=card
        )
        for card in cards
    ]

# FastAPI startup event to load pickles into memory    
@app.on_event("startup")  
def load_pickles():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX
    
    #load df
    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)
        
    #load indices
    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)
    
    #load TF-IDF matrix (Usually scipy sparse)
    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)
    
    #load tfidf vectorizer (optional, not used directly in this code but can be useful for future extensions)
    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)
        
    #Build Normalized map
    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)
    
    #Sanity check
    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must contain a DataFrame with a 'title' column")
    
#Routs
@app.get("/health")
def health():
    return {"status": "ok"}

# Home Routs
@app.get("/home", response_model=List[OMDBMovieCard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
):
    try:
        category_queries = {
            "popular": "Marvel",
            "trending": "Avengers",
            "top_rated": "Batman",
            "upcoming": "Mission Impossible",
            "now_playing": "Fast and Furious"
        }

        if category not in category_queries:
            raise HTTPException(
                status_code=400,
                detail="Invalid category"
            )

        data = await omdb_search_movies(
            query=category_queries[category],
            page=1
        )

        return await omdb_cards_from_results(
            data.get("Search", []),
            limit=limit
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Home route failed: {e}"
        ) 
        
# SEARCH MULTIPLE VIA KEYWORD
@app.get("/omdb/search")
async def omdb_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=100),
):
    """
    Returns RAW OMDb response with 'Search' list.

    Streamlit will use it for:
    - dropdown suggestions
    - grid results
    """

    return await omdb_search_movies(
        query=query,
        page=page
    )  
    
# MOVIE DETAILS
@app.get("/movie/id/{imdb_id}", response_model=OMDBMovieDetail)
async def movie_details_route(imdb_id: str):
    return await omdb_movie_details(imdb_id)            

# GENRE RECOMMENDATION
@app.get("/recommend/genre", response_model=List[OMDBMovieCard])
async def recommend_genre(
    imdb_id: str = Query(...),
    limit: int = Query(18, ge=1, le=50),
):
    details = await omdb_movie_details(imdb_id)

    if not details.genres:
        return []

    genre = details.genres[0]

    data = await omdb_search_movies(
        query=genre,
        page=1
    )

    cards = await omdb_cards_from_results(
        data.get("Search", []),
        limit=limit
    )

    return [
        c for c in cards
        if c.omdb_id != imdb_id
    ]
    
@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    recs = tfidf_recommend_titles(
        title,
        top_n=top_n
    )

    return [
        {
            "title": t,
            "score": s
        }
        for t, s in recs
    ]    
    
@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):

    best = await omdb_search_first(query)

    if not best:
        raise HTTPException(
            status_code=404,
            detail=f"No movie found for query: {query}"
        )

    imdb_id = best["imdbID"]

    details = await omdb_movie_details(imdb_id)

    # TF-IDF Recommendations
    tfidf_items = []

    try:
        recs = tfidf_recommend_titles(
            details.title,
            top_n=tfidf_top_n
        )
    except Exception:
        try:
            recs = tfidf_recommend_titles(
                query,
                top_n=tfidf_top_n
            )
        except Exception:
            recs = []

    for title, score in recs:
        card = await attach_omdb_card_by_title(title)

        tfidf_items.append(
            TFIDFRecItem(
                title=title,
                score=score,
                omdb=card
            )
        )

    # Genre Recommendations
    genre_recs = []

    if details.genres:
        genre_recs = await genre_recommendations_from_dataset(
            details.genres,
            limit=genre_limit
        )

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs
    )