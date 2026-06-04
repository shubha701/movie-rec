import requests
import streamlit as st

# CONFIG
API_BASE = "http://127.0.0.1:8000"
# API_BASE = "https://your-render-url.onrender.com"

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# STYLES
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
        max-width: 1400px;
    }

    .movie-title{
        font-size:14px;
        height:40px;
        overflow:hidden;
    }

    .small-muted{
        color:gray;
        font-size:13px;
    }

    .card{
        padding:15px;
        border-radius:12px;
        border:1px solid #ddd;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# SESSION STATE
if "view" not in st.session_state:
    st.session_state.view = "home"

if "selected_imdb_id" not in st.session_state:
    st.session_state.selected_imdb_id = None

# ROUTING
def goto_home():
    st.session_state.view = "home"
    st.session_state.selected_imdb_id = None
    st.rerun()


def goto_details(imdb_id):
    st.session_state.view = "details"
    st.session_state.selected_imdb_id = imdb_id
    st.rerun()



# API HELPER
@st.cache_data(ttl=60)
def api_get(path, params=None):
    try:
        r = requests.get(
            f"{API_BASE}{path}",
            params=params,
            timeout=30
        )

        if r.status_code != 200:
            return None, r.text

        return r.json(), None

    except Exception as e:
        return None, str(e)


# POSTER GRID
def poster_grid(cards, cols=6, key_prefix="grid"):

    if not cards:
        st.info("No movies found.")
        return

    rows = (len(cards) + cols - 1) // cols

    idx = 0

    for r in range(rows):

        columns = st.columns(cols)

        for c in range(cols):

            if idx >= len(cards):
                break

            movie = cards[idx]
            idx += 1

            imdb_id = movie.get("omdb_id")
            title = movie.get("title", "Unknown")
            poster = movie.get("poster_url")

            with columns[c]:

                if poster and poster != "N/A":
                    st.image(poster, use_container_width=True)
                else:
                    st.write("🖼️ No Poster")

                if st.button(
                    "Open",
                    key=f"{key_prefix}_{idx}_{imdb_id}"
                ):
                    goto_details(imdb_id)

                st.markdown(
                    f"<div class='movie-title'>{title}</div>",
                    unsafe_allow_html=True
                )


# ==========================================
# CONVERT TFIDF ITEMS TO CARDS
# ==========================================
def tfidf_cards(items):

    cards = []

    for item in items:

        omdb = item.get("omdb")

        if not omdb:
            continue

        cards.append(
            {
                "omdb_id": omdb.get("omdb_id"),
                "title": omdb.get("title"),
                "poster_url": omdb.get("poster_url")
            }
        )

    return cards


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:

    st.title("🎬 Menu")

    if st.button("🏠 Home"):
        goto_home()

    st.divider()

    category = st.selectbox(
        "Home Category",
        [
            "popular",
            "trending",
            "top_rated",
            "upcoming",
            "now_playing"
        ]
    )

    grid_cols = st.slider(
        "Grid Columns",
        4,
        8,
        6
    )

# ==========================================
# HEADER
# ==========================================
st.title("🎬 Movie Recommendation System")

st.caption(
    "Search movie → Open Details → Get TF-IDF & Genre Recommendations"
)

st.divider()

# ==========================================
# HOME PAGE
# ==========================================
if st.session_state.view == "home":

    search = st.text_input(
        "Search Movie",
        placeholder="Batman, Avengers, Interstellar..."
    )

    st.divider()

    # ======================================
    # SEARCH MODE
    # ======================================
    if search.strip():

        data, err = api_get(
            "/omdb/search",
            {
                "query": search
            }
        )

        if err:
            st.error(err)

        else:

            results = data.get("Search", [])

            if results:

                labels = [
                    f"{m['Title']} ({m['Year']})"
                    for m in results
                ]

                selected = st.selectbox(
                    "Suggestions",
                    ["Select Movie"] + labels
                )

                if selected != "Select Movie":

                    idx = labels.index(selected)

                    goto_details(
                        results[idx]["imdbID"]
                    )

                cards = []

                for m in results:

                    cards.append(
                        {
                            "omdb_id": m.get("imdbID"),
                            "title": m.get("Title"),
                            "poster_url": m.get("Poster")
                        }
                    )

                st.subheader("Search Results")

                poster_grid(
                    cards,
                    cols=grid_cols,
                    key_prefix="search"
                )

            else:
                st.warning("No movies found.")

        st.stop()

    # ======================================
    # HOME FEED
    # ======================================
    st.subheader(
        category.replace("_", " ").title()
    )

    movies, err = api_get(
        "/home",
        {
            "category": category,
            "limit": 24
        }
    )

    if err:
        st.error(err)

    else:
        poster_grid(
            movies,
            cols=grid_cols,
            key_prefix="home"
        )

# ==========================================
# DETAILS PAGE
# ==========================================
elif st.session_state.view == "details":

    imdb_id = st.session_state.selected_imdb_id

    if not imdb_id:
        st.warning("No movie selected.")
        st.stop()

    top1, top2 = st.columns([4, 1])

    with top2:
        if st.button("⬅ Back"):
            goto_home()

    # ======================================
    # DETAILS
    # ======================================
    movie, err = api_get(
        f"/movie/id/{imdb_id}"
    )

    if err:
        st.error(err)
        st.stop()

    left, right = st.columns([1, 2])

    with left:

        if movie.get("poster_url"):
            st.image(
                movie["poster_url"],
                use_container_width=True
            )

    with right:

        st.header(movie.get("title"))

        st.write(
            f"⭐ IMDb Rating: {movie.get('imdb_rating','N/A')}"
        )

        st.write(
            f"📅 Year: {movie.get('year','N/A')}"
        )

        st.write(
            f"⏱ Runtime: {movie.get('runtime','N/A')}"
        )

        st.write(
            f"🎬 Director: {movie.get('director','N/A')}"
        )

        st.write(
            f"🎭 Actors: {movie.get('actors','N/A')}"
        )

        genres = ", ".join(
            movie.get("genres", [])
        )

        st.write(f"🎞 Genres: {genres}")

        st.markdown("### Plot")

        st.write(
            movie.get("overview", "")
        )

    st.divider()

    # ======================================
    # RECOMMENDATIONS
    # ======================================
    st.header("Recommended Movies")

    bundle, err = api_get(
        "/movie/search",
        {
            "query": movie["title"],
            "tfidf_top_n": 12,
            "genre_limit": 12
        }
    )

    if err:

        st.warning("Recommendation service unavailable.")

    else:

        st.subheader("🔍 Similar Movies (TF-IDF)")

        poster_grid(
            tfidf_cards(
                bundle.get(
                    "tfidf_recommendations",
                    []
                )
            ),
            cols=grid_cols,
            key_prefix="tfidf"
        )

        st.divider()

        st.subheader("🎭 Genre Recommendations")

        poster_grid(
            bundle.get(
                "genre_recommendations",
                []
            ),
            cols=grid_cols,
            key_prefix="genre"
        )