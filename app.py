"""Launch: python -m streamlit run app.py --server.address 127.0.0.1"""
import streamlit as st
from demo_service import build_demo, file_signature

st.set_page_config(page_title="Product Compass | ML Demo", page_icon="🧭", layout="wide")


@st.cache_resource(show_spinner="Loading the fixed TF-IDF recommender...")
def cached_demo(signature):
    # signature is a cache key; data changes require a new fit and hash check.
    return build_demo()


def clear_interests():
    st.session_state["liked"] = []
    st.session_state["excluded"] = []
    st.session_state.pop("submitted", None)


st.caption("STEP 9  /  APPLIED MACHINE LEARNING")
st.title("Product Compass")
st.write("Choose a few subscription boxes you like. Discover five related products.")

try:
    service = cached_demo(file_signature())
except (ValueError, OSError) as error:
    st.error(str(error))
    st.info("Keep your existing data/raw and artifacts folders. No need to rerun test evaluation.")
    st.stop()

with st.sidebar:
    st.header("Behind the recommendations")
    st.metric("Historical catalog", len(service.catalog))
    st.write("**Fixed model:** TF-IDF + cosine similarity")
    st.write("**Fit data:** training + validation reviews")
    st.caption("History ends: " + service.history_end)
    st.divider()
    st.write("No account or API key is needed. Your selections stay in this browser session.")
    st.caption("Local learning demo, not a live shopping catalog. Prices and availability are not provided.")

left, right = st.columns([1, 1.6], gap="large")
with left:
    st.subheader("1. Tell us what you like")
    with st.form("interests"):
        liked = st.multiselect(
            "Products you like", service.catalog, format_func=service.label,
            max_selections=10, key="liked", placeholder="Search coffee, books, snacks...",
        )
        excluded = st.multiselect(
            "Already tried or want to exclude (optional)", service.catalog,
            format_func=service.label, max_selections=10, key="excluded",
        )
        submitted = st.form_submit_button("Recommend products", type="primary")
    st.button("Clear selections", on_click=clear_interests)
    st.caption("Leave interests empty for popular picks. Selected and excluded products never appear in results.")
    if submitted:
        # Validate before updating the last successful submitted selection.
        try:
            service.recommendations(liked, excluded)
            st.session_state["submitted"] = (tuple(liked), tuple(excluded))
        except ValueError as error:
            st.error(str(error))

with right:
    st.subheader("2. Your recommendations")
    active_likes, active_excluded = st.session_state.get("submitted", ((), ()))
    st.caption(
        f"Based on the last submitted selection: {len(active_likes)} interests, "
        f"{len(active_excluded)} exclusions."
    )
    try:
        results = service.recommendations(active_likes, active_excluded)
    except ValueError as error:
        st.warning(str(error) + " Clear selections and try again.")
        st.stop()
    if not active_likes:
        st.info("No interests submitted yet: showing popular picks.")
    elif results and all(row["score"] == 0 for row in results):
        st.info("No usable text match was found. Showing popularity fallback results.")
    if not results:
        st.info("No items remain after your exclusions. Clear a few selections and try again.")
    for rank, row in enumerate(results, start=1):
        with st.container(border=True):
            # Plain text prevents dataset titles from being interpreted as Markdown/HTML.
            st.text(f"{rank}. {row['title']}")
            st.caption(row["product_id"])
            st.write(f"**{row['reason']}** · similarity {row['score']:.3f}")
            if row["shared_terms"]:
                st.text("Shared text features: " + ", ".join(row["shared_terms"]))
    st.caption("Similarity is a ranking signal, not a probability or predicted rating.")

st.divider()
with st.expander("How this relates to our evaluated model"):
    st.write(
        "We reuse the fixed TF-IDF model and average the text vectors of your selected interests. "
        "Candidates are ranked by cosine similarity; ties use historical popularity. "
        "Shared text features show large contributions to the dot product, not causal reasons."
    )
    st.write(
        "The model is fitted on training plus validation history, excluding test reviews from fitting. "
        "This interface uses interests you enter now, whereas Step 8 evaluated historical ratings. "
        "Those offline scores do not measure this new interactive experience."
    )
    st.warning(
        "Product descriptions are a later snapshot. The product evaluation has few returning users. "
        "This demo does not establish live recommendation quality or business impact."
    )
    st.caption("Existing evaluation reports are never rewritten by this app. MovieLens remains a separate benchmark.")
