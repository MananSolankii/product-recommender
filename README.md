# Product Compass

A product recommendation project built with Python, scikit-learn and Streamlit.
Choose subscription boxes you like and receive five related products, with
shared text features explaining the matches. Exclude products you have already
tried, or leave interests empty for popular picks.

The project compares popularity, TF-IDF content recommendations and item-based
collaborative filtering using chronological validation and a later held-out test.
The demo uses the validation-selected TF-IDF model.

![Product Compass running locally with coffee recommendations](docs/demo.png)

*Local demo capture; the supplied scrolling screenshot contains stitching artifacts.*

## Results

Held-out Amazon Subscription Boxes results, averaged over 1,397 eligible test
users. Values are rounded from the completed local evaluation.

| Model | Recall@5 | NDCG@5 |
| --- | ---: | ---: |
| Popularity baseline | 0.1451 | 0.0931 |
| Selected TF-IDF model | 0.1494 | 0.0967 |

Most test users had no prior history and received popularity recommendations.
Only 52 were returning users, so the personalized subgroup has substantial
uncertainty. These are offline ranking results, not classification accuracy or
evidence of increased sales. [Full results and limitations](docs/RESULTS.md).

## Features

- Cleaning and exploration of real reviews, including duplicate removal.
- Popularity, content-based, item-based collaborative and pruned neighbor models.
- Fixed validation experiments with separate selection for products and MovieLens.
- Recall@5 and NDCG@5 for all, returning and new users.
- Reports with hashes, package versions and time boundaries.
- Streamlit demo with exclusions, popularity fallback and text explanations.
- 29 automated tests covering evaluation, models and demo behavior.

## Run the demo

 Keep your existing `.venv`, `data/raw/` and
`artifacts/` folders. Launch from the project folder:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

**Fresh Windows setup:** use Python 3.11. Extract or clone the project and open
PowerShell in the folder containing `app.py` and `requirements.txt`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-demo.txt
.\.venv\Scripts\python.exe download_data.py
.\.venv\Scripts\python.exe download_metadata.py
.\.venv\Scripts\python.exe tune_models.py --dataset products
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

On macOS/Linux, create the environment with `python3.11 -m venv .venv` and
replace `.\.venv\Scripts\python.exe` with `.venv/bin/python` in later commands.
Open the local URL printed by Streamlit. No GPU, API key or account is needed;
the initial package and dataset downloads require internet access.

The demo requiresbmatching raw data. It checks the recorded selection and input hashes, fits on training plus
validation history, and does not evaluate test data on a button click.
The observed historical demo catalog contains 471 products. This is a local
learning demo, not a live shopping catalog.

## Evaluation and reproduction

1. Split reviews globally by time into approximately 70% training, 15% validation
   and 15% test, keeping timestamp ties together.
2. Select by validation NDCG@5, then Recall@5, then fixed candidate order.
3. Refit the selected model on training plus validation and assess the later test period.

Ratings of 4 or 5 act as positive feedback. Previously seen products are excluded.
Targets outside the historical candidate catalog remain in metric denominators
as misses. Future feedback does not update the model during evaluation.

For a fresh reproduction of both datasets, after downloading product inputs:

```powershell
.\.venv\Scripts\python.exe tune_models.py --dataset both
.\.venv\Scripts\python.exe final_evaluate.py
.\.venv\Scripts\python.exe -m unittest discover -p "test_*.py"
```

MovieLens downloads automatically for its benchmark. If you already completed
the final assessment, preserve its reports; no rerun is needed to publish the
project. Do not tune against these test results. 
## Project guide

| Files | Purpose |
| --- | --- |
| `download_data.py`, `download_metadata.py`, `explore_data.py` | Obtain and clean product inputs |
| `evaluate.py`, `tune_models.py`, `final_evaluate.py` | Time split, selection and final assessment |
| `content_model.py`, `collaborative_model.py`, `neighbor_model.py` | Recommendation models |
| `benchmark_movielens.py` | Separate MovieLens benchmark |
| `demo_service.py`, `app.py` | Fixed product model and interactive UI |
| `test_*.py` | Automated checks |
| 



## Data and limitations

- [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/): Subscription Boxes
  reviews and metadata from the McAuley Lab. The cleaned set contains 16,016
  events, 15,237 users and 641 products.
- [GroupLens MovieLens 100K](https://grouplens.org/datasets/movielens/100k/): a
  separate benchmark with 100,000 ratings. Its scores do not measure product quality.

Raw datasets are downloaded from their sources and excluded from Git. Consult
each source's usage terms and attribution guidance. MovieLens reference:
F. Maxwell Harper and Joseph A. Konstan (2015), *The MovieLens Datasets: History
and Context*, [doi:10.1145/2827872](https://doi.org/10.1145/2827872).

Limits include sparse product histories, popularity fallback for new users,
later-snapshot product text and a single temporal split. Historically accurate
availability of product descriptions is not established. There is no online
experiment, statistical significance claim or production scalability claim.
