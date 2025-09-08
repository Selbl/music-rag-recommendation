ingest:
	python scripts/ingest.py

eval-retrieval:
	python scripts/eval_retrieval.py --ground_truth data/ground_truth.csv --use_rerank

eval-llm:
	python scripts/eval_llm.py --eval_queries data/eval_queries.csv

run:
	streamlit run app.py

dash:
	streamlit run dashboard.py

docker:
	docker compose up --build
