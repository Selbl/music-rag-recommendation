import os, pandas as pd, streamlit as st, plotly.express as px
from sqlalchemy import create_engine

LOG_DB = os.getenv("LOG_DB","./logs/app.db")
engine = create_engine(f"sqlite:///{LOG_DB}")

st.set_page_config(page_title="RAG Monitoring", layout="wide")
st.title("📊 Monitoring Dashboard")

def load_tbl(name):
    try: return pd.read_sql(f"select * from {name}", engine)
    except: return pd.DataFrame()

fb = load_tbl("feedback")

if fb.empty:
    st.info("No feedback yet.")
else:
    fb["ts"] = pd.to_datetime(fb.get("ts", pd.Timestamp.utcnow()))
    st.metric("Total feedback", len(fb))
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(px.histogram(fb, x="latency_s", nbins=20, title="Latency (s)"))
    c2.plotly_chart(px.pie(fb, names="feedback", title="Feedback split"))
    c3.plotly_chart(px.histogram(fb, x="method", title="Methods used"))
    # time series
    fb["date"] = fb["ts"].dt.date if "ts" in fb else pd.Timestamp.utcnow().date()
    st.plotly_chart(px.line(fb.groupby("date")["feedback"].count().reset_index(name="queries"), x="date", y="queries", title="Daily Query Volume"))
    if "note" in fb:
        st.subheader("Recent Comments")
        st.dataframe(fb.sort_index(ascending=False)[["query","method","feedback","note"]].head(20))
