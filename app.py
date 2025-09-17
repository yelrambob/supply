# app.py

import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="📦 Supply Ordering", layout="wide")

# ---------- Paths ----------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CATALOG_PATH = DATA_DIR / "catalog.csv"
LOG_PATH = DATA_DIR / "order_log.csv"

# ---------- Load catalog ----------
def load_catalog():
    try:
        df = pd.read_csv(CATALOG_PATH)
        df = df.dropna(subset=['item'])  # Drop rows without item
        df['product_number'] = df['product_number'].astype(str)
        return df
    except Exception as e:
        st.error(f"Failed to load catalog: {e}")
        return pd.DataFrame(columns=["item", "product_number", "current_qty", "per_box_qty", "sort_order"])

# ---------- Load order log ----------
def load_order_log():
    if LOG_PATH.exists():
        try:
            log_df = pd.read_csv(LOG_PATH, parse_dates=['ordered_at'])
            return log_df
        except Exception as e:
            st.error(f"Failed to load order log: {e}")
    return pd.DataFrame(columns=["item", "product_number", "qty", "ordered_at", "orderer"])

# ---------- Merge last order info ----------
def merge_last_order_info(catalog_df, log_df):
    if log_df.empty:
        catalog_df["last_ordered_at"] = ""
        catalog_df["orderer"] = ""
        return catalog_df

    last_orders = (
        log_df.sort_values("ordered_at")
              .groupby("product_number")
              .last()
              .reset_index()[["product_number", "ordered_at", "orderer"]]
    )

    merged = catalog_df.merge(last_orders, on="product_number", how="left")
    merged.rename(columns={"ordered_at": "last_ordered_at"}, inplace=True)
    return merged.sort_values("last_ordered_at", ascending=False, na_position='last')

# ---------- Main UI ----------
st.title("📦 Supply Ordering")

catalog_df = load_catalog()
log_df = load_order_log()

merged_df = merge_last_order_info(catalog_df, log_df)

# ---------- Display in scrollable table ----------
st.markdown("### Supply Catalog (Most Recently Ordered First)")
st.dataframe(
    merged_df[["item", "product_number", "current_qty", "per_box_qty", "last_ordered_at", "orderer"]],
    height=600,
    use_container_width=True
)
