import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
import smtplib, ssl
from email.message import EmailMessage

# ---------- Paths ----------
DATA_DIR = Path("data")
CATALOG_PATH = DATA_DIR / "catalog.csv"
PEOPLE_PATH = DATA_DIR / "people.txt"
LOG_PATH = DATA_DIR / "order_log.csv"

# ---------- Load ----------
@st.cache_data
def load_catalog():
    df = pd.read_csv(CATALOG_PATH)
    df = df[["item", "product_number"]].dropna()
    df["item"] = df["item"].astype(str)
    df["product_number"] = df["product_number"].astype(str)
    return df.reset_index(drop=True)

@st.cache_data
def load_people():
    if PEOPLE_PATH.exists():
        return [p.strip() for p in PEOPLE_PATH.read_text().splitlines() if p.strip()]
    return ["Unknown"]

def load_log():
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=["timestamp", "orderer", "item", "product_number", "qty"])
    try:
        df = pd.read_csv(LOG_PATH)
        if df.empty or "item" not in df.columns:
            raise ValueError
        return df
    except Exception:
        return pd.DataFrame(columns=["timestamp", "orderer", "item", "product_number", "qty"])

def save_log(df):
    log = load_log()
    combined = pd.concat([log, df], ignore_index=True)
    combined.to_csv(LOG_PATH, index=False)

def send_email(order_df, orderer, timestamp):
    config = st.secrets["smtp"]
    msg = EmailMessage()
    msg["Subject"] = f"📦 Supply Order from {orderer} at {timestamp}"
    msg["From"] = config["from"]
    msg["To"] = config["to"]

    body = f"Order placed by: {orderer} on {timestamp}\n\n"
    body += order_df.to_string(index=False)
    msg.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP(config["host"], config["port"]) as server:
        if not config.get("use_ssl", False):
            server.starttls(context=context)
        server.login(config["user"], config["password"])
        server.send_message(msg)

# ---------- App ----------
st.set_page_config("📦 Supply Order", layout="wide")
st.title("📦 Supply Ordering")

catalog = load_catalog()
people = load_people()
log_df = load_log()

# Init session state
if "quantities" not in st.session_state:
    st.session_state.quantities = {}

# Who is ordering
orderer = st.selectbox("Who is placing the order?", people)

# Search
search = st.text_input("Search items:")
filtered = catalog.copy()
if search:
    filtered = filtered[filtered["item"].str.contains(search, case=False, na=False)]

# Get latest order info
if not log_df.empty:
    log_df["timestamp"] = pd.to_datetime(log_df["timestamp"], errors="coerce")
    latest = (
        log_df.dropna(subset=["item", "product_number"])
        .sort_values("timestamp")
        .groupby(["item", "product_number"])
        .last()
        .reset_index()
        .rename(columns={"timestamp": "last_ordered_at", "orderer": "last_orderer"})
    )
else:
    latest = pd.DataFrame(columns=["item", "product_number", "last_ordered_at", "last_orderer"])

# Merge
merged = pd.merge(filtered, latest, on=["item", "product_number"], how="left")
merged["last_ordered_at"] = pd.to_datetime(merged["last_ordered_at"], errors="coerce")
merged = merged.sort_values("last_ordered_at", ascending=False, na_position="last")
merged["qty"] = 0

# Apply previous qtys
for i, row in merged.iterrows():
    key = f"{row['product_number']}_{i}"
    if key in st.session_state.quantities:
        merged.at[i, "qty"] = st.session_state.quantities[key]

# Table UI
st.subheader("🧾 Supply Table")
edited = st.data_editor(
    merged[["qty", "item", "product_number", "last_ordered_at", "last_orderer"]],
    use_container_width=True,
    hide_index=True,
    height=600,
    column_config={
        "qty": st.column_config.NumberColumn("Qty", min_value=0, step=1),
        "item": st.column_config.TextColumn("Item", disabled=True),
        "product_number": st.column_config.TextColumn("Product #", disabled=True),
        "last_ordered_at": st.column_config.DatetimeColumn("Last Ordered", format="YYYY-MM-DD HH:mm", disabled=True),
        "last_orderer": st.column_config.TextColumn("Ordered By", disabled=True)
    },
    key="order_editor"
)

# Track selections
selected_items = []
for i, row in edited.iterrows():
    key = f"{row['product_number']}_{i}"
    st.session_state.quantities[key] = row["qty"]
    if row["qty"] > 0:
        selected_items.append({
            "item": row["item"],
            "product_number": row["product_number"],
            "qty": row["qty"]
        })

# Log order
if st.button("📤 Log and Email Order"):
    if not selected_items:
        st.warning("Please select at least one item with Qty > 0.")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df = pd.DataFrame(selected_items)
        df["timestamp"] = timestamp
        df["orderer"] = orderer
        save_log(df[["timestamp", "orderer", "item", "product_number", "qty"]])
        send_email(df[["item", "product_number", "qty"]], orderer, timestamp)

        st.success("✅ Order logged and emailed!")

        st.subheader("📋 Copy/Paste Shopping List")
        lines = [f"{r['item']} — {r['product_number']} — Qty {r['qty']}" for r in selected_items]
        st.text_area("List", value="\n".join(lines), height=200)
        st.download_button(
            "⬇️ Download CSV",
            data=pd.DataFrame(selected_items)[["item", "product_number", "qty"]].to_csv(index=False).encode("utf-8"),
            file_name=f"order_{timestamp.replace(':','-')}.csv",
            mime="text/csv"
        )

# Show logs
if not log_df.empty:
    st.divider()
    st.subheader("📜 Past Orders")
    st.dataframe(log_df.sort_values("timestamp", ascending=False), use_container_width=True)
