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
    df = df.dropna(subset=["item", "product_number"])
    df["item"] = df["item"].astype(str)
    df["product_number"] = df["product_number"].astype(str)
    return df.reset_index(drop=True)

@st.cache_data
def load_people():
    if PEOPLE_PATH.exists():
        return [p.strip() for p in PEOPLE_PATH.read_text().splitlines() if p.strip()]
    return ["Unknown"]

def load_log():
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=["timestamp", "orderer", "item", "product_number", "qty"])

def save_log(new_entries):
    log = load_log()
    combined = pd.concat([log, new_entries], ignore_index=True)
    combined.to_csv(LOG_PATH, index=False)

# ---------- Email ----------
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

# Load
catalog = load_catalog()
people = load_people()
log_df = load_log()

# Persistent qty inputs
if "quantities" not in st.session_state:
    st.session_state.quantities = {}

# Orderer
orderer = st.selectbox("Who is placing the order?", people)

# Search
search = st.text_input("Search items:")
filtered = catalog.copy()
if search:
    filtered = catalog[catalog["item"].str.contains(search, case=False, na=False)]

# Item list with quantity inputs
st.subheader("🛒 Select Quantities")
qty_input = []
for i, row in filtered.iterrows():
    item_key = f"{row['product_number']}_{i}"
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"**{row['item']}** — `{row['product_number']}`")
    with col2:
        qty = st.number_input("Qty", min_value=0, step=1, value=st.session_state.quantities.get(item_key, 0), key=item_key)
        if qty > 0:
            st.session_state.quantities[item_key] = qty
            qty_input.append({
                "item": row["item"],
                "product_number": row["product_number"],
                "qty": qty
            })

# Submit button
if st.button("📤 Log and Email Order"):
    if not qty_input:
        st.warning("Please enter quantities before logging.")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        order_df = pd.DataFrame(qty_input)
        order_df["timestamp"] = timestamp
        order_df["orderer"] = orderer
        save_log(order_df[["timestamp", "orderer", "item", "product_number", "qty"]])
        send_email(order_df[["item", "product_number", "qty"]], orderer, timestamp)

        # Show success + shopping list
        st.success("Order logged and emailed.")
        st.subheader("🧾 Copy/Paste Shopping List")
        lines = [f"{r['item']} — {r['product_number']} — Qty {r['qty']}" for r in qty_input]
        st.text_area("Shopping List", value="\n".join(lines), height=200)
        st.download_button(
            "⬇️ Download CSV",
            data=pd.DataFrame(qty_input)[["item", "product_number", "qty"]].to_csv(index=False).encode("utf-8"),
            file_name=f"order_{timestamp.replace(':', '-')}.csv",
            mime="text/csv"
        )

# Order log
if not log_df.empty:
    st.divider()
    st.subheader("📜 Past Orders")
    st.dataframe(log_df.sort_values("timestamp", ascending=False), use_container_width=True)
