import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
import smtplib, ssl
from email.message import EmailMessage

# ---------- Paths ----------
CATALOG_PATH = Path("data/catalog.csv")
LOG_PATH = Path("data/order_log.csv")
PEOPLE_PATH = Path("data/people.txt")
EMAILS_PATH = Path("data/emails.csv")

# ---------- Load data ----------
def load_catalog():
    return pd.read_csv(CATALOG_PATH)

def load_people():
    return [name.strip() for name in open(PEOPLE_PATH).readlines() if name.strip()]

def load_emails():
    return pd.read_csv(EMAILS_PATH)

def load_log():
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=["timestamp", "orderer", "item", "product_number", "qty"])

# ---------- Email logic ----------
def send_email(subject, body, to_emails):
    config = st.secrets["email"]
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = config["email"]
    msg["To"] = ", ".join(to_emails)
    msg.set_content(body)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(config["smtp_server"], config["smtp_port"], context=context) as server:
        server.login(config["email"], config["password"])
        server.send_message(msg)

# ---------- UI ----------
st.set_page_config("📦 Supply Ordering", layout="wide")
st.title("📦 Supply Ordering")

catalog = load_catalog()
people = load_people()
emails = load_emails()
log = load_log()

orderer = st.selectbox("Who is placing this order?", people)
search = st.text_input("Search for a supply item:")

if search:
    filtered_catalog = catalog[catalog["item"].str.contains(search, case=False, na=False)]
else:
    filtered_catalog = catalog.copy()

st.write("### Select quantities for supplies")
qty_input = {}
for i, row in filtered_catalog.iterrows():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.text(f"{row['item']} ({row['product_number']})")
    with col2:
        qty = st.number_input(f"Qty for {row['item']}", min_value=0, step=1, key=row['product_number'])
        if qty > 0:
            qty_input[row['product_number']] = {
                "item": row['item'],
                "product_number": row['product_number'],
                "qty": qty,
            }

if qty_input:
    st.write("### 🧾 Current Order Summary")
    order_df = pd.DataFrame(qty_input.values())
    st.dataframe(order_df, use_container_width=True)

    if st.button("📤 Log and Email Order"):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entries = []
        for entry in qty_input.values():
            log_entries.append({
                "timestamp": now,
                "orderer": orderer,
                "item": entry["item"],
                "product_number": entry["product_number"],
                "qty": entry["qty"]
            })
        log_df = pd.DataFrame(log_entries)
        new_log = pd.concat([log, log_df], ignore_index=True)
        new_log.to_csv(LOG_PATH, index=False)

        # Send Email
        recipient_emails = emails["email"].dropna().tolist()
        email_body = f"Order placed by: {orderer} on {now}\n\n"
        email_body += order_df.to_string(index=False)
        send_email("📦 New Supply Order Logged", email_body, recipient_emails)

        st.success("Order logged and email sent.")

# ---------- View Log ----------
st.write("### 📜 Order Log (Most Recent First)")
if LOG_PATH.exists():
    log = pd.read_csv(LOG_PATH)
    log = log.sort_values(by="timestamp", ascending=False)
    st.dataframe(log, use_container_width=True)
else:
    st.info("No orders have been logged yet.")
