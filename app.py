import streamlit as st
import pandas as pd
from pandas.errors import EmptyDataError
from datetime import datetime
from pathlib import Path
import re
import smtplib, ssl
from email.message import EmailMessage
from supabase import create_client

st.set_page_config(page_title="Supply Ordering", page_icon="📦", layout="wide")

# ---------------- Paths ----------------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CATALOG_PATH = DATA_DIR / "catalog.csv"
PEOPLE_PATH  = DATA_DIR / "people.txt"
EMAILS_PATH  = DATA_DIR / "emails.csv"

# ---------------- Supabase ----------------
def get_supabase():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

supabase = get_supabase()

# ---------------- Robust file helpers ----------------
def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", **kwargs)
    except EmptyDataError:
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Couldn't read {path.name}: {e}")
        return pd.DataFrame()

# ---------------- SMTP ----------------
def _split_emails(txt: str) -> list[str]:
    if not txt:
        return []
    parts = re.split(r'[;,]\s*', str(txt))
    return [p.strip() for p in parts if p.strip()]

def get_smtp_config():
    try:
        smtp_config = st.secrets["smtp"]
        return {
            "host": smtp_config.get("host"),
            "port": int(smtp_config.get("port", 587)),
            "username": smtp_config.get("user"),
            "password": smtp_config.get("password", "").replace(" ", ""),
            "from": smtp_config.get("from"),
            "subject_prefix": smtp_config.get("subject_prefix", ""),
            "default_to": _split_emails(smtp_config.get("to", "")) if smtp_config.get("to") else [],
            "use_ssl": bool(smtp_config.get("use_ssl", False))
        }
    except Exception as e:
        st.error(f"Error reading SMTP config: {e}")
        return {}

def smtp_ok() -> bool:
    cfg = get_smtp_config()
    required = ["host", "port", "username", "password", "from"]
    return all(cfg.get(k) for k in required)

def send_email(subject: str, body: str, to_emails: list[str] | None):
    cfg = get_smtp_config()
    recipients = (to_emails or []) + cfg.get("default_to", [])
    recipients = sorted({e for e in recipients if e and "@" in e})
    if not recipients:
        raise RuntimeError("No recipients found.")

    msg = EmailMessage()
    msg["Subject"] = f'{cfg["subject_prefix"]}{subject}' if cfg["subject_prefix"] else subject
    msg["From"] = cfg["from"]
    msg["To"] = ", ".join(recipients)
    msg.set_content(body)

    if cfg["use_ssl"]:
        with smtplib.SMTP_SSL(cfg["host"], cfg["port"], context=ssl.create_default_context()) as server:
            server.login(cfg["username"], cfg["password"])
            server.send_message(msg)
    else:
        with smtplib.SMTP(cfg["host"], cfg["port"]) as server:
            server.ehlo()
            server.starttls(context=ssl.create_default_context())
            server.login(cfg["username"], cfg["password"])
            server.send_message(msg)

# ---------------- Load core data ----------------
@st.cache_data
def read_people() -> list[str]:
    if not PEOPLE_PATH.exists():
        return []
    try:
        return [ln.strip() for ln in PEOPLE_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception as e:
        st.warning(f"Couldn't read people.txt: {e}")
        return []

@st.cache_data
def read_catalog() -> pd.DataFrame:
    df = safe_read_csv(CATALOG_PATH)
    if df.empty:
        return pd.DataFrame(columns=["item", "product_number", "multiplier", "items_per_order", "current_qty", "sort_order"])

    for c in ["item", "product_number", "multiplier", "items_per_order", "current_qty", "sort_order"]:
        if c not in df.columns:
            df[c] = pd.NA

    df["item"] = df["item"].astype(str).str.strip()
    df["product_number"] = df["product_number"].astype(str).str.strip()
    df["multiplier"] = pd.to_numeric(df["multiplier"], errors="coerce").fillna(1).astype(int)
    df["items_per_order"] = pd.to_numeric(df["items_per_order"], errors="coerce").fillna(1).astype(int)
    df["current_qty"] = pd.to_numeric(df["current_qty"], errors="coerce").fillna(0).astype(int)

    so = pd.to_numeric(df["sort_order"], errors="coerce")
    filler = pd.Series(range(len(df)), index=df.index)
    df["sort_order"] = so.fillna(filler).astype(int)

    return df.reset_index(drop=True)

def write_catalog(df: pd.DataFrame):
    df.to_csv(CATALOG_PATH, index=False)

# ---------------- Supabase data helpers ----------------
def append_log(order_df: pd.DataFrame, orderer: str) -> str:
    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    rows = []
    for _, r in order_df.iterrows():
        rows.append({
            "item": r["item"],
            "product_number": str(r["product_number"]),
            "qty": int(r["qty"]),
            "ordered_at": now,
            "orderer": orderer
        })
    res = supabase.table("orders_log").insert(rows).execute()
    if res.error:
        st.error(f"Supabase insert error: {res.error}")
    return now

def read_log() -> pd.DataFrame:
    res = supabase.table("orders_log").select("*").order("ordered_at", desc=True).execute()
    if not res.data:
        return pd.DataFrame(columns=["item","product_number","qty","ordered_at","orderer"])
    return pd.DataFrame(res.data)

def write_last(order_df: pd.DataFrame, orderer: str):
    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    supabase.table("last_order").delete().neq("id", 0).execute()
    rows = []
    for _, r in order_df.iterrows():
        rows.append({
            "item": r["item"],
            "product_number": str(r["product_number"]),
            "qty": int(r["qty"]),
            "generated_at": now,
            "orderer": orderer
        })
    supabase.table("last_order").insert(rows).execute()

def read_last() -> pd.DataFrame:
    res = supabase.table("last_order").select("*").order("generated_at", desc=True).execute()
    if not res.data:
        return pd.DataFrame(columns=["item","product_number","qty","generated_at","orderer"])
    return pd.DataFrame(res.data)

# ---------------- Emails CSV ----------------
@st.cache_data
def read_emails() -> pd.DataFrame:
    df = safe_read_csv(EMAILS_PATH)
    if df.empty:
        return pd.DataFrame(columns=["name", "email"])
    df.columns = [str(c).strip().lower() for c in df.columns]
    email_re = re.compile(r'([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})')
    out_rows = []
    if "email" in df.columns:
        for _, r in df.iterrows():
            raw = str(r.get("email", ""))
            m = email_re.search(raw)
            if m:
                out_rows.append({"name": str(r.get("name","")), "email": m.group(1)})
    return pd.DataFrame(out_rows)

def all_recipients(emails_df: pd.DataFrame) -> list[str]:
    cfg = get_smtp_config()
    file_recipients = emails_df["email"].tolist() if not emails_df.empty else []
    recipients = {e for e in file_recipients if e} | {e for e in cfg.get("default_to", []) if e}
    return sorted({e for e in recipients if "@" in e})

# ---------------- Session state ----------------
if "orderer" not in st.session_state:
    st.session_state["orderer"] = None
if "qty_map" not in st.session_state:
    st.session_state["qty_map"] = {}  # {product_number: qty}

# ---------------- UI ----------------
st.title("📦 Supply Ordering & Inventory Tracker")

people = read_people()
emails_df = read_emails()
catalog = read_catalog()
logs = read_log()
last_order_df = read_last()

email_ready = "✅" if smtp_ok() else "❌"
st.caption(f"Loaded {len(catalog)} catalog rows • {len(logs)} log rows • Email configured: {email_ready}")

tabs = st.tabs(["Create Order", "Adjust Inventory", "Catalog", "Order Logs"])

# ---------- Create Order ----------
with tabs[0]:
    if catalog.empty:
        st.info("No catalog found.")
    else:
        c1, c2 = st.columns([2, 3])
        with c1:
            current_orderer = st.session_state.get("orderer") or (people[0] if people else "Unknown")
            orderer = st.selectbox("Who is ordering?", options=(people if people else ["Unknown"]),
                                   index=(people.index(current_orderer) if people and current_orderer in people else 0))
            st.session_state["orderer"] = orderer
        with c2:
            search = st.text_input("Search items")

        # Build table with stable keys
        table = catalog.copy()
        table["product_number"] = table["product_number"].astype(str)
        table["qty"] = table["product_number"].map(st.session_state["qty_map"]).fillna(0).astype(int)

        if search:
            table = table[table["item"].str.contains(search, case=False, na=False)]

        table = table.reset_index(drop=True)
        table["_row_key"] = table["product_number"]

        edited = st.data_editor(
            table[["qty", "item", "product_number", "multiplier", "items_per_order", "current_qty"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "qty": st.column_config.NumberColumn("Qty", min_value=0, step=1),
                "item": st.column_config.TextColumn("Item", disabled=True),
                "product_number": st.column_config.TextColumn("Product #", disabled=True),
                "multiplier": st.column_config.NumberColumn("Multiplier", disabled=True),
                "items_per_order": st.column_config.NumberColumn("Items/Order", disabled=True),
                "current_qty": st.column_config.NumberColumn("Current Qty", disabled=True),
            },
            key="order_editor",
        )

        # Save edits back into session qty_map
        for _, r in edited.iterrows():
            st.session_state["qty_map"][str(r["product_number"])] = int(r["qty"])

        selected = edited[edited["qty"] > 0]
        if st.button("🧾 Generate & Log Order"):
            if not selected.empty:
                write_last(selected, orderer)
                when_str = append_log(selected, orderer)
                if smtp_ok():
                    recipients = all_recipients(emails_df)
                    if recipients:
                        lines = [f"- {r['item']} (#{r['product_number']}): {r['qty']}" for _, r in selected.iterrows()]
                        body = "\n".join([f"New supply order at {when_str}", f"Ordered by: {orderer}", "", *lines])
                        try:
                            send_email("Supply Order Logged", body, recipients)
                            st.success(f"Emailed {len(recipients)} recipient(s).")
                        except Exception as e:
                            st.error(f"Email failed: {e}")
                st.session_state["qty_map"] = {}
                st.rerun()
