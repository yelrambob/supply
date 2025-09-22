import streamlit as st
import pandas as pd
from pandas.errors import EmptyDataError
from datetime import datetime
from pathlib import Path
import re, smtplib, ssl
from email.message import EmailMessage

st.set_page_config(page_title="Supply Ordering", page_icon="📦", layout="wide")

# ---------------- Paths ----------------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CATALOG_PATH = DATA_DIR / "catalog.csv"
LOG_PATH     = DATA_DIR / "order_log.csv"
LAST_PATH    = DATA_DIR / "last_order.csv"
PEOPLE_PATH  = DATA_DIR / "people.txt"
EMAILS_PATH  = DATA_DIR / "emails.csv"

ORDER_LOG_COLUMNS = ["item", "product_number", "qty", "ordered_at", "orderer"]
LAST_ORDER_COLUMNS = ["item", "product_number", "qty", "generated_at", "orderer"]

# ---------------- CSV helpers ----------------
def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except (FileNotFoundError, EmptyDataError):
        return pd.DataFrame()

def ensure_headers(path: Path, cols: list[str]):
    if not path.exists() or path.stat().st_size == 0:
        pd.DataFrame(columns=cols).to_csv(path, index=False)

ensure_headers(LOG_PATH, ORDER_LOG_COLUMNS)
ensure_headers(LAST_PATH, LAST_ORDER_COLUMNS)

# ---------------- SMTP ----------------
def get_smtp_config():
    s = st.secrets.get("smtp", {})
    host = s.get("server") or s.get("host")
    port = int(s.get("port", 465))
    user = s.get("username") or s.get("user")
    pwd = (s.get("password") or "").replace(" ", "")
    mail_from = s.get("from") or user
    prefix = s.get("subject_prefix", "")
    use_ssl = bool(s.get("use_ssl", port == 465))
    return {"host": host, "port": port, "user": user, "pwd": pwd, "from": mail_from, "prefix": prefix, "ssl": use_ssl}

def smtp_ok(): return all(get_smtp_config().get(k) for k in ["host","port","user","pwd","from"])

def send_email(subject, body, to_emails):
    cfg = get_smtp_config()
    msg = EmailMessage()
    msg["Subject"] = f"{cfg['prefix']}{subject}" if cfg["prefix"] else subject
    msg["From"] = cfg["from"]
    msg["To"] = ", ".join(to_emails)
    msg.set_content(body)
    if cfg["ssl"]:
        with smtplib.SMTP_SSL(cfg["host"], cfg["port"], context=ssl.create_default_context()) as s:
            s.login(cfg["user"], cfg["pwd"]); s.send_message(msg)
    else:
        with smtplib.SMTP(cfg["host"], cfg["port"]) as s:
            s.starttls(context=ssl.create_default_context()); s.login(cfg["user"], cfg["pwd"]); s.send_message(msg)

# ---------------- Loaders ----------------
@st.cache_data
def read_people():
    return PEOPLE_PATH.read_text(encoding="utf-8").splitlines() if PEOPLE_PATH.exists() else []

@st.cache_data
def read_catalog():
    df = safe_read_csv(CATALOG_PATH)
    if df.empty: return pd.DataFrame(columns=["item","product_number","current_qty","sort_order"])
    for c in ["item","product_number","current_qty","sort_order"]:
        if c not in df: df[c] = pd.NA
    df["item"] = df["item"].astype(str).str.strip()
    df["product_number"] = df["product_number"].astype(str).str.strip()
    df["current_qty"] = pd.to_numeric(df["current_qty"], errors="coerce").fillna(0).astype(int)
    return df.reset_index(drop=True)

def write_catalog(df): df.to_csv(CATALOG_PATH, index=False)

def read_log():
    df = safe_read_csv(LOG_PATH)
    return df if not df.empty else pd.DataFrame(columns=ORDER_LOG_COLUMNS)

def append_log(order_df, orderer):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = order_df.copy(); df["ordered_at"] = now; df["orderer"] = orderer
    pd.concat([read_log(), df], ignore_index=True).to_csv(LOG_PATH, index=False)
    return now

def read_last():
    df = safe_read_csv(LAST_PATH)
    return df if not df.empty else pd.DataFrame(columns=LAST_ORDER_COLUMNS)

def write_last(df, orderer):
    out = df.copy(); out["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S"); out["orderer"] = orderer
    out.to_csv(LAST_PATH, index=False)

def read_emails():
    df = safe_read_csv(EMAILS_PATH)
    if df.empty: return []
    emails = []
    for col in df.columns:
        for v in df[col]:
            if pd.isna(v): continue
            found = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", str(v))
            emails.extend(found)
    return sorted(set(emails))

# ---------------- State helpers ----------------
if "qty_map" not in st.session_state: st.session_state["qty_map"] = {}
def qkey(i,p): return f"{i}||{p}"
def clear_qty(): st.session_state["qty_map"] = {}; st.session_state.pop("order_editor", None)

# ---------------- UI ----------------
st.title("📦 Supply Ordering")

people, catalog, logs, last_df, emails = read_people(), read_catalog(), read_log(), read_last(), read_emails()
st.caption(f"{len(catalog)} items • {len(logs)} logs • Email: {'✅' if smtp_ok() else '❌'} • Recipients: {len(emails)}")

tabs = st.tabs(["Create Order","Logs","Catalog","Tools"])

# ---------- Create Order ----------
with tabs[0]:
    orderer = st.selectbox("Who is ordering?", options=people or ["(add names in people.txt)"])
    if st.button("Clear quantities"): clear_qty()
    if catalog.empty: st.warning("No catalog.csv found")
    else:
        table = catalog.copy()
        table["qty"] = [st.session_state["qty_map"].get(qkey(r.item,r.product_number),0) for r in table.itertuples()]
        with st.form("order_form"):
            edited = st.data_editor(table[["qty","item","product_number"]], hide_index=True, use_container_width=True, key="order_editor")
            c1,c2 = st.columns(2)
            log_btn = c1.form_submit_button("Log Order")
            log_dec_btn = c2.form_submit_button("Log & Decrement")
        # update state
        for r in edited.itertuples():
            st.session_state["qty_map"][qkey(r.item,r.product_number)] = int(r.qty or 0)
        if log_btn or log_dec_btn:
            chosen = edited[edited.qty>0][["item","product_number","qty"]]
            if chosen.empty: st.error("No qty >0"); st.stop()
            if "(add" in orderer: st.error("Pick valid orderer"); st.stop()
            write_last(chosen, orderer); when = append_log(chosen, orderer)
            if log_dec_btn:
                for r in chosen.itertuples():
                    mask = (catalog["item"]==r.item)&(catalog["product_number"]==r.product_number)
                    catalog.loc[mask,"current_qty"] = catalog.loc[mask,"current_qty"]-r.qty
                write_catalog(catalog)
            if smtp_ok() and emails:
                body = "\n".join([f"- {r.item} #{r.product_number}: {r.qty}" for r in chosen.itertuples()])
                try:
                    send_email("Supply Order", f"Ordered by {orderer} at {when}\n\n{body}", emails)
                    st.success(f"Emailed {len(emails)} recipient(s)")
                except Exception as e: st.error(f"Email failed: {e}")
            clear_qty(); st.rerun()

# ---------- Logs ----------
with tabs[1]:
    st.dataframe(logs, use_container_width=True) if not logs.empty else st.info("No logs")

# ---------- Catalog ----------
with tabs[2]:
    st.dataframe(catalog, use_container_width=True) if not catalog.empty else st.info("No catalog")

# ---------- Tools ----------
with tabs[3]:
    if st.button("Clear logs"): ensure_headers(LOG_PATH, ORDER_LOG_COLUMNS); st.success("Logs cleared")
    if st.button("Clear last order"): ensure_headers(LAST_PATH, LAST_ORDER_COLUMNS); st.success("Last order cleared")
    if st.button("Test email"):
        try: send_email("Test","This is a test",emails); st.success("Sent")
        except Exception as e: st.error(e)
