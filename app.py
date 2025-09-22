import streamlit as st
import pandas as pd
from pandas.errors import EmptyDataError
from datetime import datetime
from pathlib import Path
import re
import smtplib, ssl
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

# ---------------- File helpers ----------------
def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except (FileNotFoundError, EmptyDataError):
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Couldn't read {path.name}: {e}")
        return pd.DataFrame()

def ensure_headers(path: Path, columns: list[str]):
    if (not path.exists()) or path.stat().st_size == 0:
        pd.DataFrame(columns=columns).to_csv(path, index=False)

ensure_headers(LOG_PATH, ORDER_LOG_COLUMNS)
ensure_headers(LAST_PATH, LAST_ORDER_COLUMNS)

# ---------------- SMTP ----------------
def get_smtp_config():
    s = st.secrets.get("smtp", {})
    host = s.get("server") or s.get("host")
    port = int(s.get("port", 465))
    username = s.get("username") or s.get("user")  # handle "user" from secrets.toml
    password = (s.get("password") or "").replace(" ", "")
    mail_from = s.get("from") or username or ""
    subject_prefix = s.get("subject_prefix", "")
    default_to = s.get("to", "")
    force_from_user = bool(s.get("force_from_user", False))

    if "use_ssl" in s:
        use_ssl = bool(s.get("use_ssl"))
    else:
        use_ssl = (port == 465)

    if force_from_user or ("gmail.com" in (username or "")):
        mail_from = username or mail_from

    return {
        "host": host,
        "port": port,
        "username": username,
        "password": password,
        "from": mail_from,
        "subject_prefix": subject_prefix,
        "default_to": default_to,
        "use_ssl": use_ssl,
    }

def smtp_ok() -> bool:
    cfg = get_smtp_config()
    required = ["host", "port", "username", "password", "from"]
    return all(cfg.get(k) for k in required)

def send_email(subject: str, body: str, to_emails: list[str]):
    cfg = get_smtp_config()
    if not to_emails:
        raise RuntimeError("No recipients provided.")

    msg = EmailMessage()
    if cfg["subject_prefix"]:
        msg["Subject"] = f'{cfg["subject_prefix"]}{subject}'
    else:
        msg["Subject"] = subject
    msg["From"] = cfg["from"]
    msg["To"] = ", ".join(to_emails)
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
        return pd.DataFrame(columns=["item", "product_number", "current_qty", "sort_order"])
    for c in ["item", "product_number", "current_qty", "sort_order"]:
        if c not in df.columns:
            df[c] = pd.NA
    df["item"] = df["item"].astype(str).str.strip()
    df["product_number"] = df["product_number"].astype(str).str.strip()
    df["current_qty"] = pd.to_numeric(df["current_qty"], errors="coerce").fillna(0).astype(int)
    so = pd.to_numeric(df["sort_order"], errors="coerce")
    filler = pd.Series(range(len(df)), index=df.index)
    df["sort_order"] = so.fillna(filler).astype(int)
    df = df[(df["item"] != "") & (df["product_number"] != "")]
    return df[["item", "product_number", "current_qty", "sort_order"]].reset_index(drop=True)

def write_catalog(df: pd.DataFrame):
    df = df.copy()
    df["item"] = df["item"].astype(str)
    df["product_number"] = df["product_number"].astype(str)
    df["current_qty"] = pd.to_numeric(df.get("current_qty", 0), errors="coerce").fillna(0).astype(int)
    so = pd.to_numeric(df.get("sort_order", pd.Series(range(len(df)))), errors="coerce")
    df["sort_order"] = so.fillna(pd.Series(range(len(df)), index=df.index)).astype(int)
    df.to_csv(CATALOG_PATH, index=False)

def read_log() -> pd.DataFrame:
    df = safe_read_csv(LOG_PATH)
    if df.empty:
        return pd.DataFrame(columns=ORDER_LOG_COLUMNS)
    for c in ORDER_LOG_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    df["ordered_at"] = pd.to_datetime(df["ordered_at"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype(int)
    return df[ORDER_LOG_COLUMNS].sort_values("ordered_at", ascending=False)

def append_log(order_df: pd.DataFrame, orderer: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = order_df.copy()
    df["ordered_at"] = now
    df["orderer"] = orderer
    df = df[ORDER_LOG_COLUMNS]
    prev = safe_read_csv(LOG_PATH)
    combined = pd.concat([prev, df], ignore_index=True) if not prev.empty else df
    combined.to_csv(LOG_PATH, index=False)
    return now

def read_last() -> pd.DataFrame:
    df = safe_read_csv(LAST_PATH)
    if df.empty:
        return pd.DataFrame(columns=LAST_ORDER_COLUMNS)
    return df[LAST_ORDER_COLUMNS]

def write_last(df: pd.DataFrame, orderer: str):
    out = df.copy()
    out["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out["orderer"] = orderer
    out = out[LAST_ORDER_COLUMNS]
    out.to_csv(LAST_PATH, index=False)

def last_info_map() -> pd.DataFrame:
    logs = read_log()
    if logs.empty:
        return pd.DataFrame(columns=["item","product_number","last_ordered_at","last_qty","last_orderer"])
    tail = logs.groupby(["item","product_number"], as_index=False).tail(1)
    return tail.rename(columns={"ordered_at":"last_ordered_at","qty":"last_qty","orderer":"last_orderer"})

# ---------------- Emails CSV ----------------
def read_emails() -> pd.DataFrame:
    df = safe_read_csv(EMAILS_PATH)
    if df.empty:
        return pd.DataFrame(columns=["name", "email"])
    df.columns = [str(c).strip().lower() for c in df.columns]
    email_re = re.compile(r'([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})')
    out_rows = []
    for _, r in df.iterrows():
        s = " ".join(str(x) for x in r if pd.notna(x))
        m = email_re.search(s)
        if m:
            out_rows.append({"name": "", "email": m.group(1)})
    return pd.DataFrame(out_rows).drop_duplicates(subset=["email"]).reset_index(drop=True)

def all_recipients(emails_df: pd.DataFrame) -> list[str]:
    return sorted(emails_df["email"].astype(str).unique().tolist()) if not emails_df.empty else []

# ---------------- Persisted qty ----------------
def qkey(item: str, pn: str) -> str:
    return f"{item}||{str(pn)}"

if "qty_map" not in st.session_state:
    st.session_state["qty_map"] = {}
if "editor_key" not in st.session_state:
    st.session_state["editor_key"] = 0

# ---------------- UI ----------------
st.title("📦 Supply Ordering & Inventory Tracker")

people = read_people()
emails_df = read_emails()
catalog = read_catalog()
logs = read_log()
last_order_df = read_last()

st.caption(
    f"Loaded {len(catalog)} catalog rows • {len(logs)} log rows • "
    f"Email configured: {'✅' if smtp_ok() else '❌'} • Recipients found: "
    f"{0 if emails_df.empty else len(emails_df)}"
)

with st.sidebar:
    st.write("SMTP config loaded:", get_smtp_config())

tabs = st.tabs(["Create Order", "Adjust Inventory", "Catalog", "Order Logs", "Tools"])

# ---------- Create Order ----------
with tabs[0]:
    if catalog.empty:
        st.info("No catalog found.")
    else:
        c1, c2, c3 = st.columns([2, 2, 3])
        with c1:
            orderer = st.selectbox("Who is ordering?", people or ["(add names in people.txt)"], key="orderer")
        with c2:
            search = st.text_input("Search items", key="order_search")
        with c3:
            if st.button("🧼 Clear quantities", use_container_width=True):
                st.session_state["qty_map"] = {}
                st.session_state["editor_key"] += 1
                st.rerun()

        table = catalog.merge(last_info_map(), on=["item","product_number"], how="left")
        if search:
            table = table[table["item"].str.contains(search, case=False, na=False)]

        qty_map = st.session_state["qty_map"]
        table["qty"] = table.apply(lambda r: qty_map.get(qkey(r["item"], r["product_number"]), 0), axis=1)

        edited = st.data_editor(
            table[["qty","item","product_number","last_ordered_at","last_qty","last_orderer"]],
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            key=f"editor_{st.session_state['editor_key']}",
            column_config={"qty": st.column_config.NumberColumn("Qty", min_value=0, step=1)},
        )

        # force sync
        for _, r in edited.iterrows():
            st.session_state["qty_map"][qkey(r["item"], r["product_number"])] = int(r["qty"]) if pd.notna(r["qty"]) else 0

        def _log_and_email(order_df: pd.DataFrame):
            when_str = append_log(order_df, orderer)
            write_last(order_df, orderer)
            recips = all_recipients(emails_df)
            if smtp_ok() and recips:
                lines = [f"- {r['item']} (#{r['product_number']}): {r['qty']}" for _, r in order_df.iterrows()]
                body = f"New order at {when_str} by {orderer}\n\n" + "\n".join(lines)
                send_email("Supply Order Logged", body, recips)
                st.success(f"Emailed {len(recips)} recipient(s).")
            st.session_state["qty_map"] = {}
            st.session_state["editor_key"] += 1
            st.rerun()

        if st.button("🧾 Log Order", use_container_width=True):
            rows = [{"item": i.split("||")[0], "product_number": i.split("||")[1], "qty": q}
                    for i, q in st.session_state["qty_map"].items() if q > 0]
            if rows:
                _log_and_email(pd.DataFrame(rows))

# ---------- Adjust Inventory ----------
with tabs[1]:
    if not catalog.empty:
        edited = st.data_editor(catalog, use_container_width=True, hide_index=True, key="inv_editor")
        if st.button("💾 Save inventory"):
            write_catalog(edited)
            st.success("Inventory saved.")

# ---------- Catalog ----------
with tabs[2]:
    st.dataframe(catalog, use_container_width=True, hide_index=True)

# ---------- Order Logs ----------
with tabs[3]:
    st.dataframe(logs, use_container_width=True, hide_index=True)

# ---------- Tools ----------
with tabs[4]:
    if st.button("✉️ Test Email"):
        recips = all_recipients(emails_df)
        if smtp_ok() and recips:
            send_email("Test — Supply App", "This is a test.", recips)
            st.success("Test email sent.")
        else:
            st.error("SMTP not configured or no recipients.")
