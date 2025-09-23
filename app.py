import streamlit as st
import pandas as pd
from pandas.errors import EmptyDataError
from datetime import datetime
from pathlib import Path
import re
import smtplib, ssl
from email.message import EmailMessage
import base64, json, requests

st.set_page_config(page_title="Supply Ordering", page_icon="", layout="wide")

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

# ---------------- GitHub Sync ----------------
def gh_config():
    try:
        gh = st.secrets["github"]
        return {
            "token": gh.get("token"),
            "repo": gh.get("repo"),
            "branch": gh.get("branch", "main"),
        }
    except Exception:
        return {}

def gh_api_url(path: str) -> str:
    cfg = gh_config()
    return f"https://api.github.com/repos/{cfg['repo']}/contents/{path}"

def gh_read(path: str) -> bytes | None:
    cfg = gh_config()
    if not cfg.get("token"):
        return None
    url = gh_api_url(path)
    r = requests.get(url, headers={"Authorization": f"token {cfg['token']}"})
    if r.status_code == 200:
        data = r.json()
        return base64.b64decode(data["content"])
    return None

def gh_write(path: str, content_bytes: bytes, message: str):
    cfg = gh_config()
    if not cfg.get("token"):
        return
    url = gh_api_url(path)
    headers = {"Authorization": f"token {cfg['token']}"}
    sha = None
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        sha = r.json().get("sha")
    payload = {
        "message": message,
        "branch": cfg["branch"],
        "content": base64.b64encode(content_bytes).decode("utf-8"),
    }
    if sha:
        payload["sha"] = sha
    r = requests.put(url, headers=headers, data=json.dumps(payload))
    if r.status_code not in (200,201):
        st.error(f"GitHub write failed for {path}: {r.text}")

# Pull GitHub copies at startup
for fname, local_path, cols in [
    ("data/order_log.csv", LOG_PATH, ORDER_LOG_COLUMNS),
    ("data/last_order.csv", LAST_PATH, LAST_ORDER_COLUMNS),
]:
    remote = gh_read(fname)
    if remote:
        local_path.write_bytes(remote)
    else:
        if not local_path.exists():
            pd.DataFrame(columns=cols).to_csv(local_path, index=False)

# ---------------- Robust file helpers ----------------
def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding="utf-8-sig", **kwargs)
        except Exception:
            return pd.read_csv(path, encoding="latin-1", **kwargs)
    except EmptyDataError:
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
        raise RuntimeError("No recipients (emails.csv empty and [smtp].to not set).")

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
    df = df[(df["item"] != "") & (df["product_number"] != "")]
    return df.reset_index(drop=True)

def write_catalog(df: pd.DataFrame):
    df.to_csv(CATALOG_PATH, index=False)

def read_log() -> pd.DataFrame:
    df = safe_read_csv(LOG_PATH)
    if df.empty:
        return pd.DataFrame(columns=ORDER_LOG_COLUMNS)
    df["ordered_at"] = pd.to_datetime(df["ordered_at"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype(int)
    return df[ORDER_LOG_COLUMNS].sort_values("ordered_at", ascending=False)

def append_log(order_df: pd.DataFrame, orderer: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = order_df.copy()
    df["ordered_at"] = now
    df["orderer"] = orderer
    expected = ORDER_LOG_COLUMNS
    prev = safe_read_csv(LOG_PATH)
    if not prev.empty:
        combined = pd.concat([prev[expected], df[expected]], ignore_index=True)
    else:
        combined = df[expected]
    combined.to_csv(LOG_PATH, index=False)
    gh_write("data/order_log.csv", combined.to_csv(index=False).encode("utf-8"), f"Update order_log {now}")
    return now

def read_last() -> pd.DataFrame:
    df = safe_read_csv(LAST_PATH)
    if df.empty:
        return pd.DataFrame(columns=LAST_ORDER_COLUMNS)
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype(int)
    return df[LAST_ORDER_COLUMNS]

def write_last(df: pd.DataFrame, orderer: str):
    out = df.copy()
    out["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out["orderer"] = orderer
    out = out[LAST_ORDER_COLUMNS]
    out.to_csv(LAST_PATH, index=False)
    gh_write("data/last_order.csv", out.to_csv(index=False).encode("utf-8"), f"Update last_order {orderer}")

def last_info_map() -> pd.DataFrame:
    logs = read_log()
    if logs.empty:
        return pd.DataFrame(columns=["item","product_number","last_ordered_at","last_qty","last_orderer"])
    logs = logs.sort_values("ordered_at")
    tail = logs.groupby(["item","product_number"], as_index=False).tail(1)
    return tail.rename(columns={"ordered_at":"last_ordered_at","qty":"last_qty","orderer":"last_orderer"})

# ---------------- Emails CSV ----------------
@st.cache_data
def read_emails() -> pd.DataFrame:
    df = safe_read_csv(EMAILS_PATH)
    if df.empty:
        return pd.DataFrame(columns=["name", "email"])
    df.columns = [str(c).strip().lower() for c in df.columns]
    email_re = re.compile(r'([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})')
    def extract_email(s: str) -> str:
        m = email_re.search(str(s or ""))
        return m.group(1) if m else ""
    out_rows = []
    if "email" in df.columns:
        for _, r in df.iterrows():
            email = extract_email(r.get("email", ""))
            if email:
                out_rows.append({"name": r.get("name",""), "email": email})
    else:
        for _, r in df.iterrows():
            email = extract_email(r.iloc[0])
            if email:
                out_rows.append({"name":"", "email": email})
    return pd.DataFrame(out_rows).drop_duplicates(subset=["email"]).reset_index(drop=True)

def all_recipients(emails_df: pd.DataFrame) -> list[str]:
    cfg = get_smtp_config()
    file_recipients = emails_df["email"].tolist() if not emails_df.empty else []
    return sorted(set(file_recipients + cfg.get("default_to", [])))

# ---------------- Session state ----------------
if "orderer" not in st.session_state:
    st.session_state["orderer"] = None
if "quantities" not in st.session_state:
    st.session_state["quantities"] = {}
def qkey(item: str, pn: str) -> str:
    return f"{item}||{pn}"

# ---------------- UI ----------------
st.title("📦 Supply Ordering & Inventory Tracker")

people = read_people()
emails_df = read_emails()
catalog = read_catalog()
logs = read_log()
last_order_df = read_last()

st.caption(
    f"Loaded {len(catalog)} catalog rows • {len(logs)} log rows • "
    f"Email: {'✅' if smtp_ok() else '❌'} • Recipients: {len(all_recipients(emails_df))}"
)

tabs = st.tabs(["Create Order", "Adjust Inventory", "Catalog", "Order Logs", "Tools"])

# --- Tab 0: Create Order ---
with tabs[0]:
    st.subheader("Create New Order")
    if catalog.empty:
        st.info("No catalog found.")
    else:
        # search + select orderer
        orderer = st.selectbox("Orderer", people or ["(add names in data/people.txt)"])
        st.session_state["orderer"] = orderer
        search = st.text_input("Search items")
        table = catalog.copy()
        if search:
            table = table[table["item"].str.contains(search, case=False, na=False)]
        table["qty"] = [st.session_state["quantities"].get(qkey(r["item"], r["product_number"]), 0) for _, r in table.iterrows()]
        edited = st.data_editor(
            table[["qty","item","product_number"]],
            hide_index=True,
            column_config={"qty": st.column_config.NumberColumn("Qty", min_value=0, step=1)},
        )
        for _, r in edited.iterrows():
            st.session_state["quantities"][qkey(r["item"], r["product_number"])] = int(r["qty"])
        if st.button("🧾 Log Order"):
            rows = [{"item":i.split("||")[0],"product_number":i.split("||")[1],"qty":q} for i,q in st.session_state["quantities"].items() if q>0]
            if rows:
                df = pd.DataFrame(rows)
                write_last(df, orderer)
                append_log(df, orderer)
                st.session_state["quantities"] = {}
                st.success("Order logged and synced to GitHub.")

# --- Other tabs remain same (Adjust Inventory, Catalog, Order Logs, Tools) ---
