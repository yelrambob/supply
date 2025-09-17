import streamlit as st
import pandas as pd
from pandas.errors import EmptyDataError
from datetime import datetime
from pathlib import Path
from shutil import copy2
import io
import smtplib, ssl
from email.message import EmailMessage

st.set_page_config(page_title="Supply Ordering", page_icon="📦", layout="wide")

# ---------- Paths (absolute to this file) ----------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CATALOG_PATH = DATA_DIR / "catalog.csv"
CATALOG_DEFAULT_PATH = DATA_DIR / "catalog.default.csv"
PEOPLE_PATH = DATA_DIR / "people.txt"
PEOPLE_DEFAULT_PATH = DATA_DIR / "people.default.txt"
EMAILS_PATH = DATA_DIR / "emails.csv"

ORDER_LOG_PATH = DATA_DIR / "order_log.csv"
ORDER_LOG_COLUMNS = ["item", "product_number", "qty", "ordered_at", "orderer"]

# ---------- Seed files ----------
def ensure_seed_files():
    if (not CATALOG_PATH.exists()) or (CATALOG_PATH.stat().st_size == 0):
        if CATALOG_DEFAULT_PATH.exists() and CATALOG_DEFAULT_PATH.stat().st_size > 0:
            copy2(CATALOG_DEFAULT_PATH, CATALOG_PATH)
    if (not PEOPLE_PATH.exists()) or (PEOPLE_PATH.stat().st_size == 0):
        if PEOPLE_DEFAULT_PATH.exists() and PEOPLE_DEFAULT_PATH.stat().st_size > 0:
            copy2(PEOPLE_DEFAULT_PATH, PEOPLE_PATH)

# ---------- CSV helpers ----------
def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(
            path,
            encoding="utf-8",
            on_bad_lines="skip",
            engine="python",
            **kwargs
        )
    except EmptyDataError:
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not read {path.name}: {e}")
        return pd.DataFrame()

def _norm_colname(c: str) -> str:
    # Normalize headers: trim, lowercase, replace separators with underscores, collapse repeats
    c = str(c).lstrip("\ufeff").strip().lower()
    for ch in [" ", "-", "/", "\\", ".", ",", ":", ";", "\t"]:
        c = c.replace(ch, "_")
    while "__" in c:
        c = c.replace("__", "_")
    return c.strip("_")

def _best_text_col(df: pd.DataFrame, prefer: list[str]) -> str | None:
    for p in prefer:
        if p in df.columns:
            return p
    # otherwise choose the column with the most non-empty values
    best, best_count = None, -1
    for c in df.columns:
        s = df[c].astype(object).where(pd.notnull(df[c]), "")
        cnt = s.astype(str).str.strip().replace({"nan":"", "NaN":"", "None":""}).ne("").sum()
        if cnt > best_count:
            best, best_count = c, cnt
    return best

@st.cache_data
def load_catalog() -> pd.DataFrame:
    raw = safe_read_csv(CATALOG_PATH, header=0)
    if raw.empty:
        return pd.DataFrame(columns=["item", "product_number", "current_qty", "per_box_qty", "sort_order"])

    # Normalize headers robustly
    raw.columns = [_norm_colname(c) for c in raw.columns]

    # Drop fully-empty columns
    raw = raw.dropna(axis=1, how="all")

    # Prefer common names; "product number" becomes "product_number" via normalization
    item_col = _best_text_col(raw, ["item", "items", "name", "description", "desc"])
    pn_col   = _best_text_col(
        raw,
        [
            "product_number", "product_no", "productnum", "product_num", "productno",
            "productid", "product_id", "sku", "code", "id", "pn"
        ],
    )

    if not item_col or not pn_col or item_col == pn_col:
        st.error(
            "Could not reliably detect `item` and `product_number` in data/catalog.csv.\n"
            "Ensure two columns exist: item name and product number."
        )
        return pd.DataFrame(columns=["item", "product_number", "current_qty", "per_box_qty", "sort_order"])

    def clean_series(s: pd.Series) -> pd.Series:
        s = s.astype(object).where(pd.notnull(s), "")
        s = s.apply(lambda x: x if isinstance(x, str) else str(x))
        s = s.str.strip().replace({"nan":"", "NaN":"", "None":""})
        return s

    out = pd.DataFrame({
        "item": clean_series(raw[item_col]),
        "product_number": clean_series(raw[pn_col]),
    })

    # Optional columns
    for opt in ["current_qty", "per_box_qty", "sort_order"]:
        out[opt] = raw[opt] if opt in raw.columns else None

    # Drop rows missing essentials
    out = out[(out["item"] != "") & (out["product_number"] != "")]
    if out.empty:
        st.error("`catalog.csv` loaded but has no usable rows after cleanup (blank item names or product numbers).")
        return pd.DataFrame(columns=["item", "product_number", "current_qty", "per_box_qty", "sort_order"])

    # Stable sort
    out["sort_order"] = pd.to_numeric(out["sort_order"], errors="coerce")
    out = out.sort_values(by=["sort_order", "item"], ascending=[True, True], na_position="last").reset_index(drop=True)

    # Ensure product_number treated as string
    out["product_number"] = clean_series(out["product_number"])
    return out[["item", "product_number", "current_qty", "per_box_qty", "sort_order"]]

def load_people(path: Path) -> list:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception as e:
        st.warning(f"Could not read people file: {e}")
        return []

@st.cache_data
def load_emails() -> pd.DataFrame:
    df = safe_read_csv(EMAILS_PATH)
    if df.empty:
        return pd.DataFrame(columns=["name", "email"])
    df.columns = [_norm_colname(c) for c in df.columns]
    if "email" not in df.columns:
        return pd.DataFrame(columns=["name", "email"])
    if "name" not in df.columns:
        df["name"] = ""
    df["email"] = df["email"].astype(str).str.strip()
    df = df[df["email"].str.contains("@", na=False)]
    df = df.drop_duplicates(subset=["email"])
    return df[["name", "email"]]

def load_order_log() -> pd.DataFrame:
    df = safe_read_csv(ORDER_LOG_PATH)
    if df.empty:
        return pd.DataFrame(columns=ORDER_LOG_COLUMNS)
    for c in ORDER_LOG_COLUMNS:
        if c not in df.columns:
            df[c] = None
    try:
        df["ordered_at"] = pd.to_datetime(df["ordered_at"], errors="coerce")
    except Exception:
        pass
    df = df.sort_values(by="ordered_at", ascending=False, na_position="last").reset_index(drop=True)
    return df[ORDER_LOG_COLUMNS]

def save_order_log(new_rows: pd.DataFrame):
    if ORDER_LOG_PATH.exists() and ORDER_LOG_PATH.stat().st_size > 0:
        existing = safe_read_csv(ORDER_LOG_PATH)
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows.copy()
    combined.to_csv(ORDER_LOG_PATH, index=False)

# ---------- Email helpers ----------
def email_enabled() -> bool:
    try:
        s = st.secrets["smtp"]
        required = ["server", "port", "username", "password", "from"]
        return all(k in s and s[k] for k in required)
    except Exception:
        return False

def send_email(subject: str, body_text: str, to_emails: list, attachments: list = None):
    s = st.secrets["smtp"]
    context = ssl.create_default_context()
    msg = EmailMessage()
    msg["From"] = s["from"]
    msg["To"] = ", ".join(to_emails)
    if "reply_to" in s and s["reply_to"]:
        msg["Reply-To"] = s["reply_to"]
    prefix = s.get("subject_prefix", "")
    msg["Subject"] = f"{prefix}{subject}" if prefix else subject
    msg.set_content(body_text)

    if attachments:
        for fname, data_bytes, mimetype in attachments:
            parts = mimetype.split("/", 1)
            maintype = parts[0] if len(parts) > 0 else "application"
            subtype = parts[1] if len(parts) > 1 else "octet-stream"
            msg.add_attachment(data_bytes, maintype=maintype, subtype=subtype, filename=fname)

    with smtplib.SMTP_SSL(s["server"], int(s["port"]), context=context) as server:
        server.login(s["username"], s["password"])
        server.send_message(msg)

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

# ---------- Init ----------
ensure_seed_files()
catalog = load_catalog()
people = load_people(PEOPLE_PATH)
emails_df = load_emails()
order_log = load_order_log()

# Quick preview to confirm the list is loading
st.caption(f"Loaded {len(catalog)} catalog rows from `{CATALOG_PATH.name}`.")
if catalog.empty:
    st.stop()
st.dataframe(catalog.head(5), use_container_width=True, hide_index=True)

# Maintain qty state per product_number so it persists across searches/reruns
if "qty" not in st.session_state:
    st.session_state["qty"] = {}  # key: product_number (str) -> int

# ---------- Header / Controls ----------
left, mid, right = st.columns([3, 2, 2])
with left:
    st.title("📦 Supply Ordering")
with mid:
    orderer_options = people if people else [""]
    orderer = st.selectbox("Who is ordering?", options=orderer_options, index=0)
with right:
    search = st.text_input("Search items", placeholder="Type to filter… (quantities persist)")

# Apply filter (quantities remain in session state regardless)
if search:
    mask = catalog["item"].fillna("").str.contains(search, case=False, na=False)
    view = catalog[mask].copy()
else:
    view = catalog.copy()

# ---------- Current Cart (Top) ----------
def current_cart_rows():
    rows = []
    # Use full catalog so items not in view still count
    for _, r in catalog.iterrows():
        pn = str(r["product_number"])
        q = int(st.session_state["qty"].get(pn, 0) or 0)
        if q > 0:
            rows.append({"item": r["item"], "product_number": pn, "qty": q})
    return rows

cart_rows = current_cart_rows()
cart_df = pd.DataFrame(cart_rows, columns=["item", "product_number", "qty"])
total_items = int(cart_df["qty"].sum()) if not cart_df.empty else 0

st.markdown(f"### 🧾 Current Cart — **{total_items} total**")
cart_holder = st.empty()
if not cart_df.empty:
    cart_holder.dataframe(cart_df, use_container_width=True, hide_index=True)
else:
    cart_holder.info("No items selected yet. Add quantities below.")

# ---------- Catalog (editable qty) ----------
st.markdown("### Catalog")
st.caption("Tip: Use the search box to filter. Quantities remain saved even when you change the search.")

# Header row
h1, h2, h3, h4 = st.columns([6, 2, 2, 2])
with h1: st.markdown("**Item**")
with h2: st.markdown("**Current**")
with h3: st.markdown("**Per Box**")
with h4: st.markdown("**Order Qty**")

def render_row(row, i: int):
    pn = str(row["product_number"])
    item_name = str(row["item"])

    col1, col2, col3, col4 = st.columns([6, 2, 2, 2])
    with col1:
        st.write(item_name)
        st.caption(f"#{pn}")
    with col2:
        val = row.get("current_qty", "")
        st.write("" if pd.isna(val) else str(val))
        st.caption("Current Qty")
    with col3:
        val = row.get("per_box_qty", "")
        st.write("" if pd.isna(val) else str(val))
        st.caption("Per Box")
    with col4:
        # Unique widget key per row to avoid DuplicateElementKey,
        # while syncing central qty per product_number.
        widget_key = f"qty_{pn}_{i}"
        if widget_key not in st.session_state:
            st.session_state[widget_key] = int(st.session_state["qty"].get(pn, 0) or 0)
        new_val = st.number_input("Order Qty", key=widget_key, min_value=0, step=1, label_visibility="collapsed")
        st.session_state["qty"][pn] = int(new_val or 0)

for i, (_, row) in enumerate(view.iterrows()):
    with st.container(border=True):
        render_row(row, i)

st.divider()

# ---------- Log & Email ----------
def email_enabled_hint(enabled: bool):
    if not enabled:
        st.caption("Email disabled—configure `.streamlit/secrets.toml` with [smtp] to enable.")

email_ok = email_enabled()
recipients_df = emails_df.copy()
recipient_list = recipients_df["email"].tolist() if not recipients_df.empty else []

colA, colB = st.columns([1, 1])
with colA:
    email_toggle = st.checkbox("Email order to recipients in data/emails.csv",
                               value=True if recipient_list else False,
                               help="Uses SMTP settings from secrets.toml under [smtp]")
with colB:
    cc_orderer = st.checkbox("CC the orderer if found in emails.csv", value=True)

email_enabled_hint(email_ok)

log_btn = st.button("✅ Log Order (and Email)")

if log_btn:
    cart_rows = current_cart_rows()
    if not cart_rows:
        st.error("No items with quantity > 0 to log.")
    else:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_rows = []
        for r in cart_rows:
            new_rows.append({
                "item": r["item"],
                "product_number": r["product_number"],
                "qty": int(r["qty"]),
                "ordered_at": ts,
                "orderer": orderer or ""
            })
        new_df = pd.DataFrame(new_rows, columns=ORDER_LOG_COLUMNS)

        # Save to log
        try:
            save_order_log(new_df)
            st.success(f"Logged {len(new_rows)} line(s) at {ts} by {orderer or 'Unknown'}")
        except Exception as e:
            st.error(f"Could not save order log: {e}")

        # Email step
        if email_toggle and email_ok:
            to_emails = set(recipient_list)
            if cc_orderer and orderer:
                match = recipients_df[recipients_df["name"].str.strip().str.lower() == orderer.strip().lower()]
                if not match.empty:
                    to_emails.update(match["email"].tolist())
            to_emails = [e for e in sorted(to_emails) if e]

            if to_emails:
                body_lines = [
                    f"New supply order logged at {ts}",
                    f"Ordered by: {orderer or 'Unknown'}",
                    "",
                    "Items:",
                ] + [f"- {r['item']} (#{r['product_number']}): {r['qty']}" for r in cart_rows]
                body_text = "\n".join(body_lines)

                attach_bytes = df_to_csv_bytes(new_df)
                attachments = [("order_log_entry.csv", attach_bytes, "text/csv")]

                try:
                    send_email(
                        subject="Supply Order Logged",
                        body_text=body_text,
                        to_emails=to_emails,
                        attachments=attachments
                    )
                    st.info(f"Emailed {len(to_emails)} recipient(s).")
                except Exception as e:
                    st.error(f"Could not send email: {e}")
            else:
                st.warning("No valid email recipients found in data/emails.csv.")
        elif email_toggle and not email_ok:
            st.warning("Email is enabled but SMTP settings are missing/invalid in secrets.toml [smtp].")
        else:
            st.caption("Email step skipped.")

        # Reset all quantities after logging
        for _, r in catalog.iterrows():
            pn = str(r["product_number"])
            st.session_state["qty"][pn] = 0
        # Clear visible widgets in the current view so the UI resets
        for i, (_, r) in enumerate(view.iterrows()):
            pn = str(r["product_number"])
            wkey = f"qty_{pn}_{i}"
            if wkey in st.session_state:
                st.session_state[wkey] = 0

        cart_holder.info("Cart cleared after logging. Add new quantities to create another order.")
        order_log = load_order_log()

# ---------- Recent Orders ----------
st.markdown("### Recent Orders (Most Recent First)")
if order_log.empty:
    st.info("No orders logged yet.")
else:
    try:
        order_log["ordered_at"] = pd.to_datetime(order_log["ordered_at"], errors="coerce")
    except Exception:
        pass
    order_log = order_log.sort_values(by="ordered_at", ascending=False, na_position="last").reset_index(drop=True)
    st.dataframe(order_log, use_container_width=True, hide_index=True)
