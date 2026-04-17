import streamlit as st
import pandas as pd
from pandas.errors import EmptyDataError
import zoneinfo
from datetime import datetime
from pathlib import Path
import re
import smtplib, ssl
from email.message import EmailMessage
from supabase import create_client

st.set_page_config(page_title="Supply Ordering", page_icon="📦", layout="wide")

NYC = zoneinfo.ZoneInfo("America/New_York")

# ---------------- Paths ----------------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CATALOG_PATH = DATA_DIR / "catalog.csv"
PEOPLE_PATH  = DATA_DIR / "people.txt"
EMAILS_PATH  = DATA_DIR / "emails.csv"

# ---------------- Supabase ----------------
@st.cache_resource
def get_supabase():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

supabase = get_supabase()

# ---------------- File helpers ----------------
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
    return [p.strip() for p in re.split(r"[;,]\s*", str(txt)) if p.strip()]

def get_smtp_config() -> dict:
    try:
        s = st.secrets["smtp"]
        return {
            "host":           s.get("host"),
            "port":           int(s.get("port", 587)),
            "username":       s.get("user"),
            "password":       s.get("password", "").replace(" ", ""),
            "from":           s.get("from"),
            "subject_prefix": s.get("subject_prefix", ""),
            "default_to":     _split_emails(s.get("to", "")) if s.get("to") else [],
            "use_ssl":        bool(s.get("use_ssl", False)),
        }
    except Exception as e:
        st.error(f"Error reading SMTP config: {e}")
        return {}

def smtp_ok() -> bool:
    cfg = get_smtp_config()
    return all(cfg.get(k) for k in ["host", "port", "username", "password", "from"])

def send_email(subject: str, body: str, to_emails: list[str] | None):
    cfg = get_smtp_config()
    recipients = sorted({
        e for e in (to_emails or []) + cfg.get("default_to", [])
        if e and "@" in e
    })
    if not recipients:
        raise RuntimeError("No recipients found.")

    msg = EmailMessage()
    msg["Subject"] = f'{cfg["subject_prefix"]}{subject}' if cfg["subject_prefix"] else subject
    msg["From"]    = cfg["from"]
    msg["To"]      = ", ".join(recipients)
    msg.add_alternative(body, subtype="html")

    if cfg["use_ssl"]:
        with smtplib.SMTP_SSL(cfg["host"], cfg["port"], context=ssl.create_default_context()) as srv:
            srv.login(cfg["username"], cfg["password"])
            srv.send_message(msg)
    else:
        with smtplib.SMTP(cfg["host"], cfg["port"]) as srv:
            srv.ehlo()
            srv.starttls(context=ssl.create_default_context())
            srv.login(cfg["username"], cfg["password"])
            srv.send_message(msg)

# ---------------- Core data loaders ----------------
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
        return pd.DataFrame(columns=["item","product_number","multiplier",
                                     "items_per_order","current_qty","sort_order","price"])

    # Normalize column names: lowercase + map known variants
    df.columns = [str(c).strip().lower() for c in df.columns]
    col_aliases = {
        "product_number":            "product_number",
        "product number":            "product_number",
        "multiplier_per_box":        "multiplier",
        "multiplier":                "multiplier",
        "recommended_qty_per_order": "items_per_order",
        "items_per_order":           "items_per_order",
        "current_qty":               "current_qty",
        "sort_order":                "sort_order",
        "price":                     "price",
        "item":                      "item",
    }
    df = df.rename(columns=col_aliases)

    for c in ["item","product_number","multiplier","items_per_order","current_qty","sort_order","price"]:
        if c not in df.columns:
            df[c] = pd.NA

    df["item"]           = df["item"].astype(str).str.strip()
    df["product_number"] = df["product_number"].astype(str).str.strip()
    df["multiplier"]     = pd.to_numeric(df["multiplier"],     errors="coerce").fillna(1).astype(int)
    df["items_per_order"]= pd.to_numeric(df["items_per_order"],errors="coerce").fillna(1).astype(int)
    df["current_qty"]    = pd.to_numeric(df["current_qty"],    errors="coerce").fillna(0).astype(int)
    df["price"]          = pd.to_numeric(df["price"],          errors="coerce").fillna(0.0).astype(float)

    so      = pd.to_numeric(df["sort_order"], errors="coerce")
    filler  = pd.Series(range(len(df)), index=df.index)
    df["sort_order"] = so.fillna(filler).astype(int)

    return df.reset_index(drop=True)

def write_catalog(df: pd.DataFrame):
    df.to_csv(CATALOG_PATH, index=False)
    read_catalog.clear()

# ---------------- Supabase helpers ----------------
def append_log(order_df: pd.DataFrame, orderer: str) -> str:
    now_str = datetime.now(NYC).isoformat(sep=" ", timespec="seconds")
    rows = [
        {
            "item":           r["item"],
            "product_number": str(r["product_number"]),
            "qty":            int(r["qty"]),
            "ordered_at":     now_str,
            "orderer":        orderer,
        }
        for _, r in order_df.iterrows()
    ]
    supabase.table("orders_log").insert(rows).execute()
    return now_str

def read_log() -> pd.DataFrame:
    res = supabase.table("orders_log").select("*").order("ordered_at", desc=True).execute()
    if not getattr(res, "data", None):
        return pd.DataFrame(columns=["item","product_number","qty","ordered_at","orderer"])
    return pd.DataFrame(res.data)

def last_info_map() -> pd.DataFrame:
    logs = read_log()
    if logs.empty:
        return pd.DataFrame(columns=["item","product_number","last_ordered_at","last_qty","last_orderer"])
    logs["ordered_at"] = pd.to_datetime(logs["ordered_at"], errors="coerce")
    tail = logs.sort_values("ordered_at").groupby(["item","product_number"], as_index=False).tail(1)
    return tail.rename(columns={
        "ordered_at": "last_ordered_at",
        "qty":        "last_qty",
        "orderer":    "last_orderer",
    })[["item","product_number","last_ordered_at","last_qty","last_orderer"]]

# ---------------- Emails ----------------
@st.cache_data
def read_emails() -> pd.DataFrame:
    df = safe_read_csv(EMAILS_PATH)
    if df.empty:
        return pd.DataFrame(columns=["name","email"])
    df.columns = [str(c).strip().lower() for c in df.columns]
    email_re = re.compile(r"([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})")
    rows = []
    if "email" in df.columns:
        for _, r in df.iterrows():
            m = email_re.search(str(r.get("email", "")))
            if m:
                rows.append({"name": str(r.get("name", "")), "email": m.group(1)})
    return pd.DataFrame(rows)

def all_recipients(emails_df: pd.DataFrame) -> list[str]:
    cfg = get_smtp_config()
    file_r = emails_df["email"].tolist() if not emails_df.empty else []
    return sorted({e for e in file_r + cfg.get("default_to", []) if e and "@" in e})

# ---------------- Email body builder ----------------
def build_email_body(qty_map: dict, catalog: pd.DataFrame, orderer: str, when_str: str) -> str:
    items = []
    for pid, qty in qty_map.items():
        if qty <= 0:
            continue
        row = catalog.loc[catalog["product_number"].astype(str) == str(pid)]
        if row.empty:
            continue
        item_name = row.iloc[0]["item"]
        price     = float(row.iloc[0].get("price", 0) or 0)
        items.append((pid, qty, item_name, qty * price))

    # First-fit bin packing: try to add each item to the first group with room
    bins: list[list[tuple]] = []
    bin_totals: list[float] = []
    for item in items:
        pid, qty, item_name, total = item
        placed = False
        for i, bin_total in enumerate(bin_totals):
            if bin_total + total <= 4999:
                bins[i].append(item)
                bin_totals[i] += total
                placed = True
                break
        if not placed:
            bins.append([item])
            bin_totals.append(total)

    # Details and group checkboxes share the same group-first order
    details_lines = [
        f"<label><input type='checkbox'/> {item_name} (#{pid}): {qty}</label>"
        for group in bins
        for pid, qty, item_name, _ in group
    ]
    group_lines = [
        f"<label><input type='checkbox'/> {', '.join(f'&quot;{pid}&quot;' for pid, *_ in grp)} = ${sub:,.0f}</label>"
        for grp, sub in zip(bins, bin_totals)
    ]

    return f"""
    <html><body>
    <p><strong>New supply order at {when_str}</strong><br>Ordered by: {orderer}</p>
    <p><strong>Details:</strong><br>{"<br>".join(details_lines)}</p>
    <p><strong>Product groups (≤$4,999 each):</strong><br>{"<br>".join(group_lines)}</p>
    </body></html>
    """

# ---------------- Session state init ----------------
if "orderer" not in st.session_state:
    st.session_state["orderer"] = None
if "qty_map" not in st.session_state:
    st.session_state["qty_map"] = {}

# ---------------- Load data ----------------
people    = read_people()
emails_df = read_emails()
catalog   = read_catalog()
logs      = read_log()

# ---------------- Page header ----------------
st.title("📦 Supply Ordering & Inventory Tracker")
email_ready = "✅" if smtp_ok() else "❌"
st.caption(f"Loaded {len(catalog)} catalog rows • {len(logs)} log rows • Email configured: {email_ready}")

# ---------------- Running order preview ----------------
selected_items = [
    {
        "item":           catalog.loc[catalog["product_number"].astype(str) == str(pid)].iloc[0]["item"],
        "product_number": pid,
        "qty":            qty,
    }
    for pid, qty in st.session_state["qty_map"].items()
    if qty > 0 and not catalog.loc[catalog["product_number"].astype(str) == str(pid)].empty
]

if selected_items:
    st.markdown("### 🛒 Current Order (in progress)")
    sel_df = pd.DataFrame(selected_items)
    st.dataframe(sel_df, hide_index=True, use_container_width=True)
    st.markdown(f"**Product Numbers:** {', '.join(str(i['product_number']) for i in selected_items)}")
    if st.button("🧹 Clear Current Order"):
        st.session_state["qty_map"] = {}
        st.rerun()
else:
    st.caption("🛒 No items currently selected.")

# ================================================================
tabs = st.tabs(["Create Order", "Adjust Inventory", "Catalog", "Order Logs"])

# ----------------------------------------------------------------
# Tab 0 — Create Order
# ----------------------------------------------------------------
with tabs[0]:
    if catalog.empty:
        st.info("No catalog found. Add items to data/catalog.csv.")
    else:
        c1, c2 = st.columns([2, 3])
        with c1:
            current_orderer = st.session_state.get("orderer") or (people[0] if people else "Unknown")
            orderer = st.selectbox(
                "Who is ordering?",
                options=people if people else ["Unknown"],
                index=people.index(current_orderer) if people and current_orderer in people else 0,
            )
            st.session_state["orderer"] = orderer
        with c2:
            search = st.text_input("🔍 Search items")

        # Merge last-order info
        last_map = last_info_map()
        table = catalog.merge(last_map, on=["item","product_number"], how="left")
        for c in ["last_ordered_at","last_qty","last_orderer"]:
            if c not in table.columns:
                table[c] = pd.NA

        table["last_ordered_at"] = pd.to_datetime(table["last_ordered_at"], errors="coerce")
        table = (
            table
            .sort_values(["last_ordered_at","item"], ascending=[False, True], na_position="last")
            .reset_index(drop=True)
        )
        table["product_number"] = table["product_number"].astype(str)

        # Inject current qty values from session state (default 0)
        table["qty"] = table["product_number"].map(
            lambda pid: st.session_state["qty_map"].get(pid, 0)
        ).astype(int)

        # Apply search filter
        if search:
            mask  = table["item"].str.contains(search, case=False, na=False)
            mask |= table["product_number"].str.contains(search, case=False, na=False)
            table = table[mask]

        edited = st.data_editor(
            table[[
                "qty","item","product_number","multiplier","items_per_order",
                "current_qty","price","last_ordered_at","last_qty","last_orderer",
            ]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "qty":            st.column_config.NumberColumn("Qty",          min_value=0, step=1),
                "item":           st.column_config.TextColumn("Item",           disabled=True),
                "product_number": st.column_config.TextColumn("Product #",      disabled=True),
                "multiplier":     st.column_config.NumberColumn("Multiplier",   disabled=True),
                "items_per_order":st.column_config.NumberColumn("Items/Order",  disabled=True),
                "current_qty":    st.column_config.NumberColumn("Current Qty",  disabled=True),
                "price":          st.column_config.NumberColumn("Price",        disabled=True),
                "last_ordered_at":st.column_config.DatetimeColumn("Last Ordered", format="YYYY-MM-DD HH:mm", disabled=True),
                "last_qty":       st.column_config.NumberColumn("Last Qty",     disabled=True),
                "last_orderer":   st.column_config.TextColumn("Last By",        disabled=True),
            },
            key="order_editor",
        )

        # ---- FIX: only rerun when a qty genuinely changes to a non-zero value ----
        rerun_needed = False
        for _, r in edited.iterrows():
            new_qty = int(r["qty"]) if pd.notna(r["qty"]) else 0
            pid     = str(r["product_number"])
            old_qty = st.session_state["qty_map"].get(pid, 0)
            if old_qty != new_qty:
                st.session_state["qty_map"][pid] = new_qty
                if new_qty != 0:          # only force rerun for meaningful changes
                    rerun_needed = True
        if rerun_needed:
            st.rerun()

        # ---- Generate & Log Order ----
        if st.button("🧾 Generate & Log Order"):
            full_order = [
                {
                    "item":           catalog.loc[catalog["product_number"].astype(str) == str(pid)].iloc[0]["item"],
                    "product_number": pid,
                    "qty":            qty,
                }
                for pid, qty in st.session_state["qty_map"].items()
                if qty > 0 and not catalog.loc[catalog["product_number"].astype(str) == str(pid)].empty
            ]

            if not full_order:
                st.warning("No items selected.")
            else:
                full_order_df = pd.DataFrame(full_order)
                when_str      = append_log(full_order_df, orderer)
                st.success(f"Order logged at {when_str}.")

                if smtp_ok():
                    recipients = all_recipients(emails_df)
                    if recipients:
                        body = build_email_body(st.session_state["qty_map"], catalog, orderer, when_str)
                        try:
                            send_email("Supply Order Logged", body, recipients)
                            st.success(f"Email sent to {len(recipients)} recipient(s).")
                        except Exception as e:
                            st.error(f"Email failed: {e}")

                st.session_state["qty_map"] = {}
                st.rerun()

# ----------------------------------------------------------------
# Tab 1 — Adjust Inventory
# ----------------------------------------------------------------
with tabs[1]:
    if catalog.empty:
        st.info("No catalog found.")
    else:
        st.write("Adjust `current_qty`, `sort_order`, or `price`, then save.")
        edited_inv = st.data_editor(
            catalog.copy().reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
            column_config={
                "item":           st.column_config.TextColumn("Item",         disabled=True),
                "product_number": st.column_config.TextColumn("Product #",    disabled=True),
                "multiplier":     st.column_config.NumberColumn("Multiplier", min_value=1,   step=1),
                "items_per_order":st.column_config.NumberColumn("Items/Order",min_value=1,   step=1),
                "current_qty":    st.column_config.NumberColumn("Current Qty",min_value=0,   step=1),
                "sort_order":     st.column_config.NumberColumn("Sort Order", min_value=0,   step=1),
                "price":          st.column_config.NumberColumn("Price ($)",  min_value=0.0, step=0.01),
            },
            key="inventory_editor",
        )
        if st.button("💾 Save inventory changes"):
            write_catalog(edited_inv)
            st.success("Inventory saved.")

# ----------------------------------------------------------------
# Tab 2 — Catalog
# ----------------------------------------------------------------
with tabs[2]:
    st.caption("Catalog source: data/catalog.csv")
    if catalog.empty:
        st.info("No catalog found.")
    else:
        st.dataframe(catalog, use_container_width=True, hide_index=True)

# ----------------------------------------------------------------
# Tab 3 — Order Logs
# ----------------------------------------------------------------
with tabs[3]:
    if logs.empty:
        st.info("No orders logged yet.")
    else:
        st.dataframe(logs, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download full log (CSV)",
            data=logs.to_csv(index=False).encode("utf-8"),
            file_name="order_log.csv",
            mime="text/csv",
        )
