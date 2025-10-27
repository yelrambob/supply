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
now = datetime.now(NYC).strftime("%Y-%m-%d %H:%M:%S")

# ---------------- Paths ----------------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CATALOG_PATH = DATA_DIR / "catalog.csv"
PEOPLE_PATH = DATA_DIR / "people.txt"
EMAILS_PATH = DATA_DIR / "emails.csv"

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
    parts = re.split(r"[;,]\s*", str(txt))
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
            "use_ssl": bool(smtp_config.get("use_ssl", False)),
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
        return pd.DataFrame(
            columns=[
                "item",
                "product_number",
                "multiplier",
                "items_per_order",
                "current_qty",
                "sort_order",
                "price",
            ]
        )

    for c in [
        "item",
        "product_number",
        "multiplier",
        "items_per_order",
        "current_qty",
        "sort_order",
        "price",
    ]:
        if c not in df.columns:
            df[c] = pd.NA

    df["item"] = df["item"].astype(str).str.strip()
    df["product_number"] = df["product_number"].astype(str).str.strip()
    df["multiplier"] = pd.to_numeric(df["multiplier"], errors="coerce").fillna(1).astype(int)
    df["items_per_order"] = pd.to_numeric(df["items_per_order"], errors="coerce").fillna(1).astype(int)
    df["current_qty"] = pd.to_numeric(df["current_qty"], errors="coerce").fillna(0).astype(int)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0).astype(float)

    so = pd.to_numeric(df["sort_order"], errors="coerce")
    filler = pd.Series(range(len(df)), index=df.index)
    df["sort_order"] = so.fillna(filler).astype(int)

    return df.reset_index(drop=True)

def write_catalog(df: pd.DataFrame):
    df.to_csv(CATALOG_PATH, index=False)

# ---------------- Supabase data helpers ----------------
def append_log(order_df: pd.DataFrame, orderer: str) -> str:
    now = datetime.now(NYC).isoformat(sep=" ", timespec="seconds")
    rows = []
    for _, r in order_df.iterrows():
        rows.append(
            {
                "item": r["item"],
                "product_number": str(r["product_number"]),
                "qty": int(r["qty"]),
                "ordered_at": now,
                "orderer": orderer,
            }
        )
    supabase.table("orders_log").insert(rows).execute()
    return now

def read_log() -> pd.DataFrame:
    res = supabase.table("orders_log").select("*").order("ordered_at", desc=True).execute()
    if not getattr(res, "data", None):
        return pd.DataFrame(columns=["item", "product_number", "qty", "ordered_at", "orderer"])
    return pd.DataFrame(res.data)

def last_info_map() -> pd.DataFrame:
    logs = read_log()
    if logs.empty:
        return pd.DataFrame(columns=["item", "product_number", "last_ordered_at", "last_qty", "last_orderer"])
    logs = logs.copy()
    logs["ordered_at"] = pd.to_datetime(logs["ordered_at"], errors="coerce")
    tail = logs.sort_values("ordered_at").groupby(["item", "product_number"], as_index=False).tail(1)
    return tail.rename(
        columns={
            "ordered_at": "last_ordered_at",
            "qty": "last_qty",
            "orderer": "last_orderer",
        }
    )[["item", "product_number", "last_ordered_at", "last_qty", "last_orderer"]]

# ---------------- Emails CSV ----------------
@st.cache_data
def read_emails() -> pd.DataFrame:
    df = safe_read_csv(EMAILS_PATH)
    if df.empty:
        return pd.DataFrame(columns=["name", "email"])
    df.columns = [str(c).strip().lower() for c in df.columns]
    email_re = re.compile(r"([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})")
    out_rows = []
    if "email" in df.columns:
        for _, r in df.iterrows():
            raw = str(r.get("email", ""))
            m = email_re.search(raw)
            if m:
                out_rows.append({"name": str(r.get("name", "")), "email": m.group(1)})
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
    st.session_state["qty_map"] = {}

# ---------------- UI ----------------
st.title("📦 Supply Ordering & Inventory Tracker")

people = read_people()
emails_df = read_emails()
catalog = read_catalog()
logs = read_log()

email_ready = "✅" if smtp_ok() else "❌"
st.caption(f"Loaded {len(catalog)} catalog rows • {len(logs)} log rows • Email configured: {email_ready}")

# --- Running List Preview ---
selected_items = []
for pid, qty in st.session_state["qty_map"].items():
    if qty > 0:
        row = catalog.loc[catalog["product_number"].astype(str) == str(pid)]
        if not row.empty:
            selected_items.append({"item": row.iloc[0]["item"], "product_number": pid, "qty": qty})

if selected_items:
    st.markdown("### 🛒 Current Order (in progress)")
    st.dataframe(pd.DataFrame(selected_items), hide_index=True, use_container_width=True)

    product_numbers = [item["product_number"] for item in selected_items]
    if product_numbers:
        st.markdown(f"**Product Numbers:** {', '.join(product_numbers)}")

    if st.button("🧹 Clear Current Order"):
        st.session_state["qty_map"] = {}
        st.rerun()
else:
    st.caption("🛒 No items currently selected.")

tabs = st.tabs(["Create Order", "Adjust Inventory", "Catalog", "Order Logs"])

# ---------- Create Order ----------
with tabs[0]:
    if catalog.empty:
        st.info("No catalog found.")
    else:
        c1, c2 = st.columns([2, 3])
        with c1:
            current_orderer = st.session_state.get("orderer") or (people[0] if people else "Unknown")
            orderer = st.selectbox(
                "Who is ordering?",
                options=(people if people else ["Unknown"]),
                index=(people.index(current_orderer) if people and current_orderer in people else 0),
            )
            st.session_state["orderer"] = orderer
        with c2:
            search = st.text_input("Search items")

        last_map = last_info_map()
        table = catalog.merge(last_map, on=["item", "product_number"], how="left")

        for c in ["last_ordered_at", "last_qty", "last_orderer"]:
            if c not in table.columns:
                table[c] = pd.NA

        table["last_ordered_at"] = pd.to_datetime(table["last_ordered_at"], errors="coerce")
        table = table.sort_values(["last_ordered_at", "item"], ascending=[False, True], na_position="last").reset_index(drop=True)
        table["product_number"] = table["product_number"].astype(str)
        table["qty"] = table["product_number"].map(st.session_state["qty_map"]).fillna(0).astype(int)

        if search:
            mask = table["item"].str.contains(search, case=False, na=False) | table["product_number"].str.contains(search, case=False, na=False)
            table = table[mask]

        table["_row_key"] = table["product_number"]
        edited = st.data_editor(
            table[
                [
                    "qty",
                    "item",
                    "product_number",
                    "multiplier",
                    "items_per_order",
                    "current_qty",
                    "price",
                    "last_ordered_at",
                    "last_qty",
                    "last_orderer",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "qty": st.column_config.NumberColumn("Qty", min_value=0, step=1),
                "item": st.column_config.TextColumn("Item", disabled=True),
                "product_number": st.column_config.TextColumn("Product #", disabled=True),
                "multiplier": st.column_config.NumberColumn("Multiplier", disabled=True),
                "items_per_order": st.column_config.NumberColumn("Items/Order", disabled=True),
                "current_qty": st.column_config.NumberColumn("Current Qty", disabled=True),
                "price": st.column_config.NumberColumn("Price", disabled=True),
                "last_ordered_at": st.column_config.DatetimeColumn("Last ordered", format="YYYY-MM-DD HH:mm", disabled=True),
                "last_qty": st.column_config.NumberColumn("Last qty", disabled=True),
                "last_orderer": st.column_config.TextColumn("Last by", disabled=True),
            },
            key="order_editor",
        )

        # --- Live update logic ---
        rerun_needed = False
        for _, r in edited.iterrows():
            new_qty = int(r["qty"])
            pid = str(r["product_number"])
            if st.session_state["qty_map"].get(pid) != new_qty:
                st.session_state["qty_map"][pid] = new_qty
                rerun_needed = True
        if rerun_needed:
            st.rerun()

        # --- Generate & Log Order ---
        if st.button("🧾 Generate & Log Order"):
            full_order = []
            for pid, qty in st.session_state["qty_map"].items():
                if qty > 0:
                    row = catalog.loc[catalog["product_number"].astype(str) == str(pid)]
                    if not row.empty:
                        full_order.append({"item": row.iloc[0]["item"], "product_number": pid, "qty": qty})
            full_order_df = pd.DataFrame(full_order)

            if not full_order_df.empty:
                when_str = append_log(full_order_df, orderer)
                if smtp_ok():
                    recipients = all_recipients(emails_df)
                    if recipients:
                        # --- Email body with $4,999 grouping ---
                        product_groups = []
                        current_group = []
                        running_total = 0.0
                        details_lines = []

                        for pid, qty in st.session_state["qty_map"].items():
                            if qty > 0:
                                row = catalog.loc[catalog["product_number"].astype(str) == str(pid)]
                                if not row.empty:
                                    item_name = row.iloc[0]["item"]
                                    price = float(row.iloc[0].get("price", 0) or 0)
                                    total = qty * price

                                    # Start new group if adding this exceeds 4999
                                    if running_total + total > 4999 and current_group:
                                        product_groups.append((current_group.copy(), running_total))
                                        current_group = []
                                        running_total = 0.0

                                    running_total += total
                                    current_group.append(pid)
                                    details_lines.append(f"- {item_name} (#{pid}): {qty}")

                        if current_group:
                            product_groups.append((current_group, running_total))

                        group_lines = []
                        for group, subtotal in reversed(product_groups):
                            product_str = ", ".join(str(p) for p in group)
                            group_lines.append(f"{product_str} = ${subtotal:,.0f}")

                        # HTML email body with checkbox
                        html_body = f"""<html>
<body>
<p>New supply order at {when_str}</p>
<p>Ordered by: {orderer}</p>
<br>
<p><strong>Details:</strong></p>
{"<br>".join(details_lines)}
<br>
<p><strong>Product:</strong></p>
{"<br>".join(group_lines)}
<br>
<hr>
<label>
    <input type="checkbox" checked onclick="this.checked=true">
    Order confirmed and processed
</label>
</body>
</html>"""

                        # Plain text version for email clients that prefer it
                        plain_body = "\n".join(
                            [
                                f"New supply order at {when_str}",
                                f"Ordered by: {orderer}",
                                "",
                                "Details:",
                                *details_lines,
                                "",
                                "Product:",
                                *group_lines,
                                "",
                                "---",
                                "[Order confirmed and processed]"
                            ]
                        )

                        try:
                            # Create the email message with both HTML and plain text
                            cfg = get_smtp_config()
                            msg = EmailMessage()
                            msg["Subject"] = f'{cfg["subject_prefix"]}Supply Order Logged' if cfg["subject_prefix"] else "Supply Order Logged"
                            msg["From"] = cfg["from"]
                            msg["To"] = ", ".join(recipients)
                            
                            # Set both HTML and plain text versions
                            msg.set_content(plain_body) # Fallback for email clients that don't support HTML
                            msg.add_alternative(html_body, subtype='html') # HTML version with checkbox
                            
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
                            
                            st.success(f"Emailed {len(recipients)} recipient(s).")
                        except Exception as e:
                            st.error(f"Email failed: {e}")

                st.session_state["qty_map"] = {}
                st.rerun()

# ---------- Adjust Inventory ----------
with tabs[1]:
    if catalog.empty:
        st.info("No catalog found.")
    else:
        st.write("Adjust `current_qty`, `sort_order`, or `price`, then save.")
        editable = catalog.copy().reset_index(drop=True)
        edited = st.data_editor(
            editable,
            use_container_width=True,
            hide_index=True,
            column_config={
                "item": st.column_config.TextColumn("Item", disabled=True),
                "product_number": st.column_config.TextColumn("Product #", disabled=True),
                "multiplier": st.column_config.NumberColumn("Multiplier", min_value=1, step=1),
                "items_per_order": st.column_config.NumberColumn("Items/Order", min_value=1, step=1),
                "current_qty": st.column_config.NumberColumn("Current Qty", min_value=0, step=1),
                "sort_order": st.column_config.NumberColumn("Sort order", min_value=0, step=1),
                "price": st.column_config.NumberColumn("Price ($)", min_value=0.0, step=0.01),
            },
            key="inventory_editor",
        )
        if st.button("💾 Save inventory changes"):
            write_catalog(edited)
            st.success("Inventory saved.")

# ---------- Catalog ----------
with tabs[2]:
    st.caption("Catalog source: data/catalog.csv")
    if catalog.empty:
        st.info("No catalog found.")
    else:
        st.dataframe(catalog, use_container_width=True, hide_index=True)

# ---------- Order Logs ----------
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
