import streamlit as st
import pandas as pd
from pandas.errors import EmptyDataError
from datetime import datetime
from pathlib import Path
import io
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

# ---------------- Helpers: files ----------------
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
def smtp_ok() -> bool:
    try:
        s = st.secrets["smtp"]
        needed = ["server", "port", "username", "password", "from"]
        return all(s.get(k) for k in needed)
    except Exception:
        return False

def send_email(subject: str, body: str, to_emails: list[str]):
    s = st.secrets["smtp"]
    ctx = ssl.create_default_context()
    msg = EmailMessage()
    prefix = s.get("subject_prefix", "")
    msg["Subject"] = f"{prefix}{subject}" if prefix else subject
    msg["From"] = s["from"]
    if s.get("reply_to"):
        msg["Reply-To"] = s["reply_to"]
    msg["To"] = ", ".join(to_emails)
    msg.set_content(body)

    with smtplib.SMTP_SSL(s["server"], int(s["port"]), context=ctx) as server:
        server.login(s["username"], s["password"])
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
def read_emails() -> pd.DataFrame:
    df = safe_read_csv(EMAILS_PATH)
    if df.empty:
        return pd.DataFrame(columns=["name", "email"])
    # normalize cols
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "email" not in df.columns:
        return pd.DataFrame(columns=["name", "email"])
    if "name" not in df.columns:
        df["name"] = ""
    df["email"] = df["email"].astype(str).str.strip()
    df = df[df["email"].str.contains("@", na=False)].drop_duplicates(subset=["email"])
    return df[["name", "email"]]

@st.cache_data
def read_catalog() -> pd.DataFrame:
    df = safe_read_csv(CATALOG_PATH)
    if df.empty:
        return pd.DataFrame(columns=["item", "product_number", "current_qty", "sort_order"])
    # normalize expected cols
    for c in ["item", "product_number", "current_qty", "sort_order"]:
        if c not in df.columns:
            df[c] = pd.NA
    # types
    df["item"] = df["item"].astype(str).str.strip()
    df["product_number"] = df["product_number"].astype(str).str.strip()
    df["current_qty"] = pd.to_numeric(df["current_qty"], errors="coerce").fillna(0).astype(int)

    so = pd.to_numeric(df["sort_order"], errors="coerce")
    filler = pd.Series(range(len(df)), index=df.index)
    df["sort_order"] = so.fillna(filler).astype(int)

    # drop unusable rows
    df = df[(df["item"] != "") & (df["product_number"] != "")]
    return df[["item", "product_number", "current_qty", "sort_order"]].reset_index(drop=True)

def write_catalog(df: pd.DataFrame):
    df = df.copy()
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
    expected = ORDER_LOG_COLUMNS
    df = df[expected]

    prev = safe_read_csv(LOG_PATH)
    if not prev.empty:
        for c in expected:
            if c not in prev.columns:
                prev[c] = pd.NA
        combined = pd.concat([prev[expected], df], ignore_index=True)
    else:
        combined = df

    combined.to_csv(LOG_PATH, index=False)
    return now

def read_last() -> pd.DataFrame:
    df = safe_read_csv(LAST_PATH)
    if df.empty:
        return pd.DataFrame(columns=LAST_ORDER_COLUMNS)
    for c in LAST_ORDER_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype(int)
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
    logs = logs.sort_values("ordered_at")
    tail = logs.groupby(["item","product_number"], as_index=False).tail(1)
    tail = tail.rename(columns={"ordered_at":"last_ordered_at","qty":"last_qty","orderer":"last_orderer"})
    return tail[["item","product_number","last_ordered_at","last_qty","last_orderer"]]

# ---------------- UI ----------------
st.title("📦 Supply Ordering & Inventory Tracker")

people = read_people()
emails_df = read_emails()
catalog = read_catalog()
logs = read_log()
last_order_df = read_last()

# Quick info line
st.caption(f"Loaded {len(catalog)} catalog rows • {len(logs)} log rows • Email configured: {'✅' if smtp_ok() else '❌'}")

tabs = st.tabs(["Create Order", "Adjust Inventory", "Catalog", "Order Logs"])

# ---------- Create Order ----------
with tabs[0]:
    # Collapsible "last generated" block (so it doesn't clutter the top)
    with st.expander("📋 Last generated order (copy/download)", expanded=False):
        if last_order_df.empty:
            st.info("No previous order.")
        else:
            lines = [f"{r['item']} — {r['product_number']} — Qty {r['qty']}" for _, r in last_order_df.iterrows()]
            meta = f"Generated at {last_order_df['generated_at'].iloc[0]} by {last_order_df['orderer'].iloc[0]}"
            st.text_area("Copy/paste", value="\n".join(lines), height=160, key="order_copy_area")
            st.caption(meta)
            st.download_button(
                "⬇️ Download CSV",
                data=last_order_df[["item","product_number","qty"]].to_csv(index=False).encode("utf-8"),
                file_name=f"order_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="order_download_btn",
            )

    if catalog.empty:
        st.info("No catalog found. Put your list in data/catalog.csv (columns: item, product_number[, current_qty, sort_order]).")
    else:
        # Controls row
        c1, c2, c3 = st.columns([2, 2, 2])
        with c1:
            orderer = st.selectbox(
                "Who is ordering?",
                options=(people if people else ["(add names in data/people.txt)"]),
                index=0,
                key="order_orderer",
            )
        with c2:
            cc_orderer = st.checkbox("CC the orderer (if email match)", value=True, key="order_cc_orderer")
        with c3:
            search = st.text_input("Search items", key="order_search")

        # Merge "last ordered" info for display/sort
        last_map = last_info_map()
        table = catalog.merge(last_map, on=["item","product_number"], how="left")
        table["last_ordered_at"] = pd.to_datetime(table.get("last_ordered_at"), errors="coerce")

        # Sorting selector
        sort_choice = st.selectbox(
            "Sort by",
            options=["Last ordered (newest first)", "Original order", "Name A→Z", "Product # asc"],
            index=0,
            key="order_sort",
        )
        if sort_choice == "Original order":
            table = table.sort_values(["sort_order", "item"], kind="stable")
        elif sort_choice == "Name A→Z":
            table = table.sort_values(["item"], kind="stable")
        elif sort_choice == "Product # asc":
            table = table.sort_values(["product_number","item"], kind="stable")
        else:  # newest first
            key = table["last_ordered_at"].fillna(pd.Timestamp("1900-01-01"))
            table = table.iloc[key.sort_values(ascending=False).index]

        # Search filter
        if search:
            table = table[table["item"].str.contains(search, case=False, na=False)]

        # Display spreadsheet-like table with a Qty column
        table = table.copy()
        table["qty"] = 0
        show_cols = ["qty", "item", "product_number", "last_ordered_at", "last_qty", "last_orderer"]
        edited = st.data_editor(
            table[show_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "qty": st.column_config.NumberColumn("Qty", min_value=0, step=1),
                "item": st.column_config.TextColumn("Item", disabled=True),
                "product_number": st.column_config.TextColumn("Product #", disabled=True),
                "last_ordered_at": st.column_config.DatetimeColumn("Last ordered", format="YYYY-MM-DD HH:mm", disabled=True),
                "last_qty": st.column_config.NumberColumn("Last qty", disabled=True),
                "last_orderer": st.column_config.TextColumn("Last by", disabled=True),
            },
            key="order_editor",
        )

        # Buttons right under the table for quick access
        b1, b2 = st.columns(2)

        def _selected_from_editor(df: pd.DataFrame) -> pd.DataFrame:
            chosen = df[df["qty"] > 0].copy()
            if chosen.empty:
                st.error("Please set Qty > 0 for at least one item.")
                return pd.DataFrame()
            if not people or orderer == "(add names in data/people.txt)":
                st.error("Please add/select an orderer in data/people.txt.")
                return pd.DataFrame()
            return chosen[["item","product_number","qty"]].copy()

        def _email_recipients(orderer_name: str, cc_orderer_flag: bool) -> list[str]:
            recips = read_emails()
            to = set(recips["email"].tolist())
            if cc_orderer_flag and orderer_name and not recips.empty:
                match = recips[recips["name"].str.strip().str.lower() == orderer_name.strip().lower()]
                to.update(match["email"].tolist())
            return sorted([e for e in to if e])

        def _log_and_email(order_df: pd.DataFrame, do_decrement: bool):
            # Save screen copy
            write_last(order_df, orderer)
            # Log rows
            when_str = append_log(order_df, orderer)

            # Optionally decrement current_qty
            if do_decrement:
                cat2 = catalog.copy()
                for _, r in order_df.iterrows():
                    mask = (cat2["item"] == r["item"]) & (cat2["product_number"].astype(str) == str(r["product_number"]))
                    cat2.loc[mask, "current_qty"] = (
                        pd.to_numeric(cat2.loc[mask, "current_qty"], errors="coerce").fillna(0).astype(int) - int(r["qty"])
                    ).clip(lower=0)
                write_catalog(cat2)

            # Email everyone in emails.csv (and cc orderer if chosen)
            if smtp_ok():
                recipients = _email_recipients(orderer, cc_orderer)
                if recipients:
                    lines = [f"- {r['item']} (#{r['product_number']}): {r['qty']}" for _, r in order_df.iterrows()]
                    body = "\n".join([
                        f"New supply order logged at {when_str}",
                        f"Ordered by: {orderer or 'Unknown'}",
                        "",
                        "Items:",
                        *lines
                    ])
                    try:
                        send_email("Supply Order Logged", body, recipients)
                        st.success(f"Emailed {len(recipients)} recipient(s).")
                    except Exception as e:
                        st.error(f"Email failed: {e}")
                else:
                    st.info("No valid recipients found in data/emails.csv.")
            else:
                st.info("Email disabled — configure .streamlit/secrets.toml [smtp].")

            st.rerun()

        with b1:
            if st.button("🧾 Generate & Log Order", use_container_width=True, key="btn_log"):
                selected = _selected_from_editor(edited)
                if not selected.empty:
                    _log_and_email(selected, do_decrement=False)

        with b2:
            if st.button("🧾 Generate, Log, & Decrement", use_container_width=True, key="btn_log_dec"):
                selected = _selected_from_editor(edited)
                if not selected.empty:
                    _log_and_email(selected, do_decrement=True)

# ---------- Adjust Inventory ----------
with tabs[1]:
    if catalog.empty:
        st.info("No catalog found.")
    else:
        st.write("Adjust `current_qty` or `sort_order`, then save.")
        editable = catalog.copy()
        edited = st.data_editor(
            editable,
            use_container_width=True,
            hide_index=True,
            column_config={
                "item": st.column_config.TextColumn("Item", disabled=True),
                "product_number": st.column_config.TextColumn("Product #", disabled=True),
                "current_qty": st.column_config.NumberColumn("Current Qty", min_value=0, step=1),
                "sort_order": st.column_config.NumberColumn("Sort order", min_value=0, step=1),
            },
            key="inventory_editor",
        )
        if st.button("💾 Save inventory changes", key="inventory_save"):
            write_catalog(edited)
            st.success("Inventory saved.")

# ---------- Catalog (read-only + quick add/remove) ----------
with tabs[2]:
    st.caption("Catalog source: data/catalog.csv")
    if catalog.empty:
        st.info("No catalog found.")
    else:
        st.dataframe(catalog, use_container_width=True, hide_index=True)

    st.markdown("**Quick add**")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        new_item = st.text_input("Item name", key="cat_add_item")
    with c2:
        new_pn = st.text_input("Product #", key="cat_add_pn")
    with c3:
        new_qty = st.number_input("Current qty", min_value=0, value=0, step=1, key="cat_add_qty")

    if st.button("➕ Add to catalog", key="cat_add_btn"):
        if new_item.strip() and new_pn.strip():
            next_order = (catalog["sort_order"].max() + 1) if not catalog.empty else 0
            new_row = pd.DataFrame([{
                "item": new_item.strip(),
                "product_number": str(new_pn).strip(),
                "current_qty": int(new_qty),
                "sort_order": int(next_order),
            }])
            updated = pd.concat([catalog, new_row], ignore_index=True).drop_duplicates(
                subset=["item","product_number"], keep="last"
            )
            write_catalog(updated)
            st.success(f"Added: {new_item.strip()}")
            st.rerun()
        else:
            st.error("Item and Product # are required.")

    st.markdown("---")
    if not catalog.empty:
        to_remove = st.multiselect("Remove item(s)", catalog["item"].tolist(), key="cat_remove_sel")
        if st.button("🗑️ Remove selected", key="cat_remove_btn"):
            updated = catalog[~catalog["item"].isin(to_remove)]
            write_catalog(updated)
            st.success(f"Removed {len(to_remove)} item(s).")
            st.rerun()

# ---------- Order Logs ----------
with tabs[3]:
    logs = read_log()
    if logs.empty:
        st.info("No orders logged yet.")
    else:
        st.dataframe(logs.sort_values("ordered_at", ascending=False), use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download full log (CSV)",
            data=logs.to_csv(index=False).encode("utf-8"),
            file_name="order_log.csv",
            mime="text/csv",
            key="log_dl",
        )

        # Quick clearing tools
        st.markdown("### Clear 'Last ordered' history")
        pairs = logs[["item","product_number"]].drop_duplicates().sort_values(["item","product_number"])
        pairs["label"] = pairs.apply(lambda r: f"{r['item']} — {r['product_number']}", axis=1)
        to_clear = st.multiselect("Select items to clear from history", pairs["label"].tolist(), key="log_clear_sel")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🧹 Clear selected", key="log_clear_selected"):
                if to_clear:
                    keep = ~logs.apply(lambda r: f"{r['item']} — {r['product_number']}" in set(to_clear), axis=1)
                    logs[keep].to_csv(LOG_PATH, index=False)
                    st.success(f"Cleared {len(logs) - keep.sum()} rows.")
                    st.rerun()
                else:
                    st.info("No items selected.")
        with c2:
            if st.button("🗑️ Clear ALL history", key="log_clear_all"):
                ensure_headers(LOG_PATH, ORDER_LOG_COLUMNS)
                st.success("Cleared entire order history.")
                st.rerun()
