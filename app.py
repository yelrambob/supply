import streamlit as st
import pandas as pd
from pandas.errors import EmptyDataError
from datetime import datetime
from pathlib import Path
import re
import smtplib, ssl
from email.message import EmailMessage

st.set_page_config(page_title="Supply Ordering", page_icon="��", layout="wide")

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

# ---------------- Robust file helpers ----------------
def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV safely: handle missing/empty files and encoding differences."""
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
    """Create an empty CSV with headers if file missing/empty."""
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
    """Get SMTP config from Streamlit secrets"""
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
    """Send email; union caller recipients with [smtp].to fallback and dedupe."""
    cfg = get_smtp_config()
    recipients = (to_emails or []) + cfg.get("default_to", [])
    # Deduplicate + simple sanity check
    recipients = sorted({e for e in recipients if e and "@" in e})
    if not recipients:
        raise RuntimeError("No recipients (emails.csv empty and [smtp].to not set).")

    msg = EmailMessage()
    msg["Subject"] = f'{cfg["subject_prefix"]}{subject}' if cfg["subject_prefix"] else subject
    msg["From"] = cfg["from"]
    msg["To"] = ", ".join(recipients)
    msg.set_content(body)

    if cfg["use_ssl"]:
        # Port 465 SSL
        with smtplib.SMTP_SSL(cfg["host"], cfg["port"], context=ssl.create_default_context()) as server:
            server.login(cfg["username"], cfg["password"])
            server.send_message(msg)
    else:
        # Port 587 STARTTLS
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

    # Ensure required columns exist
    for c in ["item", "product_number", "current_qty", "sort_order"]:
        if c not in df.columns:
            df[c] = pd.NA

    df["item"] = df["item"].astype(str).str.strip()
    df["product_number"] = df["product_number"].astype(str).str.strip()  # keep as string key
    df["current_qty"] = pd.to_numeric(df["current_qty"], errors="coerce").fillna(0).astype(int)

    # Default sort order if missing
    so = pd.to_numeric(df["sort_order"], errors="coerce")
    filler = pd.Series(range(len(df)), index=df.index)
    df["sort_order"] = so.fillna(filler).astype(int)

    # Drop rows missing keys
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
    df["item"] = df["item"].astype(str)
    df["product_number"] = df["product_number"].astype(str)
    return df[ORDER_LOG_COLUMNS].sort_values("ordered_at", ascending=False)

def append_log(order_df: pd.DataFrame, orderer: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = order_df.copy()
    df["ordered_at"] = now
    df["orderer"] = orderer
    df["item"] = df["item"].astype(str)
    df["product_number"] = df["product_number"].astype(str)
    expected = ORDER_LOG_COLUMNS

    prev = safe_read_csv(LOG_PATH)
    if not prev.empty:
        for c in expected:
            if c not in prev.columns:
                prev[c] = pd.NA
        prev["item"] = prev["item"].astype(str)
        prev["product_number"] = prev["product_number"].astype(str)
        combined = pd.concat([prev[expected], df[expected]], ignore_index=True)
    else:
        combined = df[expected]

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
    df["item"] = df["item"].astype(str)
    df["product_number"] = df["product_number"].astype(str)
    return df[LAST_ORDER_COLUMNS]

def write_last(df: pd.DataFrame, orderer: str):
    out = df.copy()
    out["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out["orderer"] = orderer
    out["item"] = out["item"].astype(str)
    out["product_number"] = out["product_number"].astype(str)
    out = out[LAST_ORDER_COLUMNS]
    out.to_csv(LAST_PATH, index=False)

def last_info_map() -> pd.DataFrame:
    logs = read_log()
    if logs.empty:
        return pd.DataFrame(columns=["item","product_number","last_ordered_at","last_qty","last_orderer"])
    logs = logs.copy()
    logs["item"] = logs["item"].astype(str)
    logs["product_number"] = logs["product_number"].astype(str)
    logs = logs.sort_values("ordered_at")
    tail = logs.groupby(["item","product_number"], as_index=False).tail(1)
    tail = tail.rename(columns={"ordered_at":"last_ordered_at","qty":"last_qty","orderer":"last_orderer"})
    tail["item"] = tail["item"].astype(str)
    tail["product_number"] = tail["product_number"].astype(str)
    return tail[["item","product_number","last_ordered_at","last_qty","last_orderer"]]

# ---------------- Emails CSV ----------------
@st.cache_data
def read_emails() -> pd.DataFrame:
    """
    Return DataFrame with columns ['name','email'] where 'name' may be empty.
    Accepts:
      - name,email
      - email
      - single column with 'Name <email>' or 'Name, email'
      - multiple comma/semicolon separated entries in one cell
    """
    df = safe_read_csv(EMAILS_PATH)
    if df.empty:
        return pd.DataFrame(columns=["name", "email"])

    df.columns = [str(c).strip().lower() for c in df.columns]
    email_re = re.compile(r'([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})')

    def extract_email(s: str) -> str:
        s = str(s or "")
        m = email_re.search(s)
        return m.group(1) if m else ""

    out_rows = []
    if "email" in df.columns:
        name_col = "name" if "name" in df.columns else None
        for _, r in df.iterrows():
            raw_email = r.get("email", "")
            email = extract_email(raw_email)
            if not email:
                continue
            name = str(r.get(name_col, "")).strip() if name_col else ""
            if not name:
                raw = str(raw_email)
                if "<" in raw and ">" in raw:
                    name = raw.split("<", 1)[0].strip().strip(",")
            out_rows.append({"name": name, "email": email})
    else:
        first_col = df.columns[0]
        for _, r in df.iterrows():
            raw = str(r.get(first_col, ""))
            parts = [p.strip() for p in re.split(r'[;,]\s*', raw) if p.strip()]
            if not parts:
                parts = [raw]
            for p in parts:
                email = extract_email(p)
                if email:
                    name = ""
                    if "<" in p and ">" in p:
                        name = p.split("<", 1)[0].strip().strip(",")
                    out_rows.append({"name": name, "email": email})

    out = pd.DataFrame(out_rows)
    if out.empty:
        return pd.DataFrame(columns=["name", "email"])

    out["email"] = out["email"].astype(str).str.strip()
    out["name"] = out["name"].astype(str).str.strip()
    out = out[out["email"].str.contains("@", na=False)].drop_duplicates(subset=["email"]).reset_index(drop=True)
    return out[["name", "email"]]

def all_recipients(emails_df: pd.DataFrame) -> list[str]:
    """Union of data/emails.csv and [smtp].to; deduped."""
    cfg = get_smtp_config()
    file_recipients = []
    if not emails_df.empty:
        file_recipients = emails_df["email"].astype(str).str.strip().tolist()
    recipients = {e for e in file_recipients if e} | {e for e in cfg.get("default_to", []) if e}
    recipients = {e for e in recipients if "@" in e}
    return sorted(recipients)

# ---------------- Session state ----------------
if "orderer" not in st.session_state:
    st.session_state["orderer"] = None

# ---------------- UI ----------------
st.title("📦 Supply Ordering & Inventory Tracker")

people = read_people()
emails_df = read_emails()
catalog = read_catalog()
logs = read_log()
last_order_df = read_last()

email_ready = "✅" if smtp_ok() else "❌"
st.caption(
    f"Loaded {len(catalog)} catalog rows • {len(logs)} log rows • "
    f"Email configured: {email_ready} • Recipients discovered: {len(all_recipients(emails_df))}"
)

tabs = st.tabs(["Create Order", "Adjust Inventory", "Catalog", "Order Logs", "Tools"])

# ---------- Create Order ----------
with tabs[0]:
    # Last generated (collapsible)
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
        c1, c2, c3 = st.columns([2, 2, 3])
        with c1:
            current_orderer = st.session_state.get("orderer") or (people[0] if people else "(add names in data/people.txt)")
            orderer = st.selectbox(
                "Who is ordering?",
                options=(people if people else ["(add names in data/people.txt)"]),
                index=(people.index(current_orderer) if people and current_orderer in people else 0),
                key="order_orderer_select",
            )
            st.session_state["orderer"] = orderer
        with c2:
            search = st.text_input("Search items", key="order_search")
        with c3:
            if st.button("🧼 Clear quantities", use_container_width=True, key="btn_clear_qty"):
                st.success("Cleared all quantities.")
                st.rerun()

        # Merge last-ordered info (force string keys both sides)
        last_map = last_info_map()

        cat2 = catalog.copy()
        cat2["item"] = cat2["item"].astype(str)
        cat2["product_number"] = cat2["product_number"].astype(str)

        lm2 = last_map.copy()
        if not lm2.empty:
            lm2["item"] = lm2["item"].astype(str)
            lm2["product_number"] = lm2["product_number"].astype(str)

        table = cat2.merge(lm2, on=["item","product_number"], how="left")
        table["last_ordered_at"] = pd.to_datetime(table.get("last_ordered_at"), errors="coerce")

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
        else:
            key = table["last_ordered_at"].fillna(pd.Timestamp("1900-01-01"))
            table = table.iloc[key.sort_values(ascending=False).index]

        # Search filter
        if search:
            table = table[table["item"].str.contains(search, case=False, na=False)]

        # Prepare UI columns - KEY DIFFERENCE: Direct DataFrame manipulation like old code
        table["last_qty"] = pd.to_numeric(table.get("last_qty"), errors="coerce")
        table["qty"] = 0

        # Prefill qty from last generated order (optional convenience)
        if not last_order_df.empty:
            prev_map = {(r["item"], str(r["product_number"])): int(r["qty"]) for _, r in last_order_df.iterrows()}
            for i, r in table.iterrows():
                key = (r["item"], str(r["product_number"]))
                if key in prev_map:
                    table.at[i, "qty"] = int(prev_map[key])

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

        # Buttons under the table
        b1, b2 = st.columns(2)

        def _selected_from_state() -> pd.DataFrame:
            chosen = edited[edited["qty"] > 0].copy()
            if chosen.empty:
                st.error("Please set Qty > 0 for at least one item.")
                return pd.DataFrame()
            if not people or st.session_state.get("orderer") == "(add names in data/people.txt)":
                st.error("Please add/select an orderer in data/people.txt.")
                return pd.DataFrame()
            return chosen[["item", "product_number", "qty"]].copy()

        def _log_and_email(order_df: pd.DataFrame, do_decrement: bool):
            orderer_local = st.session_state.get("orderer") or ""

            # Save last generated (persists after reboot)
            write_last(order_df, orderer_local)

            # Append to durable log CSV (persists after reboot)
            when_str = append_log(order_df, orderer_local)

            # Decrement inventory if chosen
            if do_decrement:
                cat2_local = catalog.copy()
                for _, r in order_df.iterrows():
                    mask = (cat2_local["item"] == r["item"]) & (cat2_local["product_number"].astype(str) == str(r["product_number"]))
                    cat2_local.loc[mask, "current_qty"] = (
                        pd.to_numeric(cat2_local.loc[mask, "current_qty"], errors="coerce").fillna(0).astype(int) - int(r["qty"])
                    ).clip(lower=0)
                write_catalog(cat2_local)

            # Email everyone from emails.csv and/or secrets.to
            if smtp_ok():
                recipients = all_recipients(emails_df)
                if recipients:
                    lines = [f"- {r['item']} (#{r['product_number']}): {r['qty']}" for _, r in order_df.iterrows()]
                    body = "\n".join([
                        f"New supply order logged at {when_str}",
                        f"Ordered by: {orderer_local or 'Unknown'}",
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
                    st.info("No recipients found in emails.csv nor [smtp].to.")
            else:
                st.info("Email disabled — fix .streamlit/secrets.toml [smtp].")

            # Refresh so "Last ordered" updates and top copy block refreshes
            st.rerun()

        with b1:
            if st.button("🧾 Generate & Log Order", use_container_width=True, key="btn_log"):
                selected = _selected_from_state()
                if not selected.empty:
                    _log_and_email(selected, do_decrement=False)

        with b2:
            if st.button("�� Generate, Log, & Decrement", use_container_width=True, key="btn_log_dec"):
                selected = _selected_from_state()
                if not selected.empty:
                    _log_and_email(selected, do_decrement=True)

# ---------- Adjust Inventory ----------
with tabs[1]:
    if catalog.empty:
        st.info("No catalog found.")
    else:
        st.write("Adjust `current_qty` or `sort_order`, then save.")
        editable = catalog.copy().reset_index(drop=True)
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

# ---------- Catalog ----------
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

# ---------- Tools (Danger Zone) ----------
with tabs[4]:
    st.subheader("⚠️ Tools (Danger Zone)")
    st.write("Choose what to clear. This **does not** touch your catalog, people, or emails.")

    col1, col2, col3 = st.columns(3)
    with col1:
        opt_qty = st.checkbox("Clear on-screen quantities", value=True, key="clear_qty_map")
    with col2:
        opt_last = st.checkbox("Clear last generated order", value=True, key="clear_last")
    with col3:
        opt_logs = st.checkbox("Clear order logs", value=True, key="clear_logs")

    confirm = st.checkbox("I understand this action cannot be undone.", key="clear_confirm")

    c1, c2 = st.columns([2, 1])
    with c1:
        if st.button("🧨 Clear ALL selected info", type="primary", disabled=not confirm, key="btn_clear_all"):
            try:
                if opt_last:
                    pd.DataFrame(columns=LAST_ORDER_COLUMNS).to_csv(LAST_PATH, index=False)
                if opt_logs:
                    pd.DataFrame(columns=ORDER_LOG_COLUMNS).to_csv(LOG_PATH, index=False)
                st.success("Selected data cleared.")
                st.rerun()
            except Exception as e:
                st.error(f"Clear failed: {e}")

    with c2:
        st.markdown("### ")
        # Test email sends to union of emails.csv AND [smtp].to and shows exact failures.
        if st.button("✉️ Send test email to recipients", key="btn_test_email"):
            try:
                if not smtp_ok():
                    st.error("SMTP not configured correctly in .streamlit/secrets.toml [smtp].")
                else:
                    recipients = all_recipients(emails_df)
                    if not recipients:
                        st.error("No recipients found in data/emails.csv nor [smtp].to.")
                    else:
                        send_email(
                            subject="Test — Supply App",
                            body="This is a test email from your Streamlit supply app.",
                            to_emails=recipients,
                        )
                        st.success(f"Test email sent to {len(recipients)} recipient(s).")
            except Exception as e:
                st.error(f"Test email failed: {e}")

    # SMTP diagnostics: see parsed config (without password) + connectivity NOOP
    with st.expander("🔎 SMTP diagnostics", expanded=False):
        cfg = get_smtp_config()
        safe_cfg = {k: v for k, v in cfg.items() if k != "password"}
        st.write("Parsed SMTP config (password hidden):")
        st.json(safe_cfg)

        st.write("Planned recipients (union of emails.csv and [smtp].to):")
        st.write(all_recipients(emails_df))

        if st.button("Run connection test", key="btn_smtp_diag"):
            try:
                if cfg["use_ssl"]:
                    with smtplib.SMTP_SSL(cfg["host"], cfg["port"], context=ssl.create_default_context()) as server:
                        code, _ = server.noop()
                    st.success(f"SSL connection OK (NOOP {code})")
                else:
                    with smtplib.SMTP(cfg["host"], cfg["port"]) as server:
                        server.ehlo()
                        server.starttls(context=ssl.create_default_context())
                        code, _ = server.noop()
                    st.success(f"STARTTLS connection OK (NOOP {code})")
            except Exception as e:
                st.error(f"SMTP connection failed: {e}")
