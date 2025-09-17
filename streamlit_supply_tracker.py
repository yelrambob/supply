import streamlit as st
import pandas as pd
import io
from datetime import datetime
from pathlib import Path
from pandas.errors import EmptyDataError

st.set_page_config(page_title="Supply Tracker", page_icon="📦", layout="wide")

# ---------- Paths ----------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CATALOG_PATH = DATA_DIR / "catalog.csv"
LOG_PATH = DATA_DIR / "order_log.csv"
PEOPLE_PATH = DATA_DIR / "people.txt"
LAST_ORDER_PATH = DATA_DIR / "last_order.csv"

LAST_ORDER_COLUMNS = ["item", "product_number", "qty", "generated_at", "orderer"]
ORDER_LOG_COLUMNS = ["item", "product_number", "qty", "ordered_at", "orderer"]

# ---------- Robust CSV helpers ----------
def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except (FileNotFoundError, EmptyDataError):
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Couldn't read {path}: {e}")
        return pd.DataFrame()

def safe_ensure_file_with_header(path: Path, columns):
    """Create CSV with headers if missing or empty."""
    try:
        if (not path.exists()) or path.stat().st_size == 0:
            pd.DataFrame(columns=columns).to_csv(path, index=False)
    except Exception as e:
        st.warning(f"Couldn't initialize {path}: {e}")

# Ensure expected files exist with correct headers
safe_ensure_file_with_header(LAST_ORDER_PATH, LAST_ORDER_COLUMNS)
safe_ensure_file_with_header(LOG_PATH, ORDER_LOG_COLUMNS)

# ---------- Utilities ----------
def clean_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a 'wide' sheet (name col + nearby numeric col) into tidy rows:
    ['item','product_number','current_qty','sort_order'].
    We preserve original input order via sort_order (NOT forced A→Z).
    """
    tidy_rows = []
    cols = df.columns.tolist()
    n = len(cols)
    i = 0
    while i < n:
        candidate_idxs = []
        if i + 1 < n: candidate_idxs.append(i + 1)
        if i + 2 < n: candidate_idxs.append(i + 2)
        if i + 3 < n: candidate_idxs.append(i + 3)
        chosen_prod_idx = None
        for idx in candidate_idxs:
            series = df.iloc[:, idx]
            numeric_like = pd.to_numeric(series, errors="coerce").notna().sum()
            if numeric_like >= max(5, len(series) * 0.1):
                chosen_prod_idx = idx
                break
        name_series = df.iloc[:, i].astype(str).str.strip()
        if chosen_prod_idx is not None:
            prod_series = pd.to_numeric(df.iloc[:, chosen_prod_idx], errors="coerce")
            for name, prod in zip(name_series, prod_series):
                if name and name.lower() != "nan" and pd.notna(prod):
                    tidy_rows.append({"item": name, "product_number": int(prod)})
            i += 2
        else:
            i += 1

    tidy = pd.DataFrame(tidy_rows).drop_duplicates().reset_index(drop=True)
    if tidy.empty:
        return pd.DataFrame(columns=["item", "product_number", "current_qty", "sort_order"])
    tidy["item"] = tidy["item"].str.replace(r"\s+", " ", regex=True).str.strip()
    tidy = tidy[tidy["item"].str.len() > 0]
    tidy["current_qty"] = 0
    tidy["sort_order"] = range(len(tidy))  # preserve original order
    return tidy[["item", "product_number", "current_qty", "sort_order"]]

def load_people():
    if PEOPLE_PATH.exists():
        txt = PEOPLE_PATH.read_text(encoding="utf-8").strip()
        if txt:
            return [p.strip() for p in txt.splitlines() if p.strip()]
    return []

def save_people(people):
    PEOPLE_PATH.write_text("\n".join(people), encoding="utf-8")

def init_catalog(upload_bytes):
    if upload_bytes is not None:
        raw = pd.read_csv(io.BytesIO(upload_bytes))
        tidy = clean_catalog(raw)
        if tidy.empty:
            st.error("Uploaded file couldn't be parsed into a catalog. Please check columns.")
        else:
            tidy.to_csv(CATALOG_PATH, index=False)
            st.success(f"Catalog created with {len(tidy)} items.")
    else:
        if not CATALOG_PATH.exists():
            st.info("No catalog found. Upload your supply CSV in the sidebar to create one.")

@st.cache_data
def read_catalog():
    df = safe_read_csv(CATALOG_PATH)
    if df.empty:
        return pd.DataFrame(columns=["item", "product_number", "current_qty", "sort_order"])
    for c in ["item", "product_number", "current_qty", "sort_order"]:
        if c not in df.columns:
            df[c] = pd.NA
    df["product_number"] = pd.to_numeric(df["product_number"], errors="coerce").astype("Int64")
    df["current_qty"]   = pd.to_numeric(df["current_qty"],   errors="coerce").fillna(0).astype(int)

    # Handle sort_order with an index-aligned filler
    if "sort_order" not in df.columns:
        df["sort_order"] = pd.Series(range(len(df)), index=df.index)
    else:
        so = pd.to_numeric(df["sort_order"], errors="coerce")
        filler = pd.Series(range(len(df)), index=df.index)
        so = so.fillna(filler)
        try:
            df["sort_order"] = so.astype(int)
        except ValueError:
            df["sort_order"] = filler.astype(int)

    return df[["item", "product_number", "current_qty", "sort_order"]].reset_index(drop=True)

def write_catalog(df: pd.DataFrame):
    df = df.copy()
    if "current_qty" not in df.columns:
        df["current_qty"] = 0
    if "sort_order" not in df.columns:
        df["sort_order"] = range(len(df))
    df["product_number"] = pd.to_numeric(df["product_number"], errors="coerce").astype("Int64")
    df["current_qty"]    = pd.to_numeric(df["current_qty"],    errors="coerce").fillna(0).astype(int)
    so = pd.to_numeric(df["sort_order"], errors="coerce")
    df["sort_order"] = so.fillna(pd.Series(range(len(df)), index=df.index)).astype(int)
    df.to_csv(CATALOG_PATH, index=False)

# APPEND (no overwrite) + DEDUPE
def append_log(order_df: pd.DataFrame, orderer: str):
    """
    Append new rows to order_log.csv and drop exact duplicates.
    Duplicate definition: same (item, product_number, qty, ordered_at, orderer).
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = order_df.copy()
    df["ordered_at"] = now
    df["orderer"] = orderer
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype(int)

    expected = ORDER_LOG_COLUMNS
    df = df[expected]

    prev = safe_read_csv(LOG_PATH)
    if not prev.empty:
        for c in expected:
            if c not in prev.columns:
                prev[c] = pd.NA
        prev = prev[expected]
        combined = pd.concat([prev, df], ignore_index=True)
        combined["qty"] = pd.to_numeric(combined["qty"], errors="coerce").fillna(0).astype(int)
        combined["product_number"] = pd.to_numeric(combined["product_number"], errors="coerce").astype("Int64")
        combined.drop_duplicates(subset=expected, keep="first", inplace=True)
    else:
        combined = df

    combined.to_csv(LOG_PATH, index=False)
    return now

def load_last_order() -> pd.DataFrame:
    df = safe_read_csv(LAST_ORDER_PATH)
    if df.empty:
        return pd.DataFrame(columns=LAST_ORDER_COLUMNS)
    for c in LAST_ORDER_COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype(int)
    return df[LAST_ORDER_COLUMNS]

def save_last_order(df: pd.DataFrame, orderer: str):
    try:
        out = df.copy()
        out["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        out["orderer"] = orderer
        out = out[LAST_ORDER_COLUMNS]
        out.to_csv(LAST_ORDER_PATH, index=False)
    except Exception as e:
        st.warning(f"Couldn't persist last order: {e}")

def last_order_info_map() -> pd.DataFrame:
    """
    Compute latest ordered_at and qty per (item, product_number) from the log.
    This powers the 'Last ordered' and 'Last qty' columns in the UI.
    """
    logs = safe_read_csv(LOG_PATH)
    needed = ["item", "product_number", "qty", "ordered_at"]
    if logs.empty or not all(c in logs.columns for c in needed):
        return pd.DataFrame(columns=["item", "product_number", "last_ordered_at", "last_qty"])
    logs["ordered_at"] = pd.to_datetime(logs["ordered_at"], errors="coerce")
    logs["qty"] = pd.to_numeric(logs["qty"], errors="coerce")
    logs = logs.dropna(subset=["ordered_at"])
    if logs.empty:
        return pd.DataFrame(columns=["item", "product_number", "last_ordered_at", "last_qty"])
    logs = logs.sort_values("ordered_at")
    idx = logs.groupby(["item", "product_number"], as_index=False).tail(1)
    return idx[["item", "product_number", "ordered_at", "qty"]].rename(
        columns={"ordered_at": "last_ordered_at", "qty": "last_qty"}
    )

# ---------- Sidebar: Setup & People ----------
st.sidebar.header("Setup")
uploaded = st.sidebar.file_uploader("Upload supply list (CSV)", type=["csv"], help="Wide or tidy CSV.")
if st.sidebar.button("Initialize / Replace Catalog"):
    init_catalog(uploaded.getvalue() if uploaded is not None else None)
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Orderers")
people = load_people()
add_person = st.sidebar.text_input("Add person")
if st.sidebar.button("Add to list") and add_person.strip():
    people.append(add_person.strip())
    people = sorted(set(people))
    save_people(people)
    st.sidebar.success(f"Added '{add_person.strip()}'")
    st.rerun()
if people:
    remove_person = st.sidebar.selectbox("Remove person", ["(choose)"] + people)
    if st.sidebar.button("Remove") and remove_person != "(choose)":
        people = [p for p in people if p != remove_person]
        save_people(people)
        st.sidebar.success(f"Removed '{remove_person}'")
        st.rerun()

# ---------- Main ----------
st.title("📦 Supply Ordering & Inventory Tracker")

tab_order, tab_inventory, tab_catalog, tab_logs = st.tabs(
    ["Create Order", "Adjust Inventory", "Edit Catalog", "Order Logs"]
)

# --- Catalog tab ---
with tab_catalog:
    st.subheader("Catalog")
    cat = read_catalog()
    st.dataframe(cat, use_container_width=True, hide_index=True)

    st.markdown("**Add new item**")
    c1, c2 = st.columns([2, 1])
    with c1:
        new_item = st.text_input("Item name")
    with c2:
        new_prod = st.text_input("Product #", help="Digits only if possible.")
    c3, c4 = st.columns([1, 1])
    with c3:
        new_qty = st.number_input("Starting qty (optional)", min_value=0, value=0, step=1)
    with c4:
        st.markdown("&nbsp;")
        if st.button("➕ Add item", use_container_width=True):
            if new_item.strip() and new_prod.strip():
                next_order = (cat["sort_order"].max() + 1) if not cat.empty else 0
                new_row = pd.DataFrame(
                    [{"item": new_item.strip(), "product_number": new_prod.strip(),
                      "current_qty": int(new_qty), "sort_order": int(next_order)}]
                )
                updated = pd.concat([cat, new_row], ignore_index=True).drop_duplicates(
                    subset=["item", "product_number"], keep="last"
                )
                write_catalog(updated)
                st.success(f"Added: {new_item.strip()}")
                st.rerun()
            else:
                st.error("Please provide both Item and Product #.")
    st.markdown("---")
    if not cat.empty:
        to_remove = st.multiselect("Select item(s) to remove", cat["item"].tolist())
        if st.button("🗑️ Remove selected"):
            updated = cat[~cat["item"].isin(to_remove)]
            write_catalog(updated)
            st.success(f"Removed {len(to_remove)} item(s).")
            st.rerun()

# --- Inventory tab ---
with tab_inventory:
    st.subheader("Adjust Inventory")
    cat = read_catalog()
    if cat.empty:
        st.info("No catalog yet. Initialize it from the sidebar.")
    else:
        st.write("Use +/− to update `current_qty` and/or `sort_order`, then **Save changes**.")
        editable = cat.copy()
        editable["current_qty"] = editable["current_qty"].astype(int)
        edited = st.data_editor(
            editable,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "item": st.column_config.TextColumn("Item", disabled=True),
                "product_number": st.column_config.TextColumn("Product #", disabled=True),
                "current_qty": st.column_config.NumberColumn("Current Qty", min_value=0, step=1),
                "sort_order": st.column_config.NumberColumn("Sort order", min_value=0, step=1),
            },
        )
        if st.button("💾 Save changes"):
            write_catalog(edited)
            st.success("Inventory saved.")

# --- Order tab ---
with tab_order:
    st.subheader("Create Order")
    cat = read_catalog()
    if cat.empty:
        st.info("No catalog yet. Initialize it from the sidebar.")
    else:
        # Last generated order (includes date & user)
        last_order_df = load_last_order()
        if not last_order_df.empty:
            st.markdown("**Last generated order (persists across sessions):**")
            st.dataframe(last_order_df, use_container_width=True, hide_index=True)

        orderer = st.selectbox(
            "Who is placing the order?",
            options=(people if people else ["(add names in sidebar)"]),
        )
        search = st.text_input("Search items")

        # Merge last-ordered info into catalog
        loi = last_order_info_map()
        table = cat.merge(loi, on=["item", "product_number"], how="left")

        # Convert last_ordered_at BEFORE sorting so sort works
        table["last_ordered_at"] = pd.to_datetime(table.get("last_ordered_at"), errors="coerce")

        # Sorting (default = newest first)
        sort_choice = st.selectbox(
            "Sort items by",
            options=["Original order", "Last ordered (newest first)", "Last ordered (oldest first)", "Product # asc", "Name A→Z"],
            index=1,  # <-- default newest first
        )
        if sort_choice == "Original order":
            table = table.sort_values(["sort_order", "item"], kind="stable")
        elif sort_choice == "Last ordered (newest first)":
            sort_key = table["last_ordered_at"].fillna(pd.Timestamp("1900-01-01"))
            table = table.iloc[sort_key.sort_values(ascending=False).index]
        elif sort_choice == "Last ordered (oldest first)":
            sort_key = table["last_ordered_at"].fillna(pd.Timestamp("2999-12-31"))
            table = table.iloc[sort_key.sort_values(ascending=True).index]
        elif sort_choice == "Product # asc":
            table = table.sort_values(["product_number", "item"], kind="stable")
        elif sort_choice == "Name A→Z":
            table = table.sort_values(["item"], kind="stable")

        # Search filter after sorting
        if search:
            table = table[table["item"].str.contains(search, case=False, na=False)]

        # Prepare UI columns (NO checkbox; just a Qty column)
        table["last_qty"] = pd.to_numeric(table.get("last_qty"), errors="coerce")
        table["qty"] = 0

        # Prefill qty from last generated order (optional convenience)
        if not last_order_df.empty:
            prev_map = {(r["item"], str(r["product_number"])): int(r["qty"]) for _, r in last_order_df.iterrows()}
            for i, r in table.iterrows():
                key = (r["item"], str(r["product_number"]))
                if key in prev_map:
                    table.at[i, "qty"] = int(prev_map[key])

        show_cols = ["qty", "item", "product_number", "last_ordered_at", "last_qty"]
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
            },
        )

        # --- TWO BUTTONS ---
        dec_inventory = st.checkbox("Decrement inventory by ordered qty")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🧾 Generate Order"):
                chosen = edited[edited["qty"] > 0].copy()
                if chosen.empty:
                    st.error("Please set Qty > 0 for at least one item.")
                else:
                    order_df = chosen[["item", "product_number", "qty"]].copy()
                    # Persist for next session (who & when generated)
                    save_last_order(order_df, orderer=(orderer if people and orderer != "(add names in sidebar)" else "(unknown)"))
                    st.success("Order generated.")
                    st.dataframe(order_df, use_container_width=True, hide_index=True)
                    csv_bytes = order_df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download CSV", data=csv_bytes,
                                       file_name=f"order_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                       mime="text/csv")
        with c2:
            if st.button("📝 Log Order"):
                chosen = edited[edited["qty"] > 0].copy()
                if chosen.empty:
                    st.error("Please set Qty > 0 for at least one item.")
                elif not people or orderer == "(add names in sidebar)":
                    st.error("Please add/select an orderer in the sidebar first.")
                else:
                    order_df = chosen[["item", "product_number", "qty"]].copy()
                    now = append_log(order_df, orderer)
                    st.success(f"Order logged at {now}.")
                    # Optional: decrement inventory
                    if dec_inventory:
                        cat2 = cat.copy()
                        # match on BOTH item and product_number
                        for _, r in order_df.iterrows():
                            mask = (cat2["item"] == r["item"]) & (cat2["product_number"].astype(str) == str(r["product_number"]))
                            cat2.loc[mask, "current_qty"] = (pd.to_numeric(cat2.loc[mask, "current_qty"], errors="coerce")
                                                             .fillna(0).astype(int) - int(r["qty"])).clip(lower=0)
                        write_catalog(cat2)
                        st.info("Inventory updated.")

        # Tools in Order tab — CLEAR last-ordered history here
        with st.expander("Tools"):
            st.caption("Clear controls: 'Last ordered' / 'Last qty' come from order history (not the screen list).")
            # build choices from current catalog view
            pairs = table[["item", "product_number"]].drop_duplicates().sort_values(["item", "product_number"])
            if pairs.empty:
                st.info("No items to clear.")
            else:
                pairs["label"] = pairs.apply(lambda r: f"{r['item']} — {r['product_number']}", axis=1)
                to_clear = st.multiselect("Select items to clear from last-ordered history", pairs["label"].tolist())
                colx, coly, colz = st.columns([1,1,2])
                with colx:
                    if st.button("🧹 Clear selected history"):
                        logs = safe_read_csv(LOG_PATH)
                        if logs.empty or not to_clear:
                            st.info("Nothing to clear.")
                        else:
                            sel = set(to_clear)
                            keep = ~logs.apply(lambda r: f"{r['item']} — {r['product_number']}" in sel, axis=1)
                            new_logs = logs[keep].copy()
                            new_logs.to_csv(LOG_PATH, index=False)
                            st.success(f"Cleared {len(logs) - len(new_logs)} log rows for selected items.")
                            st.rerun()
                with coly:
                    if st.button("🗑️ Clear ALL history"):
                        pd.DataFrame(columns=ORDER_LOG_COLUMNS).to_csv(LOG_PATH, index=False)
                        st.success("Cleared entire order history.")
                        st.rerun()
                with colz:
                    if st.button("🧼 Clear last generated order (screen list only)"):
                        pd.DataFrame(columns=LAST_ORDER_COLUMNS).to_csv(LAST_ORDER_PATH, index=False)
                        st.success("Cleared last generated order list (does not affect history).")
                        st.rerun()

# --- Logs tab ---
with tab_logs:
    st.subheader("Order Logs")
    logs = safe_read_csv(LOG_PATH)
    if not logs.empty:
        st.dataframe(logs.sort_values("ordered_at", ascending=False), use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download full log (CSV)",
            data=logs.to_csv(index=False).encode("utf-8"),
            file_name="order_log.csv",
            mime="text/csv",
        )
        st.markdown("### Clear 'Last ordered' history (same as Tools)")
        pairs = logs[["item", "product_number"]].drop_duplicates().sort_values(["item", "product_number"])
        pairs["label"] = pairs.apply(lambda r: f"{r['item']} — {r['product_number']}", axis=1)
        to_clear = st.multiselect("Select items to clear from history", pairs["label"].tolist(), key="log_clear")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 Clear selected", key="log_clear_btn"):
                if to_clear:
                    sel = set(to_clear)
                    keep = ~logs.apply(lambda r: f"{r['item']} — {r['product_number']}" in sel, axis=1)
                    new_logs = logs[keep].copy()
                    new_logs.to_csv(LOG_PATH, index=False)
                    st.success(f"Cleared {len(logs) - len(new_logs)} log rows for selected items.")
                    st.rerun()
                else:
                    st.info("No items selected.")
        with col2:
            if st.button("🗑️ Clear ALL history", key="log_clear_all"):
                pd.DataFrame(columns=ORDER_LOG_COLUMNS).to_csv(LOG_PATH, index=False)
                st.success("Cleared entire order history.")
                st.rerun()
    else:
        st.info("No orders logged yet.")
