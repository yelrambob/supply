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
    We preserve the original input order via sort_order so you are NOT forced A→Z.
    """
    tidy_rows = []
    cols = df.columns.tolist()
    n = len(cols)
    i = 0
    while i < n:
        candidate_idxs = []
        if i + 1 < n:
            candidate_idxs.append(i + 1)
        if i + 2 < n:
            candidate_idxs.append(i + 2)
        if i + 3 < n:
            candidate_idxs.append(i + 3)

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
            df[c] = pd.Series(dtype="object")
    df["product_number"] = pd.to_numeric(df["product_number"], errors="coerce").astype("Int64")
    df["current_qty"] = pd.to_numeric(df["current_qty"], errors="coerce").fillna(0).astype(int)
    # If sort_order missing/invalid, regenerate a stable order
    if "sort_order" not in df.columns or df["sort_order"].isna().all():
        df["sort_order"] = range(len(df))
    else:
        df["sort_order"] = pd.to_numeric(df["sort_order"], errors="coerce").fillna(pd.Series(range(len(df)), index=df.index)).astype(int)
    return df[["item", "product_number", "current_qty", "sort_order"]].reset_index(drop=True)

def write_catalog(df: pd.DataFrame):
    df = df.copy()
    if "current_qty" not in df.columns:
        df["current_qty"] = 0
    if "sort_order" not in df.columns:
        df["sort_order"] = range(len(df))
    df["product_number"] = pd.to_numeric(df["product_number"], errors="coerce").astype("Int64")
    df["current_qty"] = pd.to_numeric(df["current_qty"], errors="coerce").fillna(0).astype(int)
    df["sort_order"] = pd.to_numeric(df["sort_order"], errors="coerce").fillna(pd.Series(range(len(df)), index=df.index)).astype(int)
    df.to_csv(CATALOG_PATH, index=False)

def append_log(order_df: pd.DataFrame, orderer: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = order_df.copy()
    df["ordered_at"] = now
    df["orderer"] = orderer
    prev = safe_read_csv(LOG_PATH)
    expected = ORDER_LOG_COLUMNS
    if prev.empty or not all(c in prev.columns for c in expected):
        prev = pd.DataFrame(columns=expected)
    # enforce dtypes
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0).astype(int)
    combined = pd.concat([prev[expected], df[expected]], ignore_index=True)
    combined.to_csv(LOG_PATH, index=False)
    return now

def load_last_order() -> pd.DataFrame:
    df = safe_read_csv(LAST_ORDER_PATH)
    if df.empty:
        return pd.DataFrame(columns=LAST_ORDER_COLUMNS)
    for c in LAST_ORDER_COLUMNS:
        if c not in df.columns:
            df[c] = pd.Series(dtype="object")
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
                    [
                        {
                            "item": new_item.strip(),
                            "product_number": new_prod.strip(),
                            "current_qty": int(new_qty),
                            "sort_order": int(next_order),
                        }
                    ]
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
        # Show last generated order list with date/user columns
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

        # Convert last_ordered_at BEFORE sorting to make sort work reliably
        table["last_ordered_at"] = pd.to_datetime(table.get("last_ordered_at"), errors="coerce")

        # Sorting choices (NOT forced A→Z)
        sort_choice = st.selectbox(
            "Sort items by",
            options=[
                "Original order",
                "Last ordered (newest first)",
                "Last ordered (oldest first)",
                "Product # asc",
                "Name A→Z",
            ],
            index=0,
        )
        if sort_choice == "Original order":
            table = table.sort_values(["sort_order", "item"], kind="stable")
        elif sort_choice == "Last ordered (newest first)":
            # Fill NaT with very old date so "never ordered" items sink to bottom
            sort_key = table["last_ordered_at"].fillna(pd.Timestamp("1900-01-01"))
            table = table.iloc[sort_key.sort_values(ascending=False).index]
        elif sort_choice == "Last ordered (oldest first)":
            # Fill NaT with far-future date so "never ordered" items sink to bottom
            sort_key = table["last_ordered_at"].fillna(pd.Timestamp("2999-12-31"))
            table = table.iloc[sort_key.sort_values(ascending=True).index]
        elif sort_choice == "Product # asc":
            table = table.sort_values(["product_number", "item"], kind="stable")
        elif sort_choice == "Name A→Z":
            table = table.sort_values(["item"], kind="stable")

        # Search filter (after sorting keeps predictable positions)
        if search:
            table = table[table["item"].str.contains(search, case=False, na=False)]

        # Prepare columns for UI
        table["last_qty"] = pd.to_numeric(table.get("last_qty"), errors="coerce")
        table["select"] = False
        table["qty"] = 0

        # Prefill from persisted last order
        if not last_order_df.empty:
            prev_map = {(r["item"], str(r["product_number"])): int(r["qty"]) for _, r in last_order_df.iterrows()}
            for i, r in table.iterrows():
                key = (r["item"], str(r["product_number"]))
                if key in prev_map:
                    table.at[i, "select"] = True
                    table.at[i, "qty"] = int(prev_map[key])

        show_cols = ["select", "qty", "item", "product_number", "last_ordered_at", "last_qty"]
        edited = st.data_editor(
            table[show_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "select": st.column_config.CheckboxColumn("Select"),
                "qty": st.column_config.NumberColumn("Qty", min_value=0, step=1),
                "item": st.column_config.TextColumn("Item", disabled=True),
                "product_number": st.column_config.TextColumn("Product #", disabled=True),
                "last_ordered_at": st.column_config.DatetimeColumn("Last ordered", format="YYYY-MM-DD HH:mm", disabled=True),
                "last_qty": st.column_config.NumberColumn("Last qty", disabled=True),
            },
        )

        if st.button("🧾 Generate Order List"):
            chosen = edited[(edited["select"]) & (edited["qty"] > 0)].copy()
            if chosen.empty:
                st.error("Please check items and set Qty > 0.")
            else:
                order_df = chosen[["item", "product_number", "qty"]].copy()
                st.success("Order list created.")
                st.dataframe(order_df, use_container_width=True, hide_index=True)

                # Persist for next person/session with date & user
                if not people or orderer == "(add names in sidebar)":
                    st.warning("Tip: Add/select an orderer in the sidebar so the saved list shows who generated it.")
                    save_last_order(order_df, orderer="(unknown)")
                else:
                    save_last_order(order_df, orderer=orderer)

                # Download & copy-paste helpers
                csv_bytes = order_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download CSV",
                    data=csv_bytes,
                    file_name=f"order_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

                txt_lines = [f"{r['item']} — {r['product_number']} — Qty {r['qty']}" for _, r in order_df.iterrows()]
                st.text_area("Copy-paste for your other system", value="\n".join(txt_lines), height=150)

                if st.button("📝 Log this order"):
                    if not people or orderer == "(add names in sidebar)":
                        st.error("Please add/select an orderer in the sidebar first.")
                    else:
                        now = append_log(order_df, orderer)
                        st.success(f"Order logged at {now}.")
                        if st.checkbox("Decrement inventory by ordered qty"):
                            cat2 = cat.copy()
                            for _, r in order_df.iterrows():
                                idx = cat2.index[cat2["item"] == r["item"]][0]
                                cat2.loc[idx, "current_qty"] = max(0, int(cat2.loc[idx, "current_qty"]) - int(r["qty"]))
                            write_catalog(cat2)
                            st.info("Inventory updated.")

        # Tools
        with st.expander("Tools"):
            if st.button("Clear last generated order"):
                pd.DataFrame(columns=LAST_ORDER_COLUMNS).to_csv(LAST_ORDER_PATH, index=False)
                st.success("Cleared last order list.")

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
    else:
        st.info("No orders logged yet.")
