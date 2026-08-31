import streamlit as st
import pandas as pd
from pandas.errors import EmptyDataError
import zoneinfo
from datetime import datetime
from pathlib import Path
import re
import smtplib
import ssl
from email.message import EmailMessage
from supabase import create_client

st.set_page_config(
    page_title="Supply Ordering",
    page_icon="📦",
    layout="wide",
)

NYC = zoneinfo.ZoneInfo("America/New_York")

# ---------------- Paths ----------------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CATALOG_PATH = DATA_DIR / "catalog.csv"
PEOPLE_PATH = DATA_DIR / "people.txt"
EMAILS_PATH = DATA_DIR / "emails.csv"
EMAILS2_PATH = DATA_DIR / "emails2.csv"

SPECIAL_ORDERER = "Greg"

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


# ---------------- Email (Gmail SMTP) ----------------
def _split_emails(txt: str) -> list[str]:
    if not txt:
        return []

    return [
        part.strip()
        for part in re.split(r"[;,]\s*", str(txt))
        if part.strip()
    ]


def get_email_config() -> dict:
    try:
        smtp_secrets = st.secrets["smtp"]

        return {
            "host": smtp_secrets.get("host", "smtp.gmail.com"),
            "port": int(smtp_secrets.get("port", 587)),
            "username": smtp_secrets.get("user", ""),
            "password": smtp_secrets.get("password", "").replace(" ", ""),
            "from": smtp_secrets.get("from", ""),
            "subject_prefix": smtp_secrets.get("subject_prefix", ""),
            "default_to": (
                _split_emails(smtp_secrets.get("to", ""))
                if smtp_secrets.get("to")
                else []
            ),
        }
    except Exception:
        return {}


def email_ok() -> bool:
    cfg = get_email_config()

    return all(
        cfg.get(key)
        for key in ["host", "username", "password", "from"]
    )


def send_email(
    subject: str,
    body: str,
    to_emails: list[str] | None,
    include_default_recipients: bool = True,
):
    cfg = get_email_config()

    recipient_list = list(to_emails or [])

    if include_default_recipients:
        recipient_list += cfg.get("default_to", [])

    recipients = sorted({
        email.strip()
        for email in recipient_list
        if email and "@" in email
    })

    if not recipients:
        raise RuntimeError("No recipients found.")

    prefix = cfg.get("subject_prefix", "")
    full_subject = f"{prefix}{subject}" if prefix else subject

    msg = EmailMessage()
    msg["Subject"] = full_subject
    msg["From"] = cfg["from"]
    msg["To"] = ", ".join(recipients)
    msg.add_alternative(body, subtype="html")

    with smtplib.SMTP(cfg["host"], cfg["port"]) as server:
        server.ehlo()
        server.starttls(context=ssl.create_default_context())
        server.login(cfg["username"], cfg["password"])
        server.send_message(msg)


# ---------------- Core data loaders ----------------
@st.cache_data
def read_people() -> list[str]:
    if not PEOPLE_PATH.exists():
        return []

    try:
        return [
            line.strip()
            for line in PEOPLE_PATH.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
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

    # Normalize column names.
    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    column_aliases = {
        "product_number": "product_number",
        "product number": "product_number",
        "multiplier_per_box": "multiplier",
        "multiplier": "multiplier",
        "recommended_qty_per_order": "items_per_order",
        "items_per_order": "items_per_order",
        "current_qty": "current_qty",
        "sort_order": "sort_order",
        "price": "price",
        "item": "item",
    }

    df = df.rename(columns=column_aliases)

    required_columns = [
        "item",
        "product_number",
        "multiplier",
        "items_per_order",
        "current_qty",
        "sort_order",
        "price",
    ]

    for column in required_columns:
        if column not in df.columns:
            df[column] = pd.NA

    df["item"] = df["item"].astype(str).str.strip()
    df["product_number"] = (
        df["product_number"]
        .astype(str)
        .str.strip()
    )

    df["multiplier"] = (
        pd.to_numeric(
            df["multiplier"],
            errors="coerce",
        )
        .fillna(1)
        .astype(int)
    )

    df["items_per_order"] = (
        pd.to_numeric(
            df["items_per_order"],
            errors="coerce",
        )
        .fillna(1)
        .astype(int)
    )

    df["current_qty"] = (
        pd.to_numeric(
            df["current_qty"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    df["price"] = (
        pd.to_numeric(
            df["price"],
            errors="coerce",
        )
        .fillna(0.0)
        .astype(float)
    )

    sort_order = pd.to_numeric(
        df["sort_order"],
        errors="coerce",
    )

    filler = pd.Series(
        range(len(df)),
        index=df.index,
    )

    df["sort_order"] = (
        sort_order
        .fillna(filler)
        .astype(int)
    )

    return df.reset_index(drop=True)


def write_catalog(df: pd.DataFrame):
    df.to_csv(CATALOG_PATH, index=False)
    read_catalog.clear()


# ---------------- Supabase helpers ----------------
def append_log(
    order_df: pd.DataFrame,
    orderer: str,
) -> str:
    now_str = datetime.now(NYC).isoformat(
        sep=" ",
        timespec="seconds",
    )

    rows = [
        {
            "item": row["item"],
            "product_number": str(
                row["product_number"]
            ),
            "qty": int(row["qty"]),
            "ordered_at": now_str,
            "orderer": orderer,
        }
        for _, row in order_df.iterrows()
    ]

    supabase.table("orders_log").insert(
        rows
    ).execute()

    return now_str


def read_log() -> pd.DataFrame:
    response = (
        supabase
        .table("orders_log")
        .select("*")
        .order("ordered_at", desc=True)
        .execute()
    )

    if not getattr(response, "data", None):
        return pd.DataFrame(
            columns=[
                "item",
                "product_number",
                "qty",
                "ordered_at",
                "orderer",
            ]
        )

    return pd.DataFrame(response.data)


def last_info_map() -> pd.DataFrame:
    logs = read_log()

    if logs.empty:
        return pd.DataFrame(
            columns=[
                "item",
                "product_number",
                "last_ordered_at",
                "last_qty",
                "last_orderer",
            ]
        )

    logs["ordered_at"] = pd.to_datetime(
        logs["ordered_at"],
        errors="coerce",
    )

    latest_rows = (
        logs
        .sort_values("ordered_at")
        .groupby(
            ["item", "product_number"],
            as_index=False,
        )
        .tail(1)
    )

    return latest_rows.rename(
        columns={
            "ordered_at": "last_ordered_at",
            "qty": "last_qty",
            "orderer": "last_orderer",
        }
    )[
        [
            "item",
            "product_number",
            "last_ordered_at",
            "last_qty",
            "last_orderer",
        ]
    ]


# ---------------- Email recipient files ----------------
@st.cache_data
def read_email_file(path: Path) -> pd.DataFrame:
    df = safe_read_csv(path)

    if df.empty:
        return pd.DataFrame(
            columns=["name", "email"]
        )

    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    email_pattern = re.compile(
        r"([A-Za-z0-9._%+\-]+@"
        r"[A-Za-z0-9.\-]+\.[A-Za-z]{2,})"
    )

    rows = []

    if "email" in df.columns:
        for _, row in df.iterrows():
            match = email_pattern.search(
                str(row.get("email", ""))
            )

            if match:
                rows.append({
                    "name": str(
                        row.get("name", "")
                    ).strip(),
                    "email": match.group(1).strip(),
                })

    return pd.DataFrame(rows)


def all_recipients(
    emails_df: pd.DataFrame,
) -> list[str]:
    cfg = get_email_config()

    file_recipients = (
        emails_df["email"].tolist()
        if not emails_df.empty
        else []
    )

    return sorted({
        email
        for email in (
            file_recipients
            + cfg.get("default_to", [])
        )
        if email and "@" in email
    })


def recipients_for_orderer(
    orderer: str,
    normal_emails_df: pd.DataFrame,
    special_emails_df: pd.DataFrame,
) -> tuple[list[str], bool]:
    is_special_orderer = (
        str(orderer).strip().casefold()
        == SPECIAL_ORDERER.casefold()
    )

    if is_special_orderer:
        recipients = (
            special_emails_df["email"].tolist()
            if not special_emails_df.empty
            else []
        )

        recipients = sorted({
            email
            for email in recipients
            if email and "@" in email
        })

        # True means Greg gets only emails2.csv,
        # without the default SMTP recipient list.
        return recipients, True

    return all_recipients(normal_emails_df), False


# ---------------- Email body builder ----------------
def build_email_body(
    qty_map: dict,
    catalog: pd.DataFrame,
    orderer: str,
    when_str: str,
) -> str:
    items = []

    for product_number, qty in qty_map.items():
        if qty <= 0:
            continue

        matching_rows = catalog.loc[
            catalog["product_number"].astype(str)
            == str(product_number)
        ]

        if matching_rows.empty:
            continue

        item_name = matching_rows.iloc[0]["item"]
        price = float(
            matching_rows.iloc[0].get(
                "price",
                0,
            )
            or 0
        )
        category = matching_rows.iloc[0].get("category", "")

        items.append(
            (
                product_number,
                qty,
                item_name,
                qty * price,
                category,
            )
        )

    # First-fit bin packing.
    bins: list[list[tuple]] = []
    bin_totals: list[float] = []

    for item in items:
        _, _, _, total, _ = item
        placed = False

        for index, bin_total in enumerate(
            bin_totals
        ):
            if bin_total + total <= 4999:
                bins[index].append(item)
                bin_totals[index] += total
                placed = True
                break

        if not placed:
            bins.append([item])
            bin_totals.append(total)

    def _category_sort_key(category):
        try:
            return (0, float(category))
        except (TypeError, ValueError):
            return (1, str(category))

    details_items = sorted(
        items,
        key=lambda item: _category_sort_key(item[4]),
    )

    details_lines = [
        (
            "<label>"
            "<input type='checkbox'/> "
            f"{item_name} (#{product_number}): {qty}"
            "</label>"
        )
        for (
            product_number,
            qty,
            item_name,
            _,
            _,
        ) in details_items
    ]

    group_lines = [
        (
            "<label>"
            "<input type='checkbox'/> "
            f"{', '.join(f'&quot;{product_number}&quot;' for product_number, *_ in group)}"
            f" = ${subtotal:,.0f}"
            "</label>"
        )
        for group, subtotal in zip(
            bins,
            bin_totals,
        )
    ]

    return f"""
    <html>
        <body>
            <p>
                <strong>New supply order at {when_str}</strong><br>
                Ordered by: {orderer}
            </p>

            <p>
                <strong>Details:</strong><br>
                {"<br>".join(details_lines)}
            </p>

            <p>
                <strong>Product groups (≤$4,999 each):</strong><br>
                {"<br>".join(group_lines)}
            </p>
        </body>
    </html>
    """


def create_order_dataframe(
    qty_map: dict,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    full_order = [
        {
            "item": (
                catalog.loc[
                    catalog["product_number"].astype(str)
                    == str(product_number)
                ]
                .iloc[0]["item"]
            ),
            "product_number": product_number,
            "qty": qty,
        }
        for product_number, qty in qty_map.items()
        if (
            qty > 0
            and not catalog.loc[
                catalog["product_number"].astype(str)
                == str(product_number)
            ].empty
        )
    ]

    return pd.DataFrame(full_order)


def log_and_email_order(
    orderer: str,
    qty_map: dict,
    catalog: pd.DataFrame,
    normal_emails_df: pd.DataFrame,
    special_emails_df: pd.DataFrame,
):
    full_order_df = create_order_dataframe(
        qty_map,
        catalog,
    )

    if full_order_df.empty:
        st.warning("No items selected.")
        return False

    when_str = append_log(
        full_order_df,
        orderer,
    )

    st.success(f"Order logged at {when_str}.")

    if email_ok():
        recipients, special_only = recipients_for_orderer(
            orderer,
            normal_emails_df,
            special_emails_df,
        )

        if recipients:
            body = build_email_body(
                qty_map,
                catalog,
                orderer,
                when_str,
            )

            try:
                send_email(
                    "Supply Order Logged",
                    body,
                    recipients,
                    include_default_recipients=(
                        not special_only
                    ),
                )

                if special_only:
                    st.success(
                        "Greg's order email was sent only "
                        f"to the emails2.csv recipient list "
                        f"({len(recipients)} recipient(s))."
                    )
                else:
                    st.success(
                        "Email sent to "
                        f"{len(recipients)} recipient(s)."
                    )

            except Exception as e:
                st.error(f"Email failed: {e}")
        else:
            if special_only:
                st.error(
                    "Greg was selected, but no valid "
                    "email address was found in "
                    "data/emails2.csv."
                )
            else:
                st.error(
                    "No valid email recipients were found."
                )

    return True


# ---------------- Session state init ----------------
if "orderer" not in st.session_state:
    st.session_state["orderer"] = None

if "qty_map" not in st.session_state:
    st.session_state["qty_map"] = {}


# ---------------- Load data ----------------
people = read_people()
emails_df = read_email_file(EMAILS_PATH)
special_emails_df = read_email_file(
    EMAILS2_PATH
)
catalog = read_catalog()
logs = read_log()


# ---------------- Page header ----------------
st.title("📦 Supply Ordering & Inventory Tracker")

cfg = get_email_config()

email_ready = (
    "✅"
    if email_ok()
    else (
        "❌ (keys found: "
        f"{list(key for key, value in cfg.items() if value)})"
    )
)

st.caption(
    f"Loaded {len(catalog)} catalog rows • "
    f"{len(logs)} log rows • "
    f"Email configured: {email_ready}"
)


# ---------------- Running order preview ----------------
selected_items = [
    {
        "item": (
            catalog.loc[
                catalog["product_number"].astype(str)
                == str(product_number)
            ]
            .iloc[0]["item"]
        ),
        "product_number": product_number,
        "qty": qty,
    }
    for product_number, qty
    in st.session_state["qty_map"].items()
    if (
        qty > 0
        and not catalog.loc[
            catalog["product_number"].astype(str)
            == str(product_number)
        ].empty
    )
]

if selected_items:
    st.markdown(
        "### 🛒 Current Order (in progress)"
    )

    selected_df = pd.DataFrame(
        selected_items
    )

    st.dataframe(
        selected_df,
        hide_index=True,
        use_container_width=True,
    )

    st.markdown(
        "**Product Numbers:** "
        + ", ".join(
            str(item["product_number"])
            for item in selected_items
        )
    )

    if st.button(
        "🧾 Generate & Log Order",
        key="gen_log_top",
    ):
        top_orderer = (
            st.session_state.get("orderer")
            or (
                people[0]
                if people
                else "Unknown"
            )
        )

        completed = log_and_email_order(
            top_orderer,
            st.session_state["qty_map"],
            catalog,
            emails_df,
            special_emails_df,
        )

        if completed:
            st.session_state["qty_map"] = {}
            st.rerun()

else:
    st.caption(
        "🛒 No items currently selected."
    )


# ================================================================
tabs = st.tabs(
    [
        "Create Order",
        "Adjust Inventory",
        "Catalog",
        "Order Logs",
    ]
)


# ----------------------------------------------------------------
# Tab 0 — Create Order
# ----------------------------------------------------------------
with tabs[0]:
    if catalog.empty:
        st.info(
            "No catalog found. "
            "Add items to data/catalog.csv."
        )
    else:
        column_1, column_2 = st.columns(
            [2, 3]
        )

        with column_1:
            current_orderer = (
                st.session_state.get("orderer")
                or (
                    people[0]
                    if people
                    else "Unknown"
                )
            )

            orderer = st.selectbox(
                "Who is ordering?",
                options=(
                    people
                    if people
                    else ["Unknown"]
                ),
                index=(
                    people.index(current_orderer)
                    if (
                        people
                        and current_orderer in people
                    )
                    else 0
                ),
            )

            st.session_state["orderer"] = orderer

        with column_2:
            search = st.text_input(
                "🔍 Search items"
            )

        # Merge last-order information.
        last_map = last_info_map()

        table = catalog.merge(
            last_map,
            on=["item", "product_number"],
            how="left",
        )

        for column in [
            "last_ordered_at",
            "last_qty",
            "last_orderer",
        ]:
            if column not in table.columns:
                table[column] = pd.NA

        table["last_ordered_at"] = (
            pd.to_datetime(
                table["last_ordered_at"],
                errors="coerce",
            )
        )

        table = (
            table
            .sort_values(
                ["last_ordered_at", "item"],
                ascending=[False, True],
                na_position="last",
            )
            .reset_index(drop=True)
        )

        table["product_number"] = (
            table["product_number"]
            .astype(str)
        )

        table["qty"] = (
            table["product_number"]
            .map(
                lambda product_number: (
                    st.session_state["qty_map"].get(
                        product_number,
                        0,
                    )
                )
            )
            .astype(int)
        )

        if search:
            mask = table["item"].str.contains(
                search,
                case=False,
                na=False,
            )

            mask |= table[
                "product_number"
            ].str.contains(
                search,
                case=False,
                na=False,
            )

            table = table[mask]

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
                "qty": st.column_config.NumberColumn(
                    "Qty",
                    min_value=0,
                    step=1,
                ),
                "item": st.column_config.TextColumn(
                    "Item",
                    disabled=True,
                ),
                "product_number": st.column_config.TextColumn(
                    "Product #",
                    disabled=True,
                ),
                "multiplier": st.column_config.NumberColumn(
                    "Multiplier",
                    disabled=True,
                ),
                "items_per_order": st.column_config.NumberColumn(
                    "Items/Order",
                    disabled=True,
                ),
                "current_qty": st.column_config.NumberColumn(
                    "Current Qty",
                    disabled=True,
                ),
                "price": st.column_config.NumberColumn(
                    "Price",
                    disabled=True,
                ),
                "last_ordered_at": st.column_config.DatetimeColumn(
                    "Last Ordered",
                    format="YYYY-MM-DD HH:mm",
                    disabled=True,
                ),
                "last_qty": st.column_config.NumberColumn(
                    "Last Qty",
                    disabled=True,
                ),
                "last_orderer": st.column_config.TextColumn(
                    "Last By",
                    disabled=True,
                ),
            },
            key="order_editor",
        )

        # Only rerun when a quantity changes
        # to a non-zero value.
        rerun_needed = False

        for _, row in edited.iterrows():
            new_qty = (
                int(row["qty"])
                if pd.notna(row["qty"])
                else 0
            )

            product_number = str(
                row["product_number"]
            )

            old_qty = (
                st.session_state["qty_map"]
                .get(product_number, 0)
            )

            if old_qty != new_qty:
                st.session_state["qty_map"][
                    product_number
                ] = new_qty

                if new_qty != 0:
                    rerun_needed = True

        if rerun_needed:
            st.rerun()

        if st.button(
            "🧾 Generate & Log Order",
            key="gen_log_main",
        ):
            completed = log_and_email_order(
                orderer,
                st.session_state["qty_map"],
                catalog,
                emails_df,
                special_emails_df,
            )

            if completed:
                st.session_state["qty_map"] = {}
                st.rerun()

        if st.button(
            "🧹 Clear Current Order"
        ):
            st.session_state["qty_map"] = {}
            st.rerun()


# ----------------------------------------------------------------
# Tab 1 — Adjust Inventory
# ----------------------------------------------------------------
with tabs[1]:
    if catalog.empty:
        st.info("No catalog found.")
    else:
        st.write(
            "Adjust `current_qty`, `sort_order`, "
            "or `price`, then save."
        )

        edited_inventory = st.data_editor(
            catalog.copy().reset_index(
                drop=True
            ),
            use_container_width=True,
            hide_index=True,
            column_config={
                "item": st.column_config.TextColumn(
                    "Item",
                    disabled=True,
                ),
                "product_number": st.column_config.TextColumn(
                    "Product #",
                    disabled=True,
                ),
                "multiplier": st.column_config.NumberColumn(
                    "Multiplier",
                    min_value=1,
                    step=1,
                ),
                "items_per_order": st.column_config.NumberColumn(
                    "Items/Order",
                    min_value=1,
                    step=1,
                ),
                "current_qty": st.column_config.NumberColumn(
                    "Current Qty",
                    min_value=0,
                    step=1,
                ),
                "sort_order": st.column_config.NumberColumn(
                    "Sort Order",
                    min_value=0,
                    step=1,
                ),
                "price": st.column_config.NumberColumn(
                    "Price ($)",
                    min_value=0.0,
                    step=0.01,
                ),
            },
            key="inventory_editor",
        )

        if st.button(
            "💾 Save inventory changes"
        ):
            write_catalog(edited_inventory)
            st.success("Inventory saved.")


# ----------------------------------------------------------------
# Tab 2 — Catalog
# ----------------------------------------------------------------
with tabs[2]:
    st.caption(
        "Catalog source: data/catalog.csv"
    )

    if catalog.empty:
        st.info("No catalog found.")
    else:
        st.dataframe(
            catalog,
            use_container_width=True,
            hide_index=True,
        )


# ----------------------------------------------------------------
# Tab 3 — Order Logs
# ----------------------------------------------------------------
with tabs[3]:
    if logs.empty:
        st.info("No orders logged yet.")
    else:
        st.dataframe(
            logs,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download full log (CSV)",
            data=logs.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="order_log.csv",
            mime="text/csv",
        )
