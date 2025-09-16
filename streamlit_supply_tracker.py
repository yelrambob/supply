import streamlit as st
import pandas as pd
import io
from datetime import datetime
from pathlib import Path

st.set_page_config(page_title='Supply Tracker', page_icon='📦', layout='wide')

DATA_DIR = Path('data')
DATA_DIR.mkdir(exist_ok=True)
CATALOG_PATH = DATA_DIR / 'catalog.csv'
LOG_PATH = DATA_DIR / 'order_log.csv'
PEOPLE_PATH = DATA_DIR / 'people.txt'

def clean_catalog(df: pd.DataFrame) -> pd.DataFrame:
    tidy_rows = []
    cols = df.columns.tolist()
    n = len(cols)
    i = 0
    while i < n:
        candidate_idxs = []
        if i+1 < n: candidate_idxs.append(i+1)
        if i+2 < n: candidate_idxs.append(i+2)
        if i+3 < n: candidate_idxs.append(i+3)
        chosen_prod_idx = None
        for idx in candidate_idxs:
            series = df.iloc[:, idx]
            numeric_like = pd.to_numeric(series, errors='coerce').notna().sum()
            if numeric_like >= max(5, len(series)*0.1):
                chosen_prod_idx = idx
                break
        name_series = df.iloc[:, i].astype(str).str.strip()
        if chosen_prod_idx is not None:
            prod_series = pd.to_numeric(df.iloc[:, chosen_prod_idx], errors='coerce')
            for name, prod in zip(name_series, prod_series):
                if name and name.lower() != 'nan' and pd.notna(prod):
                    tidy_rows.append({'item': name, 'product_number': int(prod)})
            i += 2
        else:
            i += 1
    tidy = pd.DataFrame(tidy_rows).drop_duplicates().reset_index(drop=True)
    tidy['item'] = tidy['item'].str.replace(r'\s+',' ', regex=True).str.strip()
    tidy = tidy[tidy['item'].str.len() > 0]
    return tidy

def load_people() -> list:
    if PEOPLE_PATH.exists():
        txt = PEOPLE_PATH.read_text(encoding='utf-8').strip()
        if txt:
            return [p.strip() for p in txt.splitlines() if p.strip()]
    return []

def save_people(people: list):
    PEOPLE_PATH.write_text('\n'.join(people), encoding='utf-8')

def init_catalog(upload: bytes | None):
    if upload is not None:
        raw = pd.read_csv(io.BytesIO(upload))
        tidy = clean_catalog(raw)
        if tidy.empty:
            st.error("Uploaded file couldn't be parsed into a catalog. Please check columns.")
        else:
            tidy = tidy.sort_values('item').reset_index(drop=True)
            tidy['current_qty'] = 0
            tidy.to_csv(CATALOG_PATH, index=False)
            st.success(f'Catalog created with {len(tidy)} items.')
    else:
        if not CATALOG_PATH.exists():
            st.info('No catalog found. Upload your supply CSV in the sidebar to create one.')

@st.cache_data
def read_catalog():
    if CATALOG_PATH.exists():
        return pd.read_csv(CATALOG_PATH)
    return pd.DataFrame(columns=['item','product_number','current_qty'])

def write_catalog(df: pd.DataFrame):
    df = df.copy()
    if 'current_qty' not in df.columns:
        df['current_qty'] = 0
    df['product_number'] = pd.to_numeric(df['product_number'], errors='coerce').astype('Int64')
    df['current_qty'] = pd.to_numeric(df['current_qty'], errors='coerce').fillna(0).astype(int)
    df = df.sort_values('item').reset_index(drop=True)
    df.to_csv(CATALOG_PATH, index=False)

st.sidebar.header('Setup')
uploaded = st.sidebar.file_uploader('Upload supply list (CSV)', type=['csv'], help='Your wide or tidy CSV.')
if st.sidebar.button('Initialize / Replace Catalog'):
    init_catalog(uploaded.getvalue() if uploaded is not None else None)

st.sidebar.divider()
st.sidebar.subheader('Orderers')
people = load_people()
add_person = st.sidebar.text_input('Add person')
if st.sidebar.button('Add to list') and add_person.strip():
    people.append(add_person.strip())
    people = sorted(set(people))
    save_people(people)
    st.sidebar.success(f"Added '{add_person.strip()}'")
    st.rerun()
if people:
    remove_person = st.sidebar.selectbox('Remove person', ['(choose)'] + people)
    if st.sidebar.button('Remove') and remove_person != '(choose)':
        people = [p for p in people if p != remove_person]
        save_people(people)
        st.sidebar.success(f"Removed '{remove_person}'")
        st.rerun()

st.title('📦 Supply Ordering & Inventory Tracker')

tab_order, tab_inventory, tab_catalog, tab_logs = st.tabs(['Create Order','Adjust Inventory','Edit Catalog','Order Logs'])

with tab_catalog:
    st.subheader('Catalog')
    cat = read_catalog()
    st.dataframe(cat, use_container_width=True, hide_index=True)
    st.markdown('**Add new item**')
    c1, c2 = st.columns([2,1])
    with c1:
        new_item = st.text_input('Item name')
    with c2:
        new_prod = st.text_input('Product #', help='Digits only if possible.')
    c3, c4 = st.columns([1,1])
    with c3:
        new_qty = st.number_input('Starting qty (optional)', min_value=0, value=0, step=1)
    with c4:
        st.markdown('&nbsp;')
        if st.button('➕ Add item', use_container_width=True):
            if new_item.strip() and new_prod.strip():
                new_row = pd.DataFrame([{'item': new_item.strip(), 'product_number': new_prod.strip(), 'current_qty': int(new_qty)}])
                updated = pd.concat([cat, new_row], ignore_index=True).drop_duplicates(subset=['item','product_number'], keep='last')
                write_catalog(updated)
                st.success(f'Added: {new_item.strip()}')
                st.rerun()
            else:
                st.error('Please provide both Item and Product #.')
    st.markdown('---')
    if not cat.empty:
        to_remove = st.multiselect('Select item(s) to remove', cat['item'].tolist())
        if st.button('🗑️ Remove selected'):
            updated = cat[~cat['item'].isin(to_remove)]
            write_catalog(updated)
            st.success(f'Removed {len(to_remove)} item(s).')
            st.rerun()

with tab_inventory:
    st.subheader('Adjust Inventory')
    cat = read_catalog()
    if cat.empty:
        st.info('No catalog yet. Initialize it from the sidebar.')
    else:
        st.write('Use +/− to update `current_qty` and then **Save changes**.')
        editable = cat.copy()
        editable['current_qty'] = editable['current_qty'].astype(int)
        edited = st.data_editor(
            editable,
            num_rows='dynamic',
            use_container_width=True,
            hide_index=True,
            column_config={
                'item': st.column_config.TextColumn('Item', disabled=True),
                'product_number': st.column_config.TextColumn('Product #', disabled=True),
                'current_qty': st.column_config.NumberColumn('Current Qty', min_value=0, step=1),
            },
        )
        if st.button('💾 Save changes'):
            write_catalog(edited)
            st.success('Inventory saved.')

with tab_order:
    st.subheader('Create Order')
    cat = read_catalog()
    if cat.empty:
        st.info('No catalog yet. Initialize it from the sidebar.')
    else:
        orderer = st.selectbox('Who is placing the order?', options=(people if people else ['(add names in sidebar)']))
        search = st.text_input('Search items')
        filtered = cat[cat['item'].str.contains(search, case=False, na=False)] if search else cat
        with st.container(border=True):
            st.markdown('**Select items and set quantities**')
            selected_items = st.multiselect('Items', filtered['item'].tolist())
            quantities = {}
            for item in selected_items:
                row = cat[cat['item'] == item].iloc[0]
                quantities[item] = st.number_input(f"Qty for '{item}' (#{row['product_number']})", min_value=1, value=1, step=1, key=f'qty_{item}')
            if st.button('🧾 Generate Order List'):
                if not selected_items:
                    st.error('Please select at least one item.')
                else:
                    rows = []
                    for item in selected_items:
                        row = cat[cat['item'] == item].iloc[0]
                        rows.append({'item': item, 'product_number': row['product_number'], 'qty': int(quantities[item])})
                    order_df = pd.DataFrame(rows, columns=['item','product_number','qty'])
                    st.success('Order list created.')
                    st.dataframe(order_df, use_container_width=True, hide_index=True)
                    csv_bytes = order_df.to_csv(index=False).encode('utf-8')
                    st.download_button('⬇️ Download CSV', data=csv_bytes, file_name=f"order_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime='text/csv')
                    txt_lines = [f"{r['item']} — {r['product_number']} — Qty {r['qty']}" for _, r in order_df.iterrows()]
                    st.text_area('Copy-paste for your other system', value='\n'.join(txt_lines), height=150)
                    if st.button('📝 Log this order'):
                        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        logged = order_df.copy()
                        logged['ordered_at'] = now
                        logged['orderer'] = orderer
                        if LOG_PATH.exists():
                            prev = pd.read_csv(LOG_PATH)
                            combined = pd.concat([prev, logged], ignore_index=True)
                        else:
                            combined = logged
                        combined.to_csv(LOG_PATH, index=False)
                        st.success(f'Order logged at {now}.')
                        if st.checkbox('Decrement inventory by ordered qty'):
                            cat2 = cat.copy()
                            for _, r in order_df.iterrows():
                                idx = cat2.index[cat2['item'] == r['item']][0]
                                cat2.loc[idx, 'current_qty'] = max(0, int(cat2.loc[idx, 'current_qty']) - int(r['qty']))
                            cat2.to_csv(CATALOG_PATH, index=False)
                            st.info('Inventory updated.')

with tab_logs:
    st.subheader('Order Logs')
    if LOG_PATH.exists():
        logs = pd.read_csv(LOG_PATH)
        st.dataframe(logs.sort_values('ordered_at', ascending=False), use_container_width=True, hide_index=True)
        st.download_button('⬇️ Download full log (CSV)', data=logs.to_csv(index=False).encode('utf-8'), file_name='order_log.csv', mime='text/csv')
    else:
        st.info('No orders logged yet.')