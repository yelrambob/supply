# 📦 Streamlit Supply Ordering & Inventory Tracker

This app lets your department track supplies, generate an order list with product numbers, log orders with timestamp + orderer, and adjust inventory. It also supports adding/removing items from your catalog.

## Files
- `streamlit_supply_tracker.py` — the Streamlit app.
- `data/catalog.csv` — cleaned catalog generated from your uploaded CSV.
- `data/order_log.csv` — created automatically after your first logged order.
- `data/people.txt` — list of orderers (one name per line).

## Run locally
```bash
pip install streamlit pandas
streamlit run streamlit_supply_tracker.py
```
The app starts in your browser. Use the sidebar to initialize the catalog from your CSV, and manage the list of orderers.

## Typical workflow
1. **Initialize catalog**: Upload your wide CSV and click **Initialize / Replace Catalog** in the sidebar. The app cleans it into a tidy `catalog.csv` with columns: `item`, `product_number`, `current_qty`.
2. **Create Order** (tab):
   - Pick the **orderer**.
   - Search/filter and select items, set **quantities**.
   - Click **Generate Order List** to see a table and download a **CSV** or a plain text list you can copy into your company’s ordering system.
   - Click **Log this order** to save it to `data/order_log.csv` with timestamp and orderer. Optionally tick **Decrement inventory by ordered qty**.
3. **Adjust Inventory** (tab):
   - Edit `current_qty` and save.
4. **Edit Catalog** (tab):
   - Add new items (name + product # + starting qty).
   - Remove selected items.
5. **Order Logs** (tab):
   - View and download your full order history CSV.

## Deploy on GitHub + Streamlit Community Cloud
1. Create a new repo and add these files.
2. On [share.streamlit.io](https://share.streamlit.io), point to `streamlit_supply_tracker.py`.
3. Add a `data/` folder in the repo so logs can be created at runtime (or configure a persistent storage solution).

## Notes
- The catalog cleaner pairs each text column with the next numeric-like column in your uploaded CSV. If any items are missed, you can add them via the **Edit Catalog** tab.
- You can replace the catalog at any time by re-uploading in the sidebar (this overwrites `data/catalog.csv`).
