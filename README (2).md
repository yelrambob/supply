# 📦 Supply Ordering App — README

A simple Streamlit app for logging supply orders and notifying your team by email.

---

## 🚀 Quick Start (Everyday Use)

1. **Open the app**  
   You’ll land on **Create Order**.

2. **Choose your name**  
   Use the **“Who is ordering?”** dropdown.  
   > If you don’t see your name, ask your admin to add it to `data/people.txt` (one name per line).

3. **Find your items**  
   - Use **Search items** to filter.  
   - Change **Sort by** if helpful (Last ordered, Original order, Name, Product #).

4. **Enter quantities**  
   Type a **Qty** for each item you’re ordering. Leave others at **0**.  
   - Use **🧼 Clear quantities** to reset the table.

5. **Log the order**  
   - **🧾 Generate & Log Order** → saves the order (no inventory changes)  
   - **🧾 Generate, Log, & Decrement** → saves and **subtracts** from inventory

The app emails your recipients and clears the form for the next order. 🎉

---

## ✉️ What Happens After You Click “Log”

- The order is written to the log.
- A notification email is sent to:
  - Everyone in `data/emails.csv`, **plus**
  - Any default addresses configured in Streamlit Secrets (`[smtp].to`)
- The app clears the table so you can start fresh.

---

## 📂 Where Things Are Saved

All files live in the `data/` folder next to `app.py`:

- `data/order_log.csv` — 📜 **Full order history** (item, product #, qty, timestamp, orderer)
- `data/last_order.csv` — 📋 The **most recent** generated order (for quick copy/download)
- `data/catalog.csv` — 🗂️ Catalog of items (managed by admin)
- `data/people.txt` — 👤 Names shown in the orderer dropdown (one per line)
- `data/emails.csv` — ✉️ Email recipients  
  Example:
  ```csv
  name,email
  Leslie,leslie@example.com
  Alex,alex@example.com
  ```

> You can download the full log anytime from the **Order Logs** tab.

---

## 🧭 Tabs Overview

### Create Order
- Enter **Qty** for items to order  
- **🧾 Generate & Log** or **🧾 Generate, Log, & Decrement**  
- **🧼 Clear quantities** to reset  
- **📋 Last generated order** (expander): copy the last order or **⬇️ Download CSV**

### Adjust Inventory _(admin)_
- Edit **Current Qty** or **Sort order**, then **💾 Save inventory changes**

### Catalog _(admin)_
- View the catalog  
- **Quick add** new items  
- **Remove** selected items

### Order Logs
- View all past orders  
- **⬇️ Download full log (CSV)**

### Tools (Danger Zone)
- Clear on-screen quantities (session-only)  
- Clear last generated order  
- Clear order logs  
- **✉️ Send test email to recipients**  
- **🔎 SMTP diagnostics**: show email config + run a connection test

---

## 🧼 Common Tasks

- **Start over mid-entry** → Click **🧼 Clear quantities**, then re-enter Qty.  
- **Get last order as CSV** → Expand **📋 Last generated order** → **⬇️ Download CSV**.  
- **Confirm email delivery** → Check inboxes or use **Tools → ✉️ Send test email**.

---

## 🆘 Troubleshooting

- **Qty disappears / have to type twice**  
  Fixed. If it happens, click **🧼 Clear quantities** once and re-enter.

- **No emails sent**  
  Ask admin to check **Tools → 🔎 SMTP diagnostics** and recipients in `data/emails.csv`.

- **My name isn’t in the dropdown**  
  Ask admin to add your name to `data/people.txt` (one name per line), then refresh.

---

## ✅ Best Practices

- Only set **Qty** for items you’re ordering; leave others at **0**.  
- Use **Generate, Log, & Decrement** if you want inventory to update automatically.  
- Keep `data/emails.csv` current so the right people are notified.

---

Happy ordering! 🙌
