📦 Supply Ordering App — Quick Start Guide

Welcome! This guide shows you (the day-to-day user) how to place a supply order, what gets saved where, and a few handy tools. No technical setup needed here—just how to use the app.

🚀 What you’ll do most days
1) Open the app

You’ll see the Create Order tab by default, plus a summary line like:
Loaded X catalog rows • Y log rows • Email configured: ✅ • Recipients discovered: Z

2) Pick your name

At the top left, choose “Who is ordering?” from the dropdown.

If your name isn’t listed, tell your admin to add it to data/people.txt (one name per line).

3) (Optional) Search and sort

Use Search items to filter the list.

Use Sort by (Last ordered / Original order / Name / Product #) if it helps.

4) Enter quantities

In the table, type the Qty for each item you want to order.

Leave Qty at 0 for items you aren’t ordering.

You can clear all quantities any time with 🧼 Clear quantities.

5) Log the order

Click one of the two buttons below the table:

🧾 Generate & Log Order — records the order without changing inventory.

🧾 Generate, Log, & Decrement — also reduces each item’s current inventory by the ordered amount.

That’s it! 🎉

✉️ What happens after you log

The app saves your order, emails the recipients, and clears the table for a fresh start.

The email subject is “Supply Order Logged” and it goes to the union of:

Everyone in data/emails.csv

Any default addresses the admin set in the app’s email settings

The email body lists:

When it was logged

Who ordered

The items and quantities

📂 Where things are saved

The app stores simple CSV files in the data/ folder next to the app:

data/order_log.csv — 📜 Full history of every logged order (item, product #, qty, time, orderer).

data/last_order.csv — 📋 The most recent generated order (used for the “Last generated” section).

data/catalog.csv — 🗂️ The list of available items. (Usually managed by an admin.)

data/people.txt — 👤 The list of orderers shown in the dropdown (one name per line).

data/emails.csv — ✉️ Who gets copied on emails (columns: name,email).
Example:

name,email
Leslie,leslie@example.com
Alex,alex@example.com


You can always download the full order history from the Order Logs tab.

🧭 Tabs overview
Create Order

Enter Qty for items you want.

🧾 Generate & Log or 🧾 Generate, Log, & Decrement.

🧼 Clear quantities to reset the table.

📋 Last generated order (expander at the top) lets you:

Copy/paste the latest order

Download it as a CSV

Adjust Inventory (usually for admins)

Edit Current Qty or Sort order and click 💾 Save inventory changes.

Catalog (usually for admins)

View the catalog.

Quick add new items.

Remove selected items.

Order Logs

View the full history of orders.

⬇️ Download full log (CSV) any time.

Tools (Danger Zone)

Clear on-screen quantities (just the current session values).

Clear last generated order (resets data/last_order.csv).

Clear order logs (resets data/order_log.csv).

✉️ Send test email to recipients (verifies email works).

🔎 SMTP diagnostics (shows the email config and lets you run a connection test).

🧼 Common tasks

I want to start over: Click 🧼 Clear quantities, then continue entering new numbers.

I need the CSV of my last order: Expand 📋 Last generated order and click ⬇️ Download CSV.

I want to confirm an email went out: Check your inbox (and the recipients’), or use Tools → 🔎 SMTP diagnostics then send a test email.

🆘 Troubleshooting (quick)

I type a number and it disappears:
This was fixed—quantities should now stick on the first try. If not, hit 🧼 Clear quantities and try again.

No one is getting emails:
Ask the admin to check Tools → 🔎 SMTP diagnostics (email setup) and verify addresses in data/emails.csv.

My name isn’t in the orderer list:
Ask the admin to add your name to data/people.txt (one per line) and refresh the app.

✅ Best practices

Only set Qty for items you actually want. Leave the rest at 0.

Use Generate, Log, & Decrement if you track on-hand inventory here.

Keep data/emails.csv up to date so the right people get notifications.
