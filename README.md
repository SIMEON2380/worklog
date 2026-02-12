
# Work Log Simple Frontend (Streamlit + SQLite)

This is a simple web UI that matches your Excel columns and stores everything in a proper database (SQLite).
It can also import your existing Excel workbook.

## What you get
- Add new work entries via a form
- View/search/filter entries
- Edit directly in a grid and save
- Delete rows by ID
- Import your existing Excel (.xlsx)

## Run it (quick)
1) Install Python 3.10+
2) In this folder, install deps:

```bash
pip install -r requirements.txt
```

3) Start the app:

```bash
streamlit run app.py
```

It’ll create `worklog.db` in the same folder automatically.

## Put it on your home server
- Run it on the server and access it from your LAN.
- If you want access from outside, put it behind HTTPS + auth (Nginx Proxy Manager / Caddy).

## Swap SQLite for Postgres later (easy)
If you outgrow SQLite, we can switch the storage layer to Postgres without changing the UI much.
