# Deploying the webpage + Streamlit keep-alive on your AWS Ubuntu server

This covers three things:

1. Hosting a webpage (with images) on the Ubuntu box, served by nginx.
2. Keeping that same box running a scheduled job that wakes/pings the
   Streamlit app (`ordered.streamlit.app`) so it doesn't fall asleep.
3. Notes for adding a real domain + HTTPS later ("figure out the address later").

## 1. One-time server setup

SSH into the EC2/Lightsail instance, then:

```bash
sudo apt update && sudo apt install -y nginx python3-venv git
```

In the AWS console, open inbound ports **80** (and **443** once you add TLS) on the
instance's security group. Port 22 (SSH) should already be open to your IP.

## 2. Clone the repo onto the server

```bash
sudo mkdir -p /opt/supply
sudo chown $USER:$USER /opt/supply
git clone https://github.com/yelrambob/supply.git /opt/supply
cd /opt/supply
python3 -m venv venv
./venv/bin/pip install playwright
./venv/bin/playwright install --with-deps chromium
```

## 3. Host the webpage + images with nginx

Put your page and images here (create your own `index.html`; this repo doesn't
ship one yet):

```bash
sudo mkdir -p /var/www/supply/html /var/www/supply/images
sudo chown -R $USER:www-data /var/www/supply
# copy your page + images, e.g.:
#   scp index.html ubuntu@<server-ip>:/var/www/supply/html/
#   scp photo1.jpg ubuntu@<server-ip>:/var/www/supply/images/
```

Reference images from the page as `/images/<filename>` (see
`deploy/nginx-supply.conf`, which maps that path to
`/var/www/supply/images/`).

Install the site config:

```bash
sudo cp /opt/supply/deploy/nginx-supply.conf /etc/nginx/sites-available/supply
sudo ln -s /etc/nginx/sites-available/supply /etc/nginx/sites-enabled/supply
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```

Visit `http://<server-public-ip>/` to confirm it's serving.

## 4. Keep the Streamlit app awake from this server

A plain `curl` does **not** wake a sleeping Streamlit Community Cloud app —
the "Yes, get this app back up!" button has to actually be clicked in a real
browser. `.github/scripts/wake_streamlit.py` does this with Playwright; the
systemd units below reuse that same script on your server instead of relying
on GitHub Actions.

(This repo used to also have a `keep-alive.yml` workflow that just curled the
app every 6 hours — it was removed because it never actually worked, curl
can't click a JS button. The remaining `wake-app.yml` workflow uses the same
Playwright script as below and now runs every 6 hours too.)

```bash
sudo cp /opt/supply/deploy/wake-streamlit.service /etc/systemd/system/
sudo cp /opt/supply/deploy/wake-streamlit.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now wake-streamlit.timer

# sanity check
sudo systemctl start wake-streamlit.service   # runs it once, right now
sudo journalctl -u wake-streamlit.service -n 30 --no-pager
```

The timer fires every 6 hours by default (edit `OnCalendar=` in
`wake-streamlit.timer` to change that, then `sudo systemctl daemon-reload &&
sudo systemctl restart wake-streamlit.timer`).

Once this is running reliably you can disable the GitHub Actions versions to
avoid double-pinging: `.github/workflows/wake-app.yml` and
`.github/workflows/keep-alive.yml` (either delete them or disable via the
GitHub Actions UI).

## 5. Domain + HTTPS (later)

When you pick a domain:

1. Point an A record at the server's public IP (Route 53 or your registrar).
2. `sudo apt install -y certbot python3-certbot-nginx`
3. `sudo certbot --nginx -d your.domain.com` — this edits
   `/etc/nginx/sites-available/supply` in place to add the HTTPS server block
   and sets up auto-renewal.

Until then, `server_name _;` in the nginx config just serves whatever
hostname/IP is used to reach the box.
