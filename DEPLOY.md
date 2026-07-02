# Deploying Demo 1.0 to Streamlit Community Cloud

Private, team-only, free. Data is ephemeral (resets on reboot/redeploy) — fine for the demo.

## Final repo layout

    demo/
    ├── pulse_demo.py
    ├── dup_finder_service.py
    ├── requirements.txt
    ├── .gitignore
    └── data/
        ├── cnic_front/.gitkeep
        ├── cnic_back/.gitkeep
        ├── electricity_bill/.gitkeep
        └── predictions/
            ├── duplicated/.gitkeep
            ├── non_duplicated/.gitkeep
            └── inconclusive/.gitkeep

The `.gitkeep` files keep the (otherwise empty) folders in git. The app also
creates any missing folders at startup, so they're belt-and-suspenders.

## Steps

1. Create a GitHub repo (set it **Private**).

2. Add these files to it:
   - your `pulse_demo.py` and `dup_finder_service.py`
   - `requirements.txt`  (uses opencv-python-headless — no system packages needed)
   - `.gitignore`
   - the `data/` folder with its `.gitkeep` files

       git init
       git add .
       git commit -m "PULSE Demo 1.0"
       git branch -M main
       git remote add origin https://github.com/<you>/<repo>.git
       git push -u origin main

3. Go to https://share.streamlit.io → sign in with GitHub → **Create app** →
   **Deploy a public app from GitHub** (it can read private repos once authorized).
     - Repository:   <you>/<repo>
     - Branch:       main
     - Main file:    pulse_demo.py
     - (optional) pick a custom subdomain
   Click **Deploy**. First build takes a few minutes (compiling OpenCV wheel).

4. Make it private to your team:
   App → **Settings** → **Sharing** → set "Who can view this app" to the
   allowlist, and add your teammates' emails. NOTE: Community Cloud's free tier
   allows **one** private app. If you need more private apps, see alternatives below.

## Why opencv-python-headless

The standard `opencv-python` wheel links against GUI libs (libGL) that aren't
present on Streamlit Cloud, causing `ImportError: libGL.so.1`. The headless
wheel is the same library minus the GUI bits — correct for a server. No
`packages.txt` is needed as a result.

## Important caveats for THIS app

- **Ephemeral storage**: every reboot/redeploy wipes `data/`. Galleries rebuild
  from scratch, so duplicates only accumulate within a single uptime window.
  That matches your "demo only" choice. When you want persistence later, move
  the gallery + predictions to object storage (S3/Azure Blob/GCS) or a mounted
  volume on a host that supports it (Render disks, Railway volumes, a VM).
- **One private app limit** on the free tier. Alternatives if you outgrow it:
    - Hugging Face Spaces (Streamlit SDK) — private spaces allowed
    - Render / Railway — free-ish tiers, support persistent disks
    - Azure App Service / Container Apps — natural fit alongside Databricks,
      and keeps everything in your cloud tenant

## Local sanity check before pushing

    pip install -r requirements.txt
    streamlit run pulse_demo.py
