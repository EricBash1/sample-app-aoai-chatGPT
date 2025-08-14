# .github/workflows/deploy.yml
name: Build & deploy (dev/prod)

on:
  push:
    branches: [ main, dev ]
  workflow_dispatch:

env:
  PYTHON_VERSION: "3.11"
  NODE_VERSION: "20.x"
  PROD_APP_NAME: "ase-proposal-files"
  DEV_APP_NAME: "ase-proposal-files-dev"   # <-- change if your dev app has a different name

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      # --- Python (no venv needed in CI; App Service will use Oryx to install deps) ---
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      # --- Node / Frontend ---
      - name: Set up Node
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: npm
          cache-dependency-path: frontend/package-lock.json

      - name: Install frontend deps
        working-directory: frontend
        run: npm ci

      - name: Build frontend
        working-directory: frontend
        run: NODE_OPTIONS=--max_old_space_size=8192 npm run build

      # --- Stitch build output where Quart expects it ---
      - name: Prepare backend/static
        run: |
          rm -rf backend/static
          mkdir -p backend/static
          cp -r frontend/dist/* backend/static/

      # --- Ensure requirements.txt is in the zip root for Oryx ---
      - name: Copy requirements.txt into backend
        run: cp requirements.txt backend/requirements.txt

      # --- (Optional) sanity check
      - name: List backend contents
        run: ls -la backend && ls -la backend/static | head -n 50

      # --- Package a single deployable zip ---
      - name: Create artifact
        run: |
          cd backend
          zip -r ../app.zip .
          cd ..
      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: app
          path: app.zip

  deploy:
    runs-on: ubuntu-latest
    needs: build

    steps:
      - uses: actions/download-artifact@v4
        with:
          name: app

      # DEV deploy (branch == dev)
      - name: Deploy to Azure Web App (DEV)
        if: github.ref == 'refs/heads/dev'
        uses: azure/webapps-deploy@v3
        with:
          app-name: ${{ env.DEV_APP_NAME }}
          publish-profile: ${{ secrets.AZURE_WEBAPP_PUBLISH_PROFILE_DEV }}
          package: app.zip

      # PROD deploy (branch == main)
      - name: Deploy to Azure Web App (PROD)
        if: github.ref == 'refs/heads/main'
        uses: azure/webapps-deploy@v3
        with:
          app-name: ${{ env.PROD_APP_NAME }}
          publish-profile: ${{ secrets.AZURE_WEBAPP_PUBLISH_PROFILE }}
          package: app.zip
