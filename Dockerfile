FROM astrocrpublic.azurecr.io/runtime:3.3-4

# El codigo del proyecto vive en include/deadlock y se importa como `deadlock`.
ENV PYTHONPATH="/usr/local/airflow/include:${PYTHONPATH}"

# A diferencia de fifa_ingest (scraping de sofifa.com, protegido por Cloudflare),
# esta fuente es una API REST JSON estable (api.deadlock-api.com) que responde
# sin bloqueos ni necesidad de navegador. No hace falta Playwright/Chromium.
