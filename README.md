# Azure Function - My Content

HTTP Azure Function that returns five article recommendations for a `user_id`.
Known users use an Implicit ALS model over a catalog of 5,000 recent articles.
Unknown users receive a popularity fallback from the same catalog.

## API

`POST /api/recommend`

```json
{"user_id": 123}
```

The response contains `recommendations`, latent-factor `scores`, the selected
`strategy`, and model metadata.

## Model artifacts

Generate the files by running the notebook `notebooks/10_deploiement_als.ipynb`.
It writes them to `deployment_artifacts_notebook/` without overwriting
`deployment_artifacts/` (the files currently deployed), then compares both sets.

To update the model, upload every file of `deployment_artifacts_notebook/` to
the private `models` Blob Storage container.

## Configuration

For local execution, copy `local.settings.example.json` to
`local.settings.json`. In Azure, configure these application settings:

- `MODEL_CONTAINER=models`
- `STORAGE_ACCOUNT_URL=https://<storage-account>.blob.core.windows.net`

The Function App managed identity needs the `Storage Blob Data Reader` role on
the storage account.
