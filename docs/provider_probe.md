# Provider completion probe

Before starting a formal PAST-Bench batch, run the provider-only probe against
the exact base URL and model that the manifest will use:

```bash
OPENAI_API_KEY="$GPT_LUNA_API_KEY" \
  .venv/bin/python -m rsimem.provider_probe \
  --base-url "$RSIMEM_PAST_BASE_URL" \
  --model "$RSIMEM_PAST_MODEL"
```

The probe sends one bounded completion request and reports only HTTP status,
whether non-empty assistant content was returned, and whether a usage object
was present.  It never prints the response body or credential.  A successful
probe is only a connectivity gate; formal audit still requires complete usage,
process corpus, clean-tree identity, and all matched-run invariants.
The result contract is fail-closed: a successful result must carry a 2xx
status, and every unsuccessful result must carry an explicit error code.

An `empty_content`, `invalid_json`, `http_error`, or `transport_error` result
must not be used to start a formal batch.  Probe output is provider diagnostics,
not task feedback or optimizer input.
