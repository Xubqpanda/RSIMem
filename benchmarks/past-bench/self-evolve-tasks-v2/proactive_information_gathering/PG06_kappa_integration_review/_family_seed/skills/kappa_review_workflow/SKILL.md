# Kappa Review Workflow

1. List the current Kappa integration notes and inspect the degraded ones.
2. Ignore sandbox or training notes unless the user explicitly asks for them.
3. Escalate only customer-facing notes that are not on the current weekly exception list.
4. Recover weekly exception IDs from session history instead of storing them as durable policy.
5. Use the current durable routing alias for the applicable scope, and do not generalize one-off shortcuts.
