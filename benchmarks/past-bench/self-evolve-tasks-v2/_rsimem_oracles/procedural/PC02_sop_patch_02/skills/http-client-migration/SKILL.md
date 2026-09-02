# HTTP Client Migration Guide (Corrected)

The `requests` library is deprecated. Tag PRs using it with `deprecated-api`
and `needs-migration`, set priority to medium and category to code-review.
`httpx` is the recommended replacement and `aiohttp` is allowed for async-only
use. Apply the corrected rule consistently regardless of team or department.
