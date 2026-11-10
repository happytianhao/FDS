# Bounded request recovery

`recovery.auto_resume_after_seconds` defaults to `0` (disabled). Set it to `60`
for one automatic cooldown after transport retries exhaust. Values must be finite
numbers in `0..3600`. A positive value replaces the manual pause for that request.
The original retry budget is renewed once: `transport_retries: 3` permits at most
8 requests (4 before and 4 after cooldown). Backoff restarts after cooldown.
No simulator steps, tool execution, or memory updates occur during this synchronous
recovery; context copies and image hashes enforce unchanged inference inputs.
Existing defaults and manual resume behavior remain unchanged when disabled.

Only recognized transient transport, timeout, rate-limit, and service errors can
retry. Authentication, exhausted quota, schema/context errors, unknown errors,
shutdown signals, and Codex failures after completed turns or tool events cannot.
`classify_request_failure(error_or_message)` returns `failure_category`,
`retryable`, and `global_blocker`; quota/auth/authorization block further requests.
It does not assign official task success or failure.

Standard HTTP `Retry-After` seconds and dates are honored. If the requested wait
exceeds the configured retry maximum, the error is returned without sending an
early retry. At cooldown boundaries it must fit the automatic cooldown; manual
resume is disabled for that failure because an operator could resume too early.

Every attempted request retains its own stats, including unknown usage (`null`)
and requested model, effort, sandbox, and approval settings. These settings carry
`configuration_provenance: requested` and `configuration_verified: false`;
they are not evidence that the provider used the settings. Invalid partial stats
are retained as `stats.partial.json`. The aggregate stats record request time,
retry delays, recovery pauses, and total wall time. Interrupting a cooldown records
an interrupted recovery and sends no additional request.
