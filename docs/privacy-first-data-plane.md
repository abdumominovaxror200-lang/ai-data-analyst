# Privacy-first data plane

Every production LLM call is wrapped by one fail-closed egress boundary. Set
`LLM_EGRESS_MODE` explicitly during deployment:

- `external_redacted` keeps hosted-provider functionality. Dataset column names
  become stable request-local aliases, detected sensitive values are quarantined,
  and only schema metadata plus sanitized aggregate facts can leave the server.
- `local_only` preserves full local prompt functionality, but accepts only
  `localhost`, loopback, or literal private-network provider addresses. Public
  URLs and non-literal hostnames are rejected.
- `llm_disabled` keeps deterministic dataset tools and API endpoints available,
  while provider construction and calls are disabled.

The default is `external_redacted` so an existing hosted deployment remains
functional while adopting the safe boundary. Operators should set the variable
explicitly; raw-value external egress is no longer a supported mode.

## Local OpenAI-compatible servers

Ollama example (installation and model management are intentionally external to
this application):

```dotenv
LLM_EGRESS_MODE=local_only
LLM_BASE_URL=http://127.0.0.1:11434/v1
LLM_MODEL=your-local-model
LLM_API_KEY=
```

vLLM example:

```dotenv
LLM_EGRESS_MODE=local_only
LLM_BASE_URL=http://127.0.0.1:8000/v1
LLM_MODEL=your-local-model
LLM_API_KEY=
```

In containers, `127.0.0.1` is the application container itself. Use an explicit
private IP only when the model server is on a trusted private network. A hosted
model URL cannot be labeled `local_only`; validation fails before a request.

## Boundary and limitations

PII detection combines column-name hints with bounded value-pattern sampling for
email, phone, names, addresses, government identifiers, payment cards, and
sensitive identifiers. All source column names are aliased for external prompts,
not only columns classified as PII. Alias maps remain server-side and are scoped
to one dataset/provider instance.

No detector can infer every domain-specific secret. Deployments handling special
identifiers should use `local_only` or `llm_disabled`, and should still enforce
retention, access control, encryption, and log governance outside this boundary.
