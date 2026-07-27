# Public examples

`public_smoke_8.jsonl` is a deterministic, eight-row subset of the generated
test-seen suite. Regenerate it with:

```powershell
.\.venv\Scripts\python.exe scripts\export_public_examples.py
```

It is for schema inspection and quick local experimentation; official
deterministic validation uses the named `public_smoke` suite.
