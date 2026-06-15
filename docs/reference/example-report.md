# Example interactive report

The toolkit's `bundle()` / report renderer emits a self-contained interactive
HTML report. MkDocs copies everything under `docs/` into the built site verbatim,
so a report bundle dropped under `docs/reference/bundles/` is embeddable directly
via an `<iframe>`:

<iframe
  src="bundles/example-report.html"
  width="100%"
  height="600"
  style="border: 1px solid #ccc;"
  title="Example interactive report">
</iframe>

> **Placeholder.** The embedded file above is a stub that proves the embedding
> mechanism. The real anonymized UVA and Frontier report bundles are added in a
> later release-content task, replacing the placeholder asset in place.
