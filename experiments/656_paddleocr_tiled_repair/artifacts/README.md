# Local artifacts

OCR overlays, tile provenance, model outputs, native artifact references and
private presets stay in ignored `.local/` or external artifact storage. For an
accepted run, preserve `spotting.jsonl`, `tile_provenance.jsonl`, `report.json`
and their SHA-256 values together. None of these generated files are committed.
