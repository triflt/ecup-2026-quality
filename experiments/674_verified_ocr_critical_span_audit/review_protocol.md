# Review protocol

Open every image URL before rating a row. Do not infer missing text from the product title,
description, label, expected class, or similar products.

- `review_visual_critical_span_present`: `yes` only when image text itself contains a phrase
  that can materially determine the category rule; otherwise `no` or `unclear`.
- `review_ocr_captures_all_critical_text`: for a visual `yes`, use `yes` only when the verified
  OCR contains the complete decisive phrase. Use `na` when no/unclear visual span exists.
- `review_ocr_preserves_scope_relation`: for a visual `yes`, require that negation, inclusion,
  compatibility, and object-of-sale scope are not lost. Otherwise `na`.
- `review_unsupported_critical_text`: `yes` when OCR contains policy-critical text that is not
  actually visible in any source image.
- `review_evidence_relevant`: `yes` only when the OCR evidence concerns the sold item and the
  current category, not an accessory, compatibility statement, target of use, or generic text.

Fill all five ratings for a row or leave the entire row blank. Notes are optional.
