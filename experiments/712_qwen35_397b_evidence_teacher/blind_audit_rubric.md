# Blind-200 explanation audit

Reviewer receives `category`, the complete title/description, image count, the
four-field explanation payload without its gold `label`, and `image_files`
pointing to every SHA-verified card image under `blind200_images/`. Rows are
deterministically shuffled and contain 50 supported explanations
from each category×gold cell; cell membership is hidden.

For every row record exactly one primary verdict:

- `PASS`: the comment names the sold object, states the decisive policy
  relation, and every factual claim is supported by the cited text span or the
  cited image.
- `UNSUPPORTED`: the explanation introduces a material fact not visible in the
  card or overstates what the cited evidence proves.
- `WRONG_RELATION`: the cited fact is real, but included/standalone/empty/
  mention-only or BAD-marking semantics are wrong.
- `IRRELEVANT`: the comment is generic or does not explain this card.
- `FORMAT`: the comment is not 50–300 characters or the evidence contract is
  broken despite automated validation.

Acceptance requires at least 189/200 `PASS`, zero `UNSUPPORTED`, and no repeated
template that erases the card-specific sold object or decisive relation. Any
image-cited row must be viewed; text-only rows may be checked against the exact
quoted span and surrounding sentence.
