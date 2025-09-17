default:
    @just --list

dev:
  npm run dev

eval-all:
  @just eval article-tracking-trump
  @just eval article-tracking-trump/categories
  @just eval social-media-insults
  @just eval nhtsa-recalls
  @just eval political-fundraising-emails
  @just eval extract-fema-incidents
  @just eval grab-bag

eval config *ARGS:
    @echo "Running eval.py for {{config}}..."
    uv run eval.py src/evals/{{config}}/eval.yaml

test:
  uv run pytest
