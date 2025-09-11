default:
    @just --list

view:
  promptfoo view --yes

eval-all:
  @just eval article-tracking-trump
  @just eval article-tracking-trump/categories
  @just eval social-media-insults
  @just eval nhtsa-recalls
  @just eval political-fundraising-emails
  @just eval grab-bag

eval config *ARGS:
    @echo "Running promptfoo eval for {{config}}..."
    uv run eval.py src/evals/{{config}}/eval.yaml

view-dashboard:
  npm run dev
