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

# Last line here ignore errors code 100, which means at least one eval failed
eval config:
    @echo "Running promptfoo eval for {{config}}..."
    cd src/evals/{{config}} && promptfoo eval \
      -j 1 \
      --env-path .env \
      -c promptfoo-config.yaml \
      --output results/results.csv \
      || test $? -eq 100
    @echo "Aggregating results for {{config}}..."
    python aggregate_results.py src/evals/{{config}}/results/results.csv

view-dashboard:
  npm run dev

