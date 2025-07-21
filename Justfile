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
eval config *ARGS:
    @just start-llm-provider
    @echo "Running promptfoo eval for {{config}}..."
    cd src/evals/{{config}} && promptfoo eval \
      -j 1 \
      --no-cache \
      --env-path ../../.env \
      -c promptfoo-config.yaml \
      --output results/results.csv \
      {{ ARGS }} \
      || test $? -eq 100
    @just stop-llm-provider
    @echo "Aggregating results for {{config}}..."
    python aggregate_results.py src/evals/{{config}}/results/results.csv

view-dashboard:
  npm run dev

# TODO make this better, e.g. this does not detect if the port is already taken
start-llm-provider:
    @echo "Starting llm provider..."
    @pkill -f llm_provider.py || true
    uv run llm_provider.py > server.log 2>&1 &
    @echo "llm provider started on http://127.0.0.1:4242"

stop-llm-provider:
    @echo "Stopping llm provider..."
    @pkill -f llm_provider.py || true
    @echo "llm provider stopped"

