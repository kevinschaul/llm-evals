default:
    @just --list

view:
  promptfoo view --yes

run config:
  @just eval {{config}}
  @just aggregate {{config}}

eval config:
    # Ignore error code 100, which means at least one eval failed
    @echo "Running promptfoo eval for {{config}}..."
    cd src/evals/{{config}} && promptfoo eval \
      -c promptfoo-config.yaml \
      --output results/results.csv \
      || test $? -eq 100

aggregate config:
  @echo "Aggregating results for {{config}}..."
  python aggregate_results.py src/evals/{{config}}/results/results.csv

view-dashboard:
  npm run dev

