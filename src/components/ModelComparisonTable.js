import * as htl from "htl"

export default function ModelComparisonTable(results) {
  // Handle case where results data might not be loaded yet
  if (!results || results.length === 0) {
    return {
      columns: ["test", "providers_compared", "responses"],
      format: {
        responses: () => htl.html`<div style="color: #666; font-style: italic;">No data available</div>`,
      },
      required: false,
      multiple: false,
    }
  }

  // Group results by test to compare model responses
  const grouped = {};
  results.forEach(row => {
    const testKey = row.test || row.prompt_id || "unknown";
    if (!grouped[testKey]) {
      grouped[testKey] = {
        test: testKey,
        responses: [],
        providers: new Set()
      };
    }
    grouped[testKey].responses.push({
      provider: row.provider_id,
      result: row.result,
      passed: row.passed,
      error: row.error,
      duration_ms: row.duration_ms
    });
    grouped[testKey].providers.add(row.provider_id);
  });

  // Convert to array format for table
  const comparisonData = Object.values(grouped).map(group => ({
    test: group.test,
    providers_compared: Array.from(group.providers).join(", "),
    responses: group.responses
  }));

  return {
    columns: ["test", "providers_compared", "responses"],
    format: {
      test: (d) => d && d.length > 50 ? d.substring(0, 50) + "..." : d,
      providers_compared: (d) => htl.html`<span style="font-size: 0.9em; color: #666;">${d}</span>`,
      responses: (responses) => {
        return htl.html`<div style="display: flex; flex-direction: column; gap: 8px; max-width: 600px;">
          ${responses.map(response => htl.html`
            <div style="
              border: 1px solid #e0e0e0;
              border-radius: 4px;
              padding: 8px;
              background: ${response.error ? '#ffeaea' : (response.passed === true || response.passed === 'True' ? '#e8f5e8' : response.passed === false || response.passed === 'False' ? '#ffeaea' : '#f8f9fa')};
            ">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <strong style="color: #333;">${response.provider}</strong>
                <div style="display: flex; align-items: center; gap: 8px;">
                  ${response.passed !== null && response.passed !== undefined ? 
                    (response.passed === true || response.passed === 'True' || response.passed === 'true') ?
                      htl.html`<span style="background: #d5edca; padding: 2px 6px; border-radius: 3px; font-size: 0.8em;">✔</span>` :
                      htl.html`<span style="background: #f9dddb; padding: 2px 6px; border-radius: 3px; font-size: 0.8em;">✗</span>` :
                    htl.html`<span style="background: #e3f2fd; padding: 2px 6px; border-radius: 3px; font-size: 0.8em;">?</span>`
                  }
                  <span style="font-size: 0.8em; color: #666;">${response.duration_ms}ms</span>
                </div>
              </div>
              ${response.error ? 
                htl.html`<div style="color: #d32f2f; font-size: 0.9em; font-family: monospace;">${response.error}</div>` :
                htl.html`<div style="color: #333; font-size: 0.9em; line-height: 1.4;">${response.result || 'No response'}</div>`
              }
            </div>
          `)}
        </div>`;
      }
    },
    align: {
      providers_compared: "center",
    },
    widths: {
      test: 150,
      providers_compared: 120,
      responses: 600,
    },
    layout: "fixed",
    required: false,
    multiple: false,
  }
}