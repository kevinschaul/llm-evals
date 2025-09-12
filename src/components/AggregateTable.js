import sparkBar from "./sparkBar.js"

export default function AggregateTable(aggregate) {
  // Handle case where aggregate data might not be loaded yet
  if (!aggregate || aggregate.length === 0) {
    return {
      columns: ["provider_id", "total_tests", "passed", "failed", "errors", "pass_rate"],
      sort: "pass_rate",
      reverse: true,
      format: {
        pass_rate: (d) => sparkBar(1)(d / 100),
      },
      align: {
        pass_rate: "right",
      },
    }
  }
  
  // Check if data has prompt_id column (multi-prompt evaluation)
  const hasPromptId = aggregate[0].prompt_id !== undefined;
  
  // Check if data has no_assertions column (tri-state assertion system)
  const hasNoAssertions = aggregate[0].no_assertions !== undefined;
  
  // Build columns array based on available data
  let columns = ["provider_id"];
  if (hasPromptId) {
    columns.push("prompt_id");
  }
  columns.push("total_tests", "passed", "failed");
  if (hasNoAssertions) {
    columns.push("no_assertions");
  }
  columns.push("errors", "pass_rate");
  
  return {
    columns,
    sort: "pass_rate",
    reverse: true,
    format: {
      pass_rate: (d) => sparkBar(1)(d / 100),
    },
    align: {
      pass_rate: "right",
    },
  }
}
