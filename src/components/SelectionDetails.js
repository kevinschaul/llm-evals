import * as htl from "htl"

export default function getSelectionDetailsConfig() {
  return {
    getRows: (key, value) => {
      const valueStr = String(value)
      if (key === "body" || valueStr.length > 200) return 15
      if (key === "subject" || valueStr.length > 100) return 3
      if (key === "result" || valueStr.length > 50) return 5
      return 2
    },
    coreKeys: ["provider_id", "test", "result", "error", "duration_ms", "passed"]
  }
}