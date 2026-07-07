import { useState } from "react"

const MAX_LINES_PER_FILE = 2000

interface DiffFile {
  name: string
  additions: number
  deletions: number
  lines: string[]
}

export function parseDiff(diff: string): DiffFile[] {
  const files: DiffFile[] = []
  let current: DiffFile | null = null

  for (const line of diff.split("\n")) {
    if (line.startsWith("diff --git")) {
      if (current) files.push(current)
      const match = line.match(/diff --git a\/(.*?) b\/(.*)/)
      current = {
        name: match ? match[2] : "unknown",
        additions: 0,
        deletions: 0,
        lines: [],
      }
      continue
    }
    if (!current) continue
    current.lines.push(line)
    if (line.startsWith("+") && !line.startsWith("+++")) current.additions++
    else if (line.startsWith("-") && !line.startsWith("---")) current.deletions++
  }
  if (current) files.push(current)
  return files
}

function lineClass(line: string): string {
  if (line.startsWith("+++") || line.startsWith("---")) return "hunk"
  if (line.startsWith("@@")) return "hunk"
  if (line.startsWith("+")) return "add"
  if (line.startsWith("-")) return "del"
  return ""
}

function FileDiff({ file }: { file: DiffFile }) {
  // Render the (potentially huge) line list only once expanded
  const [open, setOpen] = useState(false)
  const truncated = file.lines.length > MAX_LINES_PER_FILE

  return (
    <details
      className="diff-file"
      open={open}
      onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}
    >
      <summary>
        <span>{file.name}</span>
        <span className="diff-stat-add">+{file.additions}</span>
        <span className="diff-stat-del">−{file.deletions}</span>
      </summary>
      {open && (
        <pre className="diff-lines">
          {file.lines.slice(0, MAX_LINES_PER_FILE).map((line, i) => (
            <span key={i} className={`diff-line ${lineClass(line)}`}>
              {line || " "}
            </span>
          ))}
          {truncated && (
            <span className="diff-truncated">
              … {file.lines.length - MAX_LINES_PER_FILE} more lines not shown
            </span>
          )}
        </pre>
      )}
    </details>
  )
}

export default function DiffViewer({ diff }: { diff: string }) {
  const files = parseDiff(diff)

  if (files.length === 0) {
    return <p>No diff captured for this run.</p>
  }

  const additions = files.reduce((a, f) => a + f.additions, 0)
  const deletions = files.reduce((a, f) => a + f.deletions, 0)

  return (
    <div>
      <p style={{ fontSize: "0.85em" }}>
        {files.length} file{files.length === 1 ? "" : "s"} changed,{" "}
        <span className="diff-stat-add">+{additions}</span>{" "}
        <span className="diff-stat-del">−{deletions}</span>
      </p>
      {files.map((f) => (
        <FileDiff key={f.name} file={f} />
      ))}
    </div>
  )
}
