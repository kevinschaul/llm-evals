import { useEffect } from "react"

export interface DrawerField {
  label: string
  value: string
}

interface Props {
  title: string
  badge?: { label: string; kind: "pass" | "fail" | "none" }
  fields: DrawerField[]
  onClose: () => void
}

export default function DetailDrawer({ title, badge, fields, onClose }: Props) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose()
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [onClose])

  return (
    <div className="drawer" role="dialog" aria-label={title}>
      <div className="drawer-head">
        <h3>{title}</h3>
        {badge && <span className={`badge badge-${badge.kind}`}>{badge.label}</span>}
        <button className="drawer-close" onClick={onClose} aria-label="Close">
          ✕
        </button>
      </div>
      {fields
        .filter((f) => f.value !== "")
        .map((f) => (
          <div className="drawer-field" key={f.label}>
            <div className="drawer-field-label">{f.label}</div>
            <pre>{f.value}</pre>
          </div>
        ))}
    </div>
  )
}
