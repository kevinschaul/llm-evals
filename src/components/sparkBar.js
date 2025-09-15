import * as htl from "htl"

export default function sparkBar(max) {
  return (x) => htl.html`
    <div style="
      display: flex;
      align-items: center;
      gap: 4px;
      font-weight: bold;
      padding-left: 10px;
      ">
      <div style="flex-grow: 1; text-align: right;">${(x * 100).toFixed(0)}%</div>
      <div style="width: 60px; position: relative; height: 8px;">
        <div style="
          position: absolute;
          background: #ddd;
          top: 0;
          bottom: 0;
          left: 0;
          width: 100%;
          z-index: -1;
          ">
        </div>
        <div style="
          position: absolute;
          background: var(--theme-green);
          top: 0;
          bottom: 0;
          left: 0;
          width: ${(100 * x) / max}%;
          z-index: -1;
          ">
        </div>
      </div>
    </div>`
}
