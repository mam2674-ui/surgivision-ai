import { useState } from "react";

const references = [
  {
    name: "Laparoscopic Roux-en-Y gastric bypass",
    dataset: "MultiBypass140",
    url: "https://github.com/CAMMA-public/MultiBypass140",
    phases: ["Gastric pouch creation", "Gastrojejunal anastomosis", "Jejunojejunal anastomosis"],
  },
  {
    name: "Cataract surgery",
    dataset: "Cataract-1K",
    url: "https://github.com/Negin-Ghamsarian/Cataract-1K",
    phases: ["Incision", "Capsulorhexis", "Hydrodissection", "Phacoemulsification", "Irrigation and aspiration", "Intraocular lens implantation"],
  },
];

export function ProcedureLibrary({ cholecystectomyPhases, onAnalyze }: {
  cholecystectomyPhases: string[];
  onAnalyze: () => void;
}) {
  const procedures = [
    {
      name: "Laparoscopic cholecystectomy",
      dataset: "Cholec80",
      url: "https://camma.unistra.fr/datasets/",
      phases: cholecystectomyPhases,
    },
    ...references,
  ];
  const [selected, setSelected] = useState(0);
  const procedure = procedures[selected];
  const supportsAnalysis = selected === 0;

  return <section className="panel procedure-library">
    <label htmlFor="procedure-select">Procedure</label>
    <select id="procedure-select" value={selected} onChange={event => setSelected(Number(event.target.value))}>
      {procedures.map((entry, index) => <option key={entry.dataset} value={index}>{entry.name}</option>)}
    </select>
    <div className="procedure-details">
      <h2>{procedure.name}</h2>
      <p>{supportsAnalysis
        ? "Video analysis requires a compatible Cholec80 checkpoint and a connected inference API."
        : "Reference only. Video analysis for this procedure requires a separately trained and evaluated model."}</p>
      <h3>{supportsAnalysis ? "Model phase labels" : "Example annotated phases"}</h3>
      <ul>{procedure.phases.map(phase => <li key={phase}>{phase}</li>)}</ul>
      <a href={procedure.url} target="_blank" rel="noreferrer">View {procedure.dataset} dataset</a>
      {supportsAnalysis && <button className="button button-primary" onClick={onAnalyze}>Open video analysis</button>}
    </div>
  </section>;
}
