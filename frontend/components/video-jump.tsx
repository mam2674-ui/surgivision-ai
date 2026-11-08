import {useState} from "react";
import {Button} from "@/components/ui/button";

type Segment = {phase: string; start_seconds: number};

export function VideoJump({duration, segments, formatTime, seek}: {
  duration: number;
  segments: Segment[];
  formatTime: (seconds: number) => string;
  seek: (seconds: number) => void;
}) {
  const [timestamp, setTimestamp] = useState("");
  const [error, setError] = useState("");

  return <div className="video-jump">
    <form onSubmit={event => {
      event.preventDefault();
      const value = timestamp.trim();
      const valid = /^(?:\d+:)?\d+:[0-5]\d$|^\d+(?:\.\d+)?$/.test(value);
      const seconds = value.split(":").reduce((total, part) => total * 60 + Number(part), 0);
      if (!valid || seconds > duration) {
        setError(`Enter a time from 00:00 to ${formatTime(duration)}.`);
        return;
      }
      setError("");
      seek(seconds);
    }}>
      <label htmlFor="jump-timestamp">Jump to time</label>
      <input id="jump-timestamp" placeholder="mm:ss or seconds" value={timestamp}
        disabled={!duration} aria-invalid={!!error} aria-describedby="jump-help"
        onChange={event => { setTimestamp(event.target.value); setError(""); }}/>
      <Button type="submit" disabled={!duration || !timestamp.trim()}>Go</Button>
    </form>
    <label className="jump-phase">
      Jump to phase
      <select disabled={!segments.length} value="" onChange={event => {
        const segment = segments[Number(event.target.value)];
        if (event.target.value && segment) seek(segment.start_seconds);
      }}>
        <option value="">{segments.length ? "Choose a detected phase" : "No detected phases yet"}</option>
        {segments.map((segment, index) => <option key={segment.start_seconds} value={index}>
          {segment.phase} · {formatTime(segment.start_seconds)}
        </option>)}
      </select>
    </label>
    <p id="jump-help" role="status">{error || (!duration ? "Upload a video to jump to a specific part." : "")}</p>
  </div>;
}
