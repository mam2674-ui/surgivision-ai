import type {Metadata} from "next";
import "./globals.css";
export const metadata:Metadata={title:"SurgiVision AI — Surgical workflow research",description:"Understanding surgical workflow through computer vision. Research and educational software only."};
export default function RootLayout({children}:{children:React.ReactNode}){return <html lang="en"><body>{children}</body></html>}
