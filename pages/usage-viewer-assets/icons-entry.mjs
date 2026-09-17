// Build-only entry for the local Lucide 0.378.0 subset. See docs/usage-viewer-security.md.
import {
  createIcons,
  AlertTriangle,
  Database,
  FileText,
  Gauge,
  Info,
  RefreshCw,
} from "lucide"

window.lucide = {
  createIcons: () =>
    createIcons({
      icons: { AlertTriangle, Database, FileText, Gauge, Info, RefreshCw },
    }),
}
