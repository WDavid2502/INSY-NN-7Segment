# INSY – Neuronales Netz from Scratch: 7-Segment-Decoder (3 Bit)

Schulprojekt INSY, 5. Klasse, HTL Braunau – David Weinberger

Ein selbst programmiertes neuronales Netz (Python, nur Standardbibliothek) lernt einen 7-Segment-Decoder
für die Ziffern 0–7. Eine interaktive Webseite zeigt live, wie das Netz rechnet und wie es gelernt hat.

## 📄 Dokumentation

- **[Information/DOKUMENTATION.html](../Information/DOKUMENTATION.html)** – fertige Dokumentation
  (herunterladen → im Browser öffnen; GitHub zeigt HTML nur als Quelltext)
- [Information/DOKUMENTATION.qmd](../Information/DOKUMENTATION.qmd) – Quarto-Quelle (inkl. Gedankengänge & Entscheidungen)
- [Information/token.md](../Information/token.md) – geschätzter KI-Verbrauch pro Aufgabe

## Ordner

| Ordner | Inhalt |
|---|---|
| `Code/` | Netz (`nn.py`), Trainingsskripte, Webseite (`index.html`), Exporte |
| `Information/` | Angabe (PDF), Plan, Dokumentation, Projektregeln |

## Ausführen

```bash
python3 Code/train_7seg.py                       # trainieren + testen
python3 -m http.server 8000 --directory Code     # dann http://localhost:8000/index.html
quarto render Information/DOKUMENTATION.qmd      # Doku neu erzeugen
```
