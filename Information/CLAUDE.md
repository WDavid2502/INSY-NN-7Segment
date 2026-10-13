# CLAUDE.md – Neuronales Netz from Scratch & 7-Segment-Decoder

## Projektkontext
- Schulprojekt HTL, Fach INSY, 5. Klasse.
- Grundlage: `4AHITS_NN_from_scratch.pdf` (Theorie: Forward/Backward Pass, Sigmoid, MSE, SGD; Aufgaben Part 1–6).
- Ziel: Ein selbst geschriebenes neuronales Netz lernt einen **7-Segment-Decoder mit 3 Eingangsbits (Ziffern 0–7)**.
  In einer HTML-Seite gibt man z. B. `001` ein, das Netz reagiert live und die Anzeige zeigt „1“.
- Abweichung vom PDF (bewusst, Wunsch des Users): 3 statt 4 Eingangsbits → Netz 3-16-7.

## Ordnerstruktur
- **`Information/`** – alles Lesbare (Doku, Plan, PDF). **`Code/`** – nur Code und vom Code erzeugte Exporte.
- `CLAUDE.md` im Hauptordner ist nur ein Verweis (`@Information/CLAUDE.md`), damit diese Datei automatisch geladen wird.
- Neue Doku-Dateien → `Information/`, neuer Code → `Code/`. Im Hauptordner nichts ablegen.

## Dateien
| Datei | Zweck |
|---|---|
| `Information/PLAN.md` | Freigegebener Plan |
| `Information/token.md` | Geschätzter Token-Verbrauch pro Aufgabe |
| `Information/DOKUMENTATION.qmd` | Doku-Quelle (Quarto) inkl. Gedankengänge; rendern mit `/Applications/quarto/bin/quarto render Information/DOKUMENTATION.qmd` → `DOKUMENTATION.html` (abgabereif) |
| `Information/bilder/` | Screenshots für die Doku |
| `Information/4AHITS_NN_from_scratch.pdf` | Aufgabenstellung / Theorie |
| `Code/nn.py` | Neuron, Layer, NeuralNetwork (PDF Part 1) |
| `Code/train_xor.py` | XOR trainieren, exportieren, laden, testen (PDF Part 2, Step 1–2) |
| `Code/train_7seg.py` | 7-Segment (3 Bit) trainieren, exportieren, laden, testen (PDF Part 2, Step 3–4) |
| `Code/index.html` | Blueprint-Seite: Decoder, „Blick ins Netz“ mit Hover-Infos + Schritt-Modus, Wahrheitstabelle 0–7, Training im Zeitraffer (lädt `seven_segment_weights.js` + `seven_segment_history.js`) |
| `Code/seven_segment_history.js` | Trainingsverlauf: 55 Gewichts-Schnappschüsse + Fehlerkurve (von `train_7seg.py` erzeugt) |
| `Code/*.json / *.txt / *.h / *.js` | Exporte des trainierten Netzes |

Die Trainingsskripte wechseln selbst in `Code/` – sie können von überall gestartet werden (z. B. `python3 Code/train_7seg.py`).

## Git / Abgabe
- Öffentliches GitHub-Repo (seit 2026-10-13): https://github.com/WDavid2502/INSY-NN-7Segment (Branch `main`). Lesbare Doku: `Information/DOKUMENTATION.md` (gfm-Ausgabe von Quarto).
- README liegt in `.github/README.md` (Hauptordner bleibt leer). `token.md` im Hauptordner ist eine alte Kopie → per `.gitignore` ausgeschlossen.
- Nach Doku-Änderungen: erst `quarto render`, dann committen + pushen.

## Regeln
1. **Der User muss immer verstehen, was passiert – aber nur ungefähr.** Während der Arbeit laufend erklären, *was* gerade
   gemacht wird und *grob wie es funktioniert* (auch wie das neuronale Netz dabei arbeitet) – vor jedem Ausführen, nicht erst am Ende.
   Nichts „magisch“ lösen.
   **Detailgrad:** das große Bild statt Mathematik. Keine Formeln oder Herleitungen (z. B. *wie* Sigmoid rechnet),
   sondern *wozu* etwas da ist und was ungefähr passiert (z. B. „Sigmoid drückt jeden Wert zwischen 0 und 1“).
   Mehr Tiefe nur, wenn der User nachfragt.
   **Terminal-Befehle:** vorher in einfachen Worten erklären, was sie tun; einfach halten, nicht mehrere Befehle
   mit `&&` zu einer langen Zeile verketten.
2. **Das PDF ist das Grundgerüst** für das neuronale Netz – der Code baut darauf auf (Klassen, Methoden, Parameter).
3. **Code auf Englisch, Kommentare auf Deutsch.** Kommentardichte mittel: Docstring pro Funktion, Kommentare bei nicht offensichtlichen Stellen.
4. **Nur Python-Standardbibliothek.** Vor jeder neuen Bibliothek/Abhängigkeit den User fragen.
5. **Claude führt alles selbst aus.** Alles muss in VS Code funktionieren, keine zusätzliche Software herunterladen.
6. **Bei Unsicherheit fragen**, sonst selbstständig arbeiten.
7. **Token-Protokoll:** Nach jeder Aufgabe eine Zeile in `token.md` ergänzen (Datum, Aufgabe, geschätzte Tokens,
   größter Posten). Immer als Schätzung kennzeichnen – Claude hat keinen exakten Zähler.

## Fehler & Lösungsansätze
- **HTML kann JSON nicht per `fetch()` laden**, wenn die Seite als `file://` geöffnet wird (Browser blockiert das).
  → Lösung: Gewichte zusätzlich als JavaScript-Datei (`seven_segment_weights.js`) exportieren und per `<script src>` einbinden.
- **HTML in VS Code anzeigen:** Der VS-Code-„Simple Browser“ braucht eine `http://`-Adresse.
  → Lösung: `python3 -m http.server 8000 --directory Code` im Hauptordner starten (Standardbibliothek, kein Download),
  dann in VS Code `Cmd+Shift+P` → „Simple Browser: Show“ → `http://localhost:8000/index.html`.
- **Kein Node.js installiert** → JavaScript mit der eingebauten macOS-Engine ausführen:
  `osascript -l JavaScript datei.js` (kein Download). Damit wurde geprüft: Der JS-Forward-Pass mit
  `seven_segment_weights.js` liefert dieselben Rohwerte wie `python3 train_7seg.py`.

## Erledigt
- Netz, XOR und 7-Segment trainiert und getestet (alle Ziffern korrekt, Export = Import), HTML-Decoder läuft.
- Gewichte visualisiert, Teil 1: Neuronen + Verbindungen in `index.html`, Dicke = Stärke, Farbe = Vorzeichen,
  Sichtbarkeit = wie stark die Verbindung bei der aktuellen Eingabe reagiert.
- Gewichte visualisieren, Teil 2 (2026-10-13): Blueprint-Design, Training im Zeitraffer (Regler + Fehlerkurve,
  ganze Seite rechnet mit dem gewählten Zeitpunkt), Wahrheitstabelle, Hover-Infos, Schritt-für-Schritt-Modus.
  `nn.train()` hat dafür einen optionalen `on_epoch`-Callback. Neu trainieren mit Seed 42 ergibt identische Gewichte.

## Spätere Ideen
- PDF Part 3–6: TensorFlow-Vergleich, Forward-Inferenz in C (.txt / .h), ESP32 mit echter 7-Segment-Anzeige.
