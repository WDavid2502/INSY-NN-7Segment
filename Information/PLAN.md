# Plan: Neuronales Netz nach dem PDF und interaktiver 7-Segment-Decoder (3 Bit)

**Ziel:** Ich setze das Netz um, wie es in `4AHITS_NN_from_scratch.pdf` gefordert ist. Danach gibt man in einer HTML-Seite z. B. `001` ein, das Netz reagiert live, und die 7-Segment-Anzeige zeigt „1“.
**Ordner:** `…/INSY/5 Klasse/`. Claude führt alles selbst aus (Python 3.13), und es gibt keine Downloads.

## Dateien
0. **`PLAN.md`**: eine Kopie dieses Plans im Projektordner, als erster Schritt
1. **`CLAUDE.md`**: Projektkontext, Regeln, Fehler und Lösungsansätze, spätere Ideen. Die Regeln sind:
   - Der User muss immer verstehen, was passiert
   - Das PDF ist das Grundgerüst für das neuronale Netz, und der Code baut darauf auf
   - Code auf Englisch, Kommentare auf Deutsch, Kommentardichte mittel
   - Nur die Standardbibliothek, vor jeder neuen Bibliothek fragen
   - Claude führt alles selbst aus, alles muss in VS Code gehen
   - Bei Unsicherheit fragen, sonst selbstständig arbeiten
2. **Neuronales Netz**: Grundgerüst nach PDF Part 1 (Klassen, Methoden, Schritte 1–6) und Part 2 (XOR und 7-Segment trainieren, exportieren, laden, testen), mit den Architekturen und Trainingsparametern aus dem PDF. Einzige Abweichung: Der 7-Segment-Decoder bekommt **3 Eingangsbits (Ziffern 0–7)**, damit ist er 3-16-7.
3. **`index.html`**: Eingabefeld für `001` (oder 3 klickbare Bits). Das trainierte Netz rechnet live in JavaScript, eine SVG-7-Segment-Anzeige zeigt das Ergebnis, daneben die Rohwerte a–g. Die Gewichte kommen aus dem Export des Netzes.

4. **„Blick ins Netz“ in `index.html`** (erledigt): Neuronen und Gewichte als Grafik, die live auf die Eingabe reagiert.
5. **Ordnerstruktur** (erledigt): `Information/` für CLAUDE.md, PLAN.md, token.md und das PDF, `Code/` für den Code und die Exporte.

## Später (in CLAUDE.md notiert)
- Zeigen, wie sich die Gewichte beim Training verändern
- PDF Part 3–6 (TensorFlow, C, ESP32)

## Verifikation
- Die Trainings führe ich aus. Die Ergebnisse müssen nach der Schwelle 0.5 korrekt sein, und Export und Import müssen identische Werte liefern
- Die HTML-Seite öffne ich im VS Code Simple Browser. Wenn das nicht klappt, schauen wir gemeinsam weiter
