# Dokumentation: Neuronales Netz from Scratch – 7-Segment-Decoder (3 Bit)

**Fach:** INSY, 5. Klasse, HTL Braunau
**Autor:** David Weinberger (umgesetzt mit Claude Code als KI-Assistent)
**Stand:** 13.10.2026

---

## 1. Angabe

Grundlage ist das Aufgabenblatt `4AHITS_NN_from_scratch.pdf` (liegt in `Information/`). Es verlangt:

- **Theorie:** Forward Pass, Backward Pass (Backpropagation), Sigmoid als Aktivierungsfunktion,
  mittlerer quadratischer Fehler (MSE) als Fehlermaß, Training mit Stochastic Gradient Descent (SGD).
- **Part 1:** Ein neuronales Netz selbst programmieren, ohne fertige KI-Bibliothek. Es besteht aus den Klassen
  `Neuron`, `Layer` und `NeuralNetwork` und hat Methoden für Training, Vorhersage und Export/Import.
- **Part 2:** Das Netz zuerst auf **XOR** trainieren und testen, danach auf einen **7-Segment-Decoder**.
  Das trainierte Netz soll exportiert, wieder geladen und dann noch einmal getestet werden.
- **Part 3–6** (noch offen): Vergleich mit TensorFlow, Vorhersage in C, Einsatz auf einem ESP32 mit echter Anzeige.

### Eigene Erweiterung / Abweichung
- Der Decoder bekommt **3 Eingangsbits statt 4** und stellt damit nur die Ziffern 0–7 dar. Das Netz ist deshalb
  **3-16-7** aufgebaut: 3 Eingänge, 16 versteckte Neuronen und 7 Ausgänge für die Segmente a–g.
- Zusätzlich gibt es eine **interaktive Webseite**. Dort gibt man z. B. `001` ein, und das trainierte Netz rechnet
  live, welche Segmente leuchten.

---

## 2. Plattform & Werkzeuge

| Bereich | Verwendet |
|---|---|
| Betriebssystem | macOS 27 |
| Editor | Visual Studio Code |
| KI-Assistent | Claude Code (VS-Code-Erweiterung), Modell Claude Opus |
| Programmiersprache Netz | Python 3.13, **nur Standardbibliothek** (`random`, `math`, `json`, `os`) |
| Webseite | HTML, CSS, JavaScript ohne Framework, Grafiken als SVG |
| Webserver zum Anzeigen | `python3 -m http.server` (in Python eingebaut) |
| Browser | VS Code „Simple Browser“ (geht auch mit Safari) |
| JS-Test ohne Browser | `osascript -l JavaScript`, die eingebaute JavaScript-Engine von macOS |

Für das Projekt wurde **keine zusätzliche Software** installiert und keine externe Bibliothek verwendet.

---

## 3. Ordnerstruktur

```
5 Klasse/
├── CLAUDE.md                    Verweis auf Information/CLAUDE.md
├── Information/                 alles zum Lesen
│   ├── 4AHITS_NN_from_scratch.pdf   Angabe
│   ├── CLAUDE.md                Projektregeln für den KI-Assistenten
│   ├── PLAN.md                  freigegebener Plan
│   ├── token.md                 geschätzter KI-Verbrauch pro Aufgabe
│   └── DOKUMENTATION.md         dieses Dokument
└── Code/                        Code und erzeugte Dateien
    ├── nn.py                    das neuronale Netz (Part 1)
    ├── train_xor.py             XOR-Training (Part 2, Step 1–2)
    ├── train_7seg.py            7-Segment-Training (Part 2, Step 3–4)
    ├── index.html               interaktive Webseite
    ├── xor_net.json/.txt/.h     Exporte XOR-Netz
    ├── seven_segment_net.json/.txt/.h   Exporte 7-Segment-Netz
    ├── seven_segment_weights.js         Gewichte für die Webseite
    └── seven_segment_history.js         Trainingsverlauf für den Zeitraffer
```

---

## 4. Das neuronale Netz (`nn.py`)

Die Struktur folgt dem PDF:

- **`Neuron`** hat Gewichte und einen Bias (eine Grundneigung, die das Neuron auch ohne Eingabe hat).
  Es rechnet aus seinen Eingaben einen Wert und drückt diesen mit Sigmoid zwischen 0 und 1.
- **`Layer`** fasst mehrere Neuronen zu einer Schicht zusammen.
- **`NeuralNetwork`** verbindet die Schichten. Seine Methoden sind:
  - `forward` / `predict`: Die Eingabe läuft Schicht für Schicht durch das Netz (Forward Pass).
  - `backward`: Der Fehler wird von hinten nach vorne zurückgegeben, und jedes Gewicht wird ein kleines Stück
    korrigiert (Backpropagation).
  - `train`: Wiederholt das über viele Epochen. Die Beispiele kommen dabei jedes Mal in zufälliger Reihenfolge (SGD).
  - Export und Import: `dump_json`, `load`, `dump_txt`, `dump_initfile` (eine C-Header-Datei für später) und
    `dump_js` (für die Webseite).
- Das Netz kann auch `tanh` und `relu` als Aktivierung verwenden. Benutzt wird aber Sigmoid.

**Wie das Lernen ungefähr funktioniert:** Am Anfang sind alle Gewichte zufällig, und das Netz rät nur.
Bei jedem Beispiel vergleicht es seine Ausgabe mit der richtigen Antwort. Dann schiebt es alle Gewichte ein
kleines Stück in die Richtung, in der der Fehler kleiner wird. Nach tausenden Wiederholungen stimmen die Ausgaben.

---

## 5. Training & Ergebnisse

### XOR (`train_xor.py`)
- Netz **2-2-1**, 20 000 Epochen, Lernrate 0.1, Seed 42
- Alle 4 Fälle sind richtig. Das exportierte und wieder geladene Netz liefert dieselben Werte.

### 7-Segment-Decoder (`train_7seg.py`)
- Netz **3-16-7**, 20 000 Epochen, Lernrate 0.1, Seed 42
- Trainingsdaten: die 8 Ziffern 0–7 mit ihren Segmentmustern a–g, genommen aus der Tabelle im PDF.
- Ein Ausgabewert über 0.5 bedeutet „Segment leuchtet“.

| Eingabe | Ziffer | Ergebnis |
|---|---|---|
| 000 – 111 | 0 – 7 | **8 / 8 richtig** |

Die Rohwerte sind sehr eindeutig: Die Segmente, die leuchten sollen, liegen bei 0.98–1.00, die übrigen bei 0.00–0.03.

**Lernverlauf** (aufgezeichnet für den Zeitraffer):

| Zeitpunkt | Fehler | richtig erkannt |
|---|---|---|
| vor dem Training | 0.97 | 0 / 8 |
| Epoche 66 | 0.59 | 1 / 8 |
| Epoche 356 | 0.30 | 6 / 8 |
| Epoche 1 907 | 0.02 | 8 / 8 |
| Epoche 20 000 | 0.0005 | 8 / 8 |

Weil der Seed fest ist, kommen bei jedem neuen Training **genau dieselben Gewichte** heraus. Das wurde mit einem
Byte-Vergleich der JSON-Datei geprüft.

---

## 6. Die Webseite (`index.html`)

Die Seite ist im Stil einer technischen Zeichnung gestaltet („Blueprint“) und hat vier Bereiche:

1. **Fig. 1 – Decoder:** Man gibt 3 Bits ein oder klickt auf die drei Knöpfe. Die 7-Segment-Anzeige leuchtet, und
   darunter stehen die 7 Rohwerte des Netzes mit Balken.
2. **Fig. 2 – Blick ins Netz:** Alle Neuronen (Kreise) und Gewichte (Linien) sind als Grafik zu sehen.
   - Cyan heißt verstärkt, Koralle heißt bremst. Die Linienstärke zeigt, wie groß das Gewicht ist.
   - **Hover-Infos:** Fährt man mit der Maus über ein Neuron oder eine Linie, sieht man die Aktivität, den Bias,
     das Gewicht und wie viel Signal gerade fließt.
   - **Schritt-für-Schritt-Modus:** Der Forward Pass wird in 4 Schritten animiert und jeweils kurz erklärt.
3. **Fig. 3 – Wahrheitstabelle:** Alle 8 Eingaben mit Soll-Anzeige, Netz-Anzeige und einem Sicherheitsbalken.
4. **Fig. 4 – Training im Zeitraffer:** Ein Regler mit Abspielknopf führt durch 55 Momentaufnahmen des Trainings.
   Dazu gibt es die Fehlerkurve. Die ganze Seite rechnet dann mit den Gewichten des gewählten Zeitpunkts, so
   sieht man direkt, wie das Netz lernt.

**Technische Lösung:** Die Seite enthält keine fertige Lösungstabelle. Sie rechnet den Forward Pass selbst in
JavaScript, mit den Gewichten, die Python exportiert hat. Es wurde geprüft, dass die JavaScript-Berechnung
dieselben Werte liefert wie die Berechnung in Python.

---

## 7. Anleitung zum Ausführen

Im Projektordner im VS-Code-Terminal:

```
python3 Code/train_7seg.py
```
Damit wird das Netz trainiert und getestet, und alle Exporte werden neu erzeugt.

```
python3 -m http.server 8000 --directory Code
```
Damit wird ein kleiner lokaler Webserver gestartet. Danach in VS Code `Cmd+Shift+P` drücken,
„Simple Browser: Show“ wählen und `http://localhost:8000/index.html` öffnen.

---

## 8. Probleme & Lösungen

| Problem | Lösung |
|---|---|
| Wird die Seite direkt als Datei (`file://`) geöffnet, darf sie keine JSON-Datei laden. | Die Gewichte werden zusätzlich als `.js`-Datei exportiert und per `<script>` eingebunden. |
| Der VS-Code-Simple-Browser braucht eine `http://`-Adresse. | Ein lokaler Server mit `python3 -m http.server`. |
| Node.js ist nicht installiert, um JavaScript zu testen. | Die eingebaute macOS-Engine `osascript -l JavaScript` wird verwendet. |
| Die Tabelle ragte bei schmalem Fenster aus der Karte hinaus. | Feste Spaltenbreiten und ein früheres Umbrechen des Layouts. |
| Für den Zeitraffer fehlten Zwischenstände des Trainings. | `train()` bekommt einen optionalen Callback, der Schnappschüsse speichert, ohne das Lernen zu verändern. |

---

## 9. Arbeitsweise mit KI

Das Projekt wurde zusammen mit **Claude Code** umgesetzt. Die Regeln dafür stehen in `Information/CLAUDE.md`:
Das PDF ist das Grundgerüst, es wird nur die Standardbibliothek verwendet, der Code ist auf Englisch und die
Kommentare auf Deutsch. Vor jedem Schritt erklärt die KI, was sie macht. Der geschätzte Verbrauch wird in
`Information/token.md` protokolliert.

---

## 10. Ausblick

- **Part 3:** Dasselbe Netz mit TensorFlow bauen und die Ergebnisse vergleichen
- **Part 4–5:** Forward Pass in C mit den exportierten `.h`-Dateien
- **Part 6:** ESP32 mit einer echten 7-Segment-Anzeige
