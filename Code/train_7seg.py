"""
PDF Part 2, Step 3–4: 7-Segment-Decoder trainieren, exportieren, laden und testen.
Abweichung vom PDF: 3 Eingangsbits (Ziffern 0–7) statt 4 -> Architektur 3-16-7.
Segment-Codes a–g stammen aus der Tabelle im PDF.
"""

import json
import os
import random

from nn import NeuralNetwork

SEED = 42
EPOCHS = 20000
LEARNING_RATE = 0.1
HIDDEN_NEURONS = 16
TOLERANCE = 1e-9

#          Input      Segmente  a  b  c  d  e  f  g
DATASET = [
    ([0, 0, 0], [1, 1, 1, 1, 1, 1, 0]),  # 0
    ([0, 0, 1], [0, 1, 1, 0, 0, 0, 0]),  # 1
    ([0, 1, 0], [1, 1, 0, 1, 1, 0, 1]),  # 2
    ([0, 1, 1], [1, 1, 1, 1, 0, 0, 1]),  # 3
    ([1, 0, 0], [0, 1, 1, 0, 0, 1, 1]),  # 4
    ([1, 0, 1], [1, 0, 1, 1, 0, 1, 1]),  # 5
    ([1, 1, 0], [1, 0, 1, 1, 1, 1, 1]),  # 6
    ([1, 1, 1], [1, 1, 1, 0, 0, 0, 0]),  # 7
]


def to_bits(values: list[float]) -> list[int]:
    """Schwelle 0.5: Werte über 0.5 gelten als Segment an (1), sonst aus (0)."""
    return [1 if v > 0.5 else 0 for v in values]


def snapshot_epochs(epochs: int, count: int = 60) -> set[int]:
    """Epochen für Schnappschüsse: am Anfang dicht, später seltener (dort ändert sich weniger)."""
    return {0} | {round(epochs ** (k / (count - 1))) for k in range(count)}


def rounded_layers(net: NeuralNetwork) -> list:
    """Alle Gewichte/Bias auf 4 Stellen gerundet – reicht für die Grafik und hält die Datei klein."""
    return [
        [{"weights": [round(w, 4) for w in n["weights"]], "bias": round(n["bias"], 4)} for n in layer]
        for layer in net.to_dict()["layers"]
    ]


def dump_history_js(path: str, snapshots: list, loss_curve: list) -> None:
    """Speichert den Trainingsverlauf als JS-Datei (const HISTORY = {...}) für die HTML-Seite."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("const HISTORY = ")
        json.dump({"snapshots": snapshots, "loss_curve": loss_curve}, f, separators=(",", ":"))
        f.write(";\n")


def main() -> None:
    # In den Ordner dieses Skripts wechseln, damit die Exporte immer in Code/ landen
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    random.seed(SEED)
    inputs = [x for x, _ in DATASET]
    targets = [y for _, y in DATASET]

    net = NeuralNetwork([3, HIDDEN_NEURONS, 7])
    # Schnappschüsse für den "Training im Zeitraffer"-Regler in index.html
    wanted = snapshot_epochs(EPOCHS)
    snapshots = [{"epoch": 0, "loss": None, "layers": rounded_layers(net)}]
    loss_curve = []

    def record(epoch: int, loss: float) -> None:
        """Wird nach jeder Epoche aufgerufen: Fehlerkurve (jede 50.) und Schnappschüsse sammeln."""
        if epoch == 1 or epoch % 50 == 0:
            loss_curve.append([epoch, round(loss, 6)])
        if epoch in wanted:
            snapshots.append({"epoch": epoch, "loss": round(loss, 6), "layers": rounded_layers(net)})

    print(f"=== Training 7-Segment (3-{HIDDEN_NEURONS}-7) ===")
    net.train(inputs, targets, epochs=EPOCHS, learning_rate=LEARNING_RATE, on_epoch=record)

    # Alle Exportformate aus PDF Part 1 + JS-Datei für die HTML-Seite
    net.dump_json("seven_segment_net.json")
    net.dump_txt("seven_segment_net.txt")
    net.dump_initfile("seven_segment_net.h", prefix="SEG7")
    net.dump_js("seven_segment_weights.js")
    dump_history_js("seven_segment_history.js", snapshots, loss_curve)
    print(f"Trainingsverlauf: {len(snapshots)} Schnappschüsse gespeichert")

    print("\n=== Test: geladenes Netz (aus JSON) ===")
    loaded = NeuralNetwork.load("seven_segment_net.json")
    all_ok = True
    for digit, (x, y) in enumerate(DATASET):
        trained_out = net.predict(x)
        loaded_out = loaded.predict(x)
        same = all(abs(a - b) < TOLERANCE for a, b in zip(trained_out, loaded_out))
        correct = to_bits(loaded_out) == y
        all_ok &= same and correct
        raw = " ".join(f"{v:.2f}" for v in loaded_out)
        print(f"{digit}  {''.join(map(str, x))} -> {to_bits(loaded_out)} "
              f"erwartet {y} | {'OK' if correct else 'FALSCH'} | "
              f"Export/Import {'gleich' if same else 'UNTERSCHIED'} | roh {raw}")

    print("\nErgebnis:", "alles korrekt" if all_ok else "es gibt Fehler")


if __name__ == "__main__":
    main()
